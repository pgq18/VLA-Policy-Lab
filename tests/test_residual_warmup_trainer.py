"""DSRL-only execution with independent residual learning, using CPU node fixtures.

These are contract/regression checks, not robot task success benchmarks.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pressb.runtime.trainer import GenericTrainer, TrainerConfig
from test_rl_nodes_trainer import Inference, Simulation, StateOnlySimulation, config, metadata


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def warmup_settings():
    return dict(
        learner=dict(hidden_dim=8, batch_size=2, num_qs=2, num_subset=2,
                     residual_indices=[0, 2], residual_scale=[.02, .03],
                     policy_lr=.001, q_lr=.003, policy_freq=2),
        replay_capacity=64, learning_starts=0, utd=1., max_updates_per_wait=2,
        loss=dict(target_mode="behavior", actor_q_weight=1., actor_l2_weight=1.,
                  entropy_weight=0.),
    )


def module_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def assert_module_equal(module, expected):
    assert module.state_dict().keys() == expected.keys()
    for key, value in module.state_dict().items():
        assert torch.equal(value, expected[key]), key


class SnapshotTrainer(GenericTrainer):
    def _negotiate(self):
        super()._negotiate()
        self.initial_main = dict(actor=module_state(self.learner.actor),
                                 qs=module_state(self.learner.qs),
                                 q_targets=module_state(self.learner.q_targets),
                                 updates=self.learner.updates, transitions=self.transitions,
                                 replay_size=len(self.replay))
        if self.residual_warmup is not None:
            learner = self.residual_warmup.learner
            self.initial_shadow = dict(actor=module_state(learner.actor),
                                       qs=module_state(learner.qs),
                                       updates=learner.updates,
                                       replay_size=len(self.residual_warmup.replay))


def build(output, *, enabled=True, steps=12, fast=False, fused=False,
          clip=False, settings=None, cfg=None, runner_type=SnapshotTrainer):
    inf_meta, sim_meta = metadata(4, 3, 3, 3)
    if clip:
        sim_meta["action"].update(lower=[-.002] * 3, upper=[.002] * 3)
    inference = Inference(inf_meta)
    simulation = StateOnlySimulation(sim_meta) if fast else Simulation(sim_meta)
    options = config(output, steps=steps) if cfg is None else deepcopy(cfg)
    if cfg is None:
        # Keep primary updates disabled in equality checks: extra optimizer
        # work can otherwise change the async primary update schedule.
        options["update"].update(utd=0., replay_capacity=64)
        options["checkpoint"].update(interval=100_000, keep=None)
        if enabled:
            options["residual_warmup"] = warmup_settings() if settings is None else settings
        if fast:
            options["rollout"].update(terminal_observation_mode="state_only_if_no_bootstrap",
                                      fused_step_reset=fused)
        if clip:
            options["rollout"]["action_bounds"] = "clip"
    observed = []
    observer = SimpleNamespace(
        on_reset=lambda rows: None,
        on_transition=lambda before, actions, after: observed.extend(
            (deepcopy(old), action.copy(), deepcopy(new))
            for old, action, new in zip(before, actions, after, strict=True)),
        close=lambda status: None,
    )
    runner = runner_type(options, inference_client=inference,
                         simulation_client=simulation, observer=observer)
    return runner, inference, simulation, observed


def arrays(replay):
    return {key: value[:len(replay)].copy() for key, value in replay.arrays.items()}


def test_shadow_updates_both_networks_without_changing_dsrl_execution(tmp_path):
    baseline, base_inf, base_sim, _ = build(tmp_path / "baseline", enabled=False)
    baseline.run()
    runner, inf, sim, _ = build(tmp_path / "shadow")
    result = runner.run()
    assert result["state"] == "completed"
    actual = [item["actions"] for req in sim.steps for item in req["items"]]
    expected = [item["actions"] for req in base_sim.steps for item in req["items"]]
    np.testing.assert_array_equal(actual, expected)
    assert runner.learner.updates == baseline.learner.updates == 0
    assert_module_equal(runner.learner.actor, baseline.learner.actor.state_dict())
    shadow = runner.residual_warmup
    assert shadow.learner.updates >= 2
    assert any(not torch.equal(value, runner.initial_shadow["actor"][key])
               for key, value in shadow.learner.actor.state_dict().items())
    assert any(not torch.equal(value, runner.initial_shadow["qs"][key])
               for key, value in shadow.learner.qs.state_dict().items())
    assert shadow.learner.actor_optimizer.state
    assert shadow.learner.q_optimizer.state
    assert not shadow.learner.alpha_optimizer.state
    assert result["residual_warmup"]["executed"] is False
    assert result["residual_warmup"]["updates"] == shadow.learner.updates
    assert not inf.contexts and not base_inf.contexts


@pytest.mark.parametrize("fast,fused", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("clip", [False, True])
def test_shadow_replay_matches_behavior_actions_and_episode_boundaries(tmp_path, fast, fused, clip):
    runner, inf, sim, observed = build(tmp_path, fast=fast, fused=fused, clip=clip)
    result = runner.run()
    shadow = runner.residual_warmup
    replay = arrays(shadow.replay)
    assert len(shadow.replay) == result["transitions"] == 12
    assert not shadow.pending
    assert np.all(replay["action"] == 0)
    actual = {float(old["vector"][3]): (old, action, new)
              for old, action, new in observed}
    # Delayed insertion intentionally does not preserve vector-slot order.
    for index, obs in enumerate(replay["obs"]):
        old, action, new = actual[float(obs[3])]
        base = replay["base_action"][index].reshape(4, 3)
        executable = np.clip(base, -.002, .002) if clip else base
        np.testing.assert_array_equal(action, executable[:2])
        assert replay["reward"][index, 0] == new["reward"]
        assert replay["discount"][index, 0] == pytest.approx(new["discount"])
        assert replay["next_obs"][index, 3] - obs[3] == pytest.approx(.001, abs=1e-6)
        done = bool(replay["terminated"][index, 0] or replay["truncated"][index, 0])
        if done:
            assert np.all(replay["next_base_action"][index] == 0)
        elif float(replay["next_obs"][index, 3]) in actual:
            _, next_action, _ = actual[float(replay["next_obs"][index, 3])]
            next_base = replay["next_base_action"][index].reshape(4, 3)
            next_executable = np.clip(next_base, -.002, .002) if clip else next_base
            np.testing.assert_array_equal(next_action, next_executable[:2])
    if clip:
        # Actor conditioning must retain raw decode; critic applies the same
        # projection as execution. No silently projected-input shift at handoff.
        assert np.any(np.abs(replay["base_action"]) > .002)
        assert result["projected_action_chunks"] > 0
    assert replay["terminated"].any() and replay["truncated"].any()
    assert not inf.contexts
    if fused:
        assert sim.fused_calls


def test_no_extra_decode_for_terminal_zero_bootstrap_warmup(tmp_path):
    runner, inf, sim, _ = build(tmp_path, steps=6, fast=True, fused=True)
    sim.meta["num_envs"] = 1  # env 0 succeeds at each first chunk.
    runner.run()
    assert len(inf.decode_calls) == len(sim.steps) == 6
    assert len(runner.residual_warmup.replay) == 6
    assert not runner.residual_warmup.pending


def test_next_actions_reuse_regular_decode_until_checkpoint_boundary(tmp_path):
    runner, inf, sim, _ = build(tmp_path, steps=8)
    runner.run()
    decoded_contexts = [item["context_id"] for request in inf.decode_calls for item in request["items"]]
    # No intermediate save; continuing slots are decoded on the next rollout.
    # The final continuation needs one additional decode, not another step.
    assert len(decoded_contexts) == len(set(decoded_contexts))
    assert len(sim.steps) == 4
    assert len(inf.decode_calls) <= len(sim.steps) + 1


def test_truncated_bootstrap_uses_terminal_observation_before_reset(tmp_path):
    settings = warmup_settings()
    settings["learner"]["bootstrap_truncated"] = True
    runner, inf, _, _ = build(tmp_path, settings=settings)
    decoded = []
    call = inf.call

    def record(path, payload=None):
        observations = ([deepcopy(inf.contexts[item["context_id"]]) for item in payload["items"]]
                        if path == "/decode" else [])
        result = call(path, payload)
        if path == "/decode":
            decoded.extend((float(obs["state"][0]), np.asarray(item["actions"], np.float32))
                           for obs, item in zip(observations, result["items"], strict=True))
        return result

    inf.call = record
    runner.run()
    replay = arrays(runner.residual_warmup.replay)
    timed = np.flatnonzero(replay["truncated"][:, 0])
    assert len(timed)
    for index in timed:
        expected_marker = float(replay["next_obs"][index, 3])
        candidates = [action for marker, action in decoded if abs(marker - expected_marker) < 1e-6]
        assert candidates
        next_base = replay["next_base_action"][index].reshape(4, 3)
        assert any(np.array_equal(next_base, action) for action in candidates)
        assert np.any(next_base != 0)
    assert not inf.contexts


def test_shadow_checkpoint_resume_preserves_networks_replay_and_counters(tmp_path):
    runner, inf, sim, _ = build(tmp_path / "first", steps=6)
    first = runner.run()
    payload = torch.load(first["last_checkpoint"], weights_only=False)
    saved = payload["extra"]["residual_warmup"]
    assert saved["learner"]["updates"] == runner.residual_warmup.learner.updates
    assert saved["replay"]["size"] == 6
    options = asdict(runner.config)
    options["output_dir"] = str(tmp_path / "resumed")
    options["rollout"]["max_transitions"] = 12
    options["checkpoint"]["resume"] = first["last_checkpoint"]
    resumed, _, _, _ = build(tmp_path / "resumed", cfg=options)
    result = resumed.run()
    assert result["state"] == "completed" and result["transitions"] == 12
    for key, value in resumed.initial_shadow["actor"].items():
        assert torch.equal(value, saved["learner"]["actor"][key])
    for key, value in resumed.initial_shadow["qs"].items():
        assert torch.equal(value, saved["learner"]["qs"][key])
    assert resumed.initial_shadow["updates"] == saved["learner"]["updates"]
    assert resumed.initial_shadow["replay_size"] == 6
    assert resumed.residual_warmup.learner.updates > saved["learner"]["updates"]
    assert not resumed.residual_warmup.pending
    for key, value in saved["replay"]["arrays"].items():
        np.testing.assert_array_equal(resumed.residual_warmup.replay.arrays[key][:6], value)


def test_both_learners_update_with_periodic_atomic_checkpoints(tmp_path):
    runner, inf, sim, _ = build(tmp_path, steps=12, fast=True, fused=True)
    runner.config.update.utd = 1.
    runner.config.checkpoint.interval = 4
    sim.delay = .005
    result = runner.run()
    assert result["updates"] > 0
    assert result["residual_warmup"]["updates"] > 0
    assert result["residual_warmup"]["executed"] is False
    paths = sorted((tmp_path / "checkpoints").glob("step_*.pt"))
    assert len(paths) == 3
    for path in paths:
        saved = torch.load(path, weights_only=False)
        transitions = saved["extra"]["transitions"]
        assert saved["extra"]["replay"]["size"] == transitions
        assert saved["extra"]["residual_warmup"]["replay"]["size"] == transitions
        assert saved["extra"]["residual_warmup"]["transitions"] == transitions
    assert len(sim.steps) == 6  # Checkpoint target inference never advances physics.
    assert not inf.contexts


def test_stopping_after_decode_saves_completed_transitions_without_reexecution(tmp_path):
    runner, inf, sim, _ = build(tmp_path, steps=12, fast=True, fused=True)
    call = inf.call

    def stop_after_second_decode(path, payload=None):
        result = call(path, payload)
        if path == "/decode" and len(inf.decode_calls) == 2:
            runner.stop()
        return result

    inf.call = stop_after_second_decode
    result = runner.run()
    assert result["state"] == "stopped" and result["transitions"] == 4
    assert len(sim.steps) == 2
    assert len(runner.residual_warmup.replay) == 4
    assert not runner.residual_warmup.pending
    saved = torch.load(result["last_checkpoint"], weights_only=False)
    assert saved["extra"]["residual_warmup"]["replay"]["size"] == 4
    assert not inf.contexts


def test_lost_step_response_is_never_retried_or_added_to_shadow_replay(tmp_path):
    runner, inf, sim, _ = build(tmp_path)
    sim.lose_step_response = True
    with pytest.raises(TimeoutError, match="lost response"):
        runner.run()
    assert len(sim.steps) == 1
    assert len(runner.residual_warmup.replay) == 0
    assert not runner.residual_warmup.pending
    assert not inf.contexts


@pytest.mark.parametrize("change", ["disable", "loss", "scale", "replay"])
def test_incompatible_shadow_resume_fails_before_rollout(tmp_path, change):
    original, _, _, _ = build(tmp_path / "first", steps=6)
    checkpoint = original.run()["last_checkpoint"]
    options = asdict(original.config)
    options["output_dir"] = str(tmp_path / "wrong")
    options["checkpoint"]["resume"] = checkpoint
    options["rollout"]["max_transitions"] = 12
    if change == "disable":
        options["residual_warmup"] = None
    elif change == "loss":
        options["residual_warmup"]["loss"]["actor_l2_weight"] = 2.
    elif change == "scale":
        options["residual_warmup"]["learner"]["residual_scale"] = [.01, .01]
    else:
        options["residual_warmup"]["replay_capacity"] = 32
    wrong, inf, sim, _ = build(tmp_path / "wrong", cfg=options)
    with pytest.raises(ValueError, match="warmup|configuration"):
        wrong.run()
    assert not sim.resets and not inf.encode_calls


def residual_options(output, source, checkpoint):
    options = config(output, method="action_residual", steps=4)
    options["learner"] = asdict(source.residual_warmup.learner.config)
    options["update"].update(utd=0., learning_starts=100, replay_capacity=64)
    options["checkpoint"].update(interval=100_000, keep=None)
    options["frozen_dsrl"] = dict(checkpoint=checkpoint, deterministic=True)
    options["residual_initialize_from"] = checkpoint
    return options


@pytest.mark.parametrize("include_replay", [True, False])
def test_residual_stage_inherits_warm_networks_but_starts_fresh_rollouts(tmp_path, include_replay):
    source, _, _, _ = build(tmp_path / "dsrl", steps=6)
    source.config.checkpoint.include_replay = include_replay
    checkpoint = source.run()["last_checkpoint"]
    options = residual_options(tmp_path / "residual", source, checkpoint)
    residual, _, sim, _ = build(tmp_path / "residual", cfg=options)
    result = residual.run()
    trained = source.residual_warmup.learner
    assert result["state"] == "completed" and result["transitions"] == 4
    assert residual.initial_main["transitions"] == 0
    assert residual.initial_main["replay_size"] == 0
    assert residual.initial_main["updates"] == trained.updates
    assert_module_equal(residual.learner.actor, trained.actor.state_dict())
    assert_module_equal(residual.learner.qs, trained.qs.state_dict())
    assert_module_equal(residual.learner.q_targets, trained.q_targets.state_dict())
    assert residual.learner.actor_optimizer.state
    assert residual.learner.q_optimizer.state
    assert len(residual.replay) == 4
    assert np.all(residual.replay.arrays["action"][:4] == 0)
    assert len(sim.steps) == 2


@pytest.mark.parametrize("change", ["noise_weights", "no_warmup", "reward"])
def test_residual_initialization_rejects_unrelated_checkpoint(tmp_path, change):
    source, _, _, _ = build(tmp_path / "source", steps=6)
    checkpoint = source.run()["last_checkpoint"]
    options = residual_options(tmp_path / "wrong", source, checkpoint)
    changed = torch.load(checkpoint, weights_only=False)
    altered = tmp_path / "altered.pt"
    if change == "noise_weights":
        changed["actor"]["mean.bias"] += .01
        torch.save(changed, altered)
        options["frozen_dsrl"]["checkpoint"] = str(altered)
    else:
        if change == "no_warmup":
            del changed["extra"]["residual_warmup"]
        else:
            changed["extra"]["human_feedback"] = dict(mode="human", success_reward=2., failure_reward=-1.)
        torch.save(changed, altered)
        options["residual_initialize_from"] = str(altered)
        options["frozen_dsrl"]["checkpoint"] = str(altered)
    wrong, inf, sim, _ = build(tmp_path / "wrong", cfg=options)
    with pytest.raises(ValueError, match="warmup|DSRL|checkpoint|reward|feedback"):
        wrong.run()
    assert not sim.resets and not inf.encode_calls


@pytest.mark.parametrize("invalid", [
    dict(method="action_residual"),
    dict(mode="eval", checkpoint={"resume": "unused.pt"}),
    dict(rollout={"terminal_observation_mode": "state_only_if_no_bootstrap"},
         residual_warmup={**warmup_settings(), "learner": {
             **warmup_settings()["learner"], "bootstrap_truncated": True}}),
])
def test_invalid_shadow_configuration_rejected(invalid):
    options = dict(residual_warmup=warmup_settings())
    options.update(invalid)
    with pytest.raises(ValueError, match="warmup|bootstrap|train|dsrl"):
        TrainerConfig(**options)


def test_isolated_shadow_rng_does_not_perturb_primary_streams():
    from pressb.runtime.residual_warmup import IsolatedRNG

    torch.manual_seed(257)
    np.random.seed(89)
    random.seed(183)
    state = (torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())
    isolated = IsolatedRNG("cpu")
    with pytest.raises(RuntimeError, match="test failure"):
        with isolated.use():
            torch.rand(19)
            np.random.random(12)
            random.random()
            raise RuntimeError("test failure")
    assert torch.equal(state[0], torch.get_rng_state())
    assert np.array_equal(state[1][1], np.random.get_state()[1])
    assert state[1][2:] == np.random.get_state()[2:]
    assert state[2] == random.getstate()
    with isolated.use():
        first = torch.rand(2)
    with isolated.use():
        second = torch.rand(2)
    assert not torch.equal(first, second)  # Auxiliary RNG progresses independently.
    assert torch.equal(state[0], torch.get_rng_state())
