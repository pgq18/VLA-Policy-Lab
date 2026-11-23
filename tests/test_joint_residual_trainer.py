"""Run joint residual rollouts through SAC, the inference service and a CPU simulator fixture."""
import numpy as np
import pytest

from pressb.runtime.inference import InferenceService
from pressb.runtime.learner import SACLearner
from pressb.runtime.observation_encoding import negotiate_rl_contract
from pressb.runtime.trainer import GenericTrainer
from test_rl_nodes_trainer import StateOnlySimulation, config, metadata, one_cpu_thread


class LinearDecoder:
    def __init__(self, meta):
        self.meta = meta
        self.decoded = []

    def describe(self):
        return self.meta

    def encode_batch(self, observations, seeds):
        return [(np.random.default_rng(seed).standard_normal((4, 4)).astype(np.float32),
                 [obs["state"][0], 2., 3.]) for obs, seed in zip(observations, seeds, strict=True)]

    def baseline_noise(self, context):
        return context

    def decode_batch(self, contexts, initial_noises):
        commands = []
        for baseline, noise in zip(contexts, initial_noises, strict=True):
            initial = baseline if noise is None else noise
            self.decoded.append((baseline.copy(), initial.copy()))
            commands.append(initial[:, :3] * .01)
        return commands


class InferenceClient:
    def __init__(self, backend):
        self.service = InferenceService(backend, batch_size=2)
        self.requests = []

    def call(self, path, payload=None):
        self.requests.append((path, payload))
        return getattr(self.service, path[1:])(payload)


def make_joint_runner(output, steps, *, checkpoint=None, mode="train"):
    inf, sim = metadata(horizon=4, action_dim=3, noise_dim=4, state_dim=3)
    sim["action"].update(lower=[-.1] * 3, upper=[.1] * 3)
    cfg = config(output, method="joint_residual", steps=steps)
    cfg.update(mode=mode)
    cfg["learner"].update(noise_steps=2, noise_scale=1.5, residual_indices=[0, 2],
                          residual_scale=[.25, .03])
    cfg["rollout"].update(action_bounds="clip", terminal_observation_mode="state_only_if_no_bootstrap",
                          fused_step_reset=True)
    cfg["checkpoint"]["resume"] = checkpoint
    return GenericTrainer(cfg, inference_client=InferenceClient(LinearDecoder(inf)),
                          simulation_client=StateOnlySimulation(sim))


def test_joint_rollout_composes_both_residuals_and_replays_exact_pair(tmp_path, monkeypatch):
    runner = make_joint_runner(tmp_path, 4)
    runner.config.update.learning_starts = 0
    # The four commands isolate the baseline, the action delta, the noise
    # delta, and their combination. The first action coordinate hits a limit.
    action = np.tile([.8, -.5], 4)
    noise = np.array([.2, -.3, .4, -.5, .6, -.7, .8, -.9])
    pairs = np.array([np.r_[a * action, n * noise] for a, n in
                      ((0, 0), (1, 0), (0, 1), (1, 1))], dtype=np.float32)
    batches = iter(np.split(pairs, 2))

    def act(learner, observations, base_actions=None, **options):
        assert base_actions is None
        batch = next(batches)
        assert batch.shape == (len(observations), learner.action_dim)
        return batch.copy()

    monkeypatch.setattr(SACLearner, "act", act)
    result = runner.run()
    assert result["state"] == "completed" and result["updates"] > 0
    assert result["projected_action_coordinates"] == 4  # Two rows × two executed prefix positions.
    np.testing.assert_array_equal(runner.replay.arrays["action"][:4], pairs)
    assert {"base_action", "next_base_action"}.isdisjoint(runner.replay.arrays)
    assert all(not req["include_base_actions"] for path, req in runner.inference.requests if path == "/encode")
    decodes = [i for path, req in runner.inference.requests if path == "/decode" for i in req["items"]]
    commands = [i["actions"] for req in runner.simulation.steps for i in req["items"]]
    assert len(decodes) == len({i["context_id"] for i in decodes}) == len(commands) == 4
    for pair, request, (baseline, initial), command in zip(
            pairs, decodes, runner.inference.service.backend.decoded, commands, strict=True):
        action_delta = pair[:8].reshape(4, 2) * [.25, .03]
        noise_delta = np.tile(pair[8:].reshape(2, 4), (2, 1)) * 1.5
        assert "initial_noise" not in request
        np.testing.assert_allclose(request["noise_residual"], noise_delta)
        np.testing.assert_allclose(initial, baseline + noise_delta)
        expected = (baseline + noise_delta)[:, :3] * .01
        expected[:, [0, 2]] += action_delta
        np.testing.assert_allclose(command, np.clip(expected[:2], -.1, .1), atol=1e-7)
    assert not runner.inference.service.contexts


def test_joint_checkpoint_resumes_training_and_runs_deterministic_evaluation(tmp_path, monkeypatch):
    first = make_joint_runner(tmp_path / "train", 6)
    trained = first.run()
    replay = first.replay.arrays["action"][:6].copy()
    np.testing.assert_array_equal(replay[:2], np.zeros_like(replay[:2]))
    assert np.any(replay[2:, :8]) and np.any(replay[2:, 8:])
    continued = make_joint_runner(tmp_path / "resume", 8, checkpoint=trained["last_checkpoint"])
    result = continued.run()
    np.testing.assert_array_equal(continued.replay.arrays["action"][:6], replay)
    assert result["state"] == "completed" and result["updates"] > trained["updates"]
    act = SACLearner.act

    def eval_action(learner, obs, base_actions=None, **options):
        assert options["deterministic"] and not options["warmup"]
        return act(learner, obs, base_actions, **options)

    monkeypatch.setattr(SACLearner, "act", eval_action)
    evaluate = make_joint_runner(tmp_path / "eval", 12, checkpoint=result["last_checkpoint"], mode="eval")
    status = evaluate.run()
    assert status["state"] == "completed" and status["episodes"] == 4
    assert status["updates"] == result["updates"] and not len(evaluate.replay)


@pytest.mark.parametrize("missing", ["external_initial_noise", "additive_noise_residual"])
def test_joint_negotiation_requires_additive_external_noise(missing):
    inf, sim = metadata()
    inf["capabilities"].append("additive_noise_residual")
    inf["capabilities"].remove(missing)
    with pytest.raises(ValueError, match=missing):
        negotiate_rl_contract(inf, sim, "joint_residual", execution_horizon=2)
    assert negotiate_rl_contract(inf, sim, "action_residual", execution_horizon=2)["method"] == "action_residual"
    if missing == "additive_noise_residual":
        assert negotiate_rl_contract(inf, sim, "dsrl", execution_horizon=2)["method"] == "dsrl"
