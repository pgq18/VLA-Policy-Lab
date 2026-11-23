"""Elevator episode scheduling without Isaac or changes to sampled panel offsets."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.elevator.parallel import ElevatorTask, FLOORS
from test_elevator_parallel import task_config, measured_task


def make_task(task_config, **options):
    return ElevatorTask({**task_config, "floor_sampling": "round_robin", **options})


def preview(task, ids, seed=42, options=None):
    return {eid: task.sample_reset(eid, seed, options or {}) for eid in ids}


@pytest.mark.parametrize("value", [None, "", "cyclic", True, 1, []])
def test_invalid_floor_sampling_fails_during_configuration(task_config, value):
    with pytest.raises(ValueError, match="floor_sampling"):
        make_task(task_config, floor_sampling=value)


def test_default_random_sampler_and_explicit_random_are_identical(task_config):
    default = ElevatorTask(task_config)
    explicit = make_task(task_config, floor_sampling="random")
    assert default.metadata() == explicit.metadata()
    for eid in range(2):
        for seed in (0, 42, 12345):
            raw = default.sample_reset(eid, seed, {})
            assert raw == explicit.sample_reset(eid, seed, {})
            assert set(raw) == {"floor", "task", "offset_x_m", "offset_y_m"}
            assert default.finalize_reset_specs([eid], {eid: raw}) == {eid: raw}
    assert default.scheduled_episodes == explicit.scheduled_episodes == 0


def test_round_robin_preserves_panel_draws_and_sampling_is_pure(task_config):
    random = ElevatorTask(task_config)
    cyclic = make_task(task_config)
    before = cyclic.metadata()
    for eid in range(2):
        for seed in (0, 42, 987654):
            expected = random.sample_reset(eid, seed, {})
            spec = cyclic.sample_reset(eid, seed, {})
            assert {key: spec[key] for key in expected} == expected
            assert spec["floor_explicit_override"] is False
            finalized = cyclic.finalize_reset_specs([eid], {eid: spec})[eid]
            assert finalized["floor"] == 24
            for axis in "xy":
                assert finalized[f"offset_{axis}_m"] == expected[f"offset_{axis}_m"]
    assert cyclic.scheduled_episodes == 0
    assert cyclic.metadata() == before
    assert cyclic.metadata()["task_config_sha256"] != random.metadata()["task_config_sha256"]


def test_global_sequence_continues_after_initial64_across_actual_reset_order(task_config):
    task = make_task(task_config, num_envs=64)
    first_ids = list(range(64))
    initial = task.finalize_reset_specs(first_ids, preview(task, first_ids))
    assert [initial[eid]["floor"] for eid in first_ids] == [24 + i % 12 for i in range(64)]
    assert [initial[eid]["sampling_index"] for eid in first_ids] == list(range(64))
    assert task.scheduled_episodes == 0
    task._commit_reset_sequence(first_ids, initial)
    assert task.scheduled_episodes == 64
    # Fused reset validates every active slot, even when only two later end.
    for _ in range(7):
        candidates = preview(task, first_ids)
    assert task.scheduled_episodes == 64
    actual_ids = [17, 3]
    actual = task.finalize_reset_specs(actual_ids, candidates)
    assert list(actual) == actual_ids
    assert [actual[i]["sampling_index"] for i in actual_ids] == [64, 65]
    assert [actual[i]["floor"] for i in actual_ids] == [28, 29]
    assert task.scheduled_episodes == 64
    task._commit_reset_sequence(actual_ids, actual)
    final = task.finalize_reset_specs([7], preview(task, first_ids))
    assert final[7]["floor"] == 30 and final[7]["sampling_index"] == 66


def test_explicit_floor_and_task_override_respect_and_consume_episode_index(task_config):
    task = make_task(task_config)
    raw = {0: task.sample_reset(0, 1, {"floor": 35}),
           1: task.sample_reset(1, 1, {"task": "Press 30 floor."})}
    selected = task.finalize_reset_specs([0, 1], raw)
    assert [selected[i]["floor"] for i in (0, 1)] == [35, 30]
    assert [selected[i]["sampling_index"] for i in (0, 1)] == [0, 1]
    task._commit_reset_sequence([0, 1], selected)
    next_spec = task.finalize_reset_specs([0], preview(task, [0]))[0]
    assert next_spec["floor"] == 26 and next_spec["sampling_index"] == 2


def test_configured_floor_order_and_fresh_run_restart(task_config):
    task = make_task(task_config, floors=[35, 24, 30])
    sequence = []
    for _ in range(7):
        specs = task.finalize_reset_specs([1], preview(task, [1]))
        sequence.append(specs[1]["floor"])
        task._commit_reset_sequence([1], specs)
    assert sequence == [35, 24, 30, 35, 24, 30, 35]
    task.close_run()
    assert task.scheduled_episodes == 0
    assert task.finalize_reset_specs([1], preview(task, [1]))[1]["floor"] == 35


def test_invalid_preview_and_stale_finalization_cannot_advance_counter(task_config):
    task = make_task(task_config)
    first = task.finalize_reset_specs([0], preview(task, [0]))
    task._commit_reset_sequence([0], first)
    for invalid in ({"floor": 23}, {"task": "bad"}, {"offset_x_m": 100},
                    {"floor": 24, "task": "Press 25 floor."}):
        with pytest.raises(ValueError):
            task.sample_reset(0, 42, invalid)
        assert task.scheduled_episodes == 1
    with pytest.raises(RuntimeError, match="finalized"):
        task._commit_reset_sequence([0], first)
    assert task.scheduled_episodes == 1


def _reset_fixture(task, monkeypatch, fail_velocities=False):
    from pressb.tasks.elevator import dataset_scene
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Gf=SimpleNamespace(Vec3f=lambda *x: x)))
    monkeypatch.setattr(dataset_scene, "_panel_shift", lambda *args: None)
    monkeypatch.setattr(task, "_set_light", lambda *args: None)
    task.engine = SimpleNamespace(stage=SimpleNamespace(GetPrimAtPath=lambda path: object()),
                                  offsets=np.zeros((2, 3)))
    task.panels = [dict(prefix=f"/World/envs/env_{i}", anchors=[],
                       transforms={floor: [] for floor in FLOORS}) for i in range(2)]
    task.source_offsets = [(0., 0.)] * 2
    task.rests = np.zeros((2, 12, 3))
    task.items = [None] * 2
    task.forces = np.zeros((2, 12))
    task.button_rows = np.arange(24).reshape(2, 12)
    task.settled_relative = np.zeros((2, 12, 3))
    task.settled_orientations = np.tile([1., 0., 0., 0.], (2, 12, 1))
    task.settled_velocities = np.zeros((2, 12, 6))
    calls = []

    def set_poses(*args, **kwargs):
        calls.append(("poses", task.scheduled_episodes))

    def set_velocities(*args, **kwargs):
        calls.append(("velocities", task.scheduled_episodes))
        if fail_velocities:
            raise RuntimeError("tensor reset failed")

    task.buttons = SimpleNamespace(set_world_poses=set_poses, set_velocities=set_velocities)
    return calls


@pytest.mark.parametrize("fail", [False, True])
def test_physical_task_reset_commits_only_after_tensor_writes(task_config, monkeypatch, fail):
    task = make_task(task_config)
    calls = _reset_fixture(task, monkeypatch, fail_velocities=fail)
    specs = task.finalize_reset_specs([1, 0], preview(task, [1, 0]))
    if fail:
        with pytest.raises(RuntimeError, match="tensor reset failed"):
            task.reset_task([1, 0], specs)
        assert task.scheduled_episodes == 0
    else:
        info = task.reset_task([1, 0], specs)
        assert task.scheduled_episodes == 2
        assert info[1]["reset"]["sampling_index"] == 0
        assert info[0]["reset"]["sampling_index"] == 1
        assert task.items[1]["floor"] == 24 and task.items[0]["floor"] == 25
    assert calls == [("poses", 0), ("velocities", 0)]


def test_round_robin_transition_info_keeps_target_and_sequence(measured_task):
    task = measured_task
    task.floor_sampling = "round_robin"
    task.engine.backend.reset_specs[0].update(sampling_index=12)
    task.buttons.positions[0, 0] += .002
    task.forces[0, 0] = .03
    info = task.after_tick([0])[0]["info"]
    assert info["target_floor"] == 24 and info["sampling_index"] == 12
    assert info["floor_sampling"] == "round_robin"


def test_real_shared_fused_scheduler_finalizes_only_actual_resets_and_rpc_retries(task_config):
    """Use the real shared state machine/service with a small CPU physics fixture."""
    from pressb.runtime.parallel import SharedWorldBackend
    from pressb.runtime.simulation import SimulationService
    from test_parallel_backend import DynamicsEngine, LiftTask

    sampler = make_task(task_config)

    class SampledLift(LiftTask):
        def sample_reset(self, env_id, seed, options):
            self.sample_calls.append(env_id)
            spec = sampler.sample_reset(env_id, seed, options)
            return {**spec, "x": spec["offset_x_m"]}

        def finalize_reset_specs(self, ids, specs):
            return sampler.finalize_reset_specs(ids, specs)

        def reset_task(self, ids, specs):
            result = super().reset_task(ids, specs)
            sampler._commit_reset_sequence(ids, specs)
            return result

        def close_run(self):
            sampler.close_run()

    task = SampledLift()
    backend = SharedWorldBackend(task, engine_factory=DynamicsEngine)
    service = SimulationService(backend)
    try:
        request = dict(run_id="first", request_id="initial", env_ids=[0, 1], seed=42)
        first = service.reset(request)
        assert sampler.scheduled_episodes == 2
        assert [row["observation"]["task"] for row in first["items"]] == ["Press 24 floor.", "Press 25 floor."]
        assert service.reset(request) == first
        assert sampler.scheduled_episodes == 2
        commands = [dict(env_id=row["env_id"], episode_id=row["episode_id"], step_id=0,
                         actions=[[0., .02]]) for row in first["items"]]
        before = backend.engine.world.current_time
        with pytest.raises(ValueError, match="floor"):
            backend.step_and_reset([dict(env_id=i, actions=[[0., .02]]) for i in [0, 1]],
                                   reset_seed=4, reset_options={"floor": 23}, max_active_envs=2)
        assert sampler.scheduled_episodes == 2 and backend.engine.world.current_time == before
        task.thresholds[0] = .01  # Only env0 ends; both were previewed.
        fused = dict(run_id="first", request_id="fused", items=commands,
                     terminal_observation_mode="state_only_if_no_bootstrap",
                     reset_seed=7, reset_options={}, max_active_envs=2)
        response = service.step_and_reset(fused)
        assert sampler.scheduled_episodes == 3
        assert [row["env_id"] for row in response["reset_items"]] == [0]
        reset = response["reset_items"][0]
        assert reset["observation"]["task"] == "Press 26 floor."
        assert reset["info"]["reset"]["sampling_index"] == 2
        assert task.sample_calls[-2:] == [0, 1]
        assert service.step_and_reset(fused) == response
        assert sampler.scheduled_episodes == 3
        before = backend.engine.world.current_time
        service.close_run({"run_id": "first"})
        assert sampler.scheduled_episodes == 0
        assert backend.engine.world.current_time == before
        fresh = service.reset(dict(run_id="second", request_id="initial", env_ids=[0, 1], seed=42))
        assert [row["observation"]["task"] for row in fresh["items"]] == ["Press 24 floor.", "Press 25 floor."]
        assert sampler.scheduled_episodes == 2
    finally:
        service.close()
