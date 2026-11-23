"""Late task scheduling only sees reset slots, never speculative fused slots."""
from copy import deepcopy

import pytest

from pressb.runtime.parallel import SharedWorldBackend
from test_parallel_backend import LiftTask, DynamicsEngine, actions


class ScheduledTask(LiftTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allocated = 0
        self.finalized = []

    def finalize_reset_specs(self, ids, specs):
        self.finalized.append(list(ids))
        return {eid: {**deepcopy(specs[eid]), 'allocation': self.allocated + n}
                for n, eid in enumerate(ids)}

    def reset_task(self, ids, specs):
        result = super().reset_task(ids, specs)
        self.allocated += len(ids)
        return result

    def close_run(self):
        self.allocated = 0


def make():
    task = ScheduledTask()
    backend = SharedWorldBackend(task, engine_factory=DynamicsEngine)
    backend.reset([0, 1], 1)
    return backend, task


def test_preview_does_not_allocate_and_fused_finalizes_only_ended_slots():
    backend, task = make()
    assert task.allocated == 2
    backend._prepare_reset([1, 0], 8, {})
    assert task.allocated == 2 and task.finalized == [[0, 1]]
    task.thresholds[:] = [.005, 99.]
    result = backend.step_and_reset(actions(horizon=1), reset_seed=8, reset_options={}, max_active_envs=2)
    assert [row['env_id'] for row in result['reset_items']] == [0]
    assert task.finalized[-1] == [0]
    assert backend.reset_specs[0]['allocation'] == 2
    assert task.allocated == 3


def test_invalid_preview_has_no_physical_or_schedule_side_effect():
    backend, task = make()
    before = backend.snapshot()
    with pytest.raises(ValueError, match='Unknown'):
        backend.step_and_reset(actions(horizon=1), reset_seed=8,
                               reset_options={'by_env': {'1': {'invalid': 1}}}, max_active_envs=2)
    assert backend.snapshot() == before
    assert task.allocated == 2 and task.finalized == [[0, 1]]


def test_invalid_finalizer_fails_before_robot_task_or_clock_changes():
    backend, task = make()
    backend.end_episode([0])
    before = backend.snapshot()
    task.finalize_reset_specs = lambda ids, specs: {1: specs[0]}
    with pytest.raises(ValueError, match='exactly the selected'):
        backend.reset([0], 7)
    assert task.allocated == 2
    assert backend.engine.world.current_time == before['0']['world_time']
    assert task.reset_calls == [[0, 1]]


def test_close_run_clears_schedule_and_requires_full_next_reset_without_ticks():
    backend, task = make()
    clock = backend.engine.world.current_time
    captures = len(backend.engine.captures)
    backend.close_run()
    assert task.allocated == 0
    assert not backend.ready.any() and backend.reset_specs == [None, None]
    assert backend.engine.world.current_time == clock
    assert len(backend.engine.captures) == captures
    with pytest.raises(ValueError, match='all environments'):
        backend.reset([0], 9)
    backend.reset([0, 1], 9)
    assert [backend.reset_specs[i]['allocation'] for i in (0, 1)] == [0, 1]
