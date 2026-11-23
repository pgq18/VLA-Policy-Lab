"""Every native task forwards physics choices to the same execution core.

These tests only construct task/config objects. Real contact, trajectory and
indexed-reset qualification belongs to the separately recorded Isaac probes.
"""
from copy import deepcopy
import json

import pytest

from pressb.tasks.elevator.parallel import ElevatorTask
from pressb.tasks.pap.parallel import PAPTask
from pressb.tasks.pap_plus.parallel import PAPPlusTask
from pressb.tasks.playground.parallel import PlaygroundTask
from pressb.tasks.table_buttons.parallel import TableButtonsTask
from pressb.tasks.cube_stack.parallel import CubeStackTask
from pressb.tasks.tabletop_cleanup.parallel import TabletopCleanupTask
from pressb.tasks.playground.backend import PlaygroundBackend
from pressb.tasks.table_buttons.backend import TableButtonsBackend
from pressb.tasks.pap.backend import PAPParallelBackend, ROOT
from test_elevator_parallel import task_config


TASKS = {
    "elevator": ElevatorTask,
    "pap": PAPTask,
    "pap_plus": PAPPlusTask,
    "playground": PlaygroundTask,
    "table_buttons": TableButtonsTask,
    "cube_stack": CubeStackTask,
    "tabletop_cleanup": TabletopCleanupTask,
}


def options_for(task, elevator_config):
    if task == "elevator":
        return deepcopy(elevator_config)
    if task in ("pap", "pap_plus"):
        config = json.loads((ROOT / f"configs/rl_nodes/{task}_simulation.json").read_text())["backend"]["config"]
        # The public RL template explicitly selects GPU. This fixture tests
        # task-level omission/override behavior independently of that preset.
        for key in ("physics_backend", "gpu_dynamics", "gpu_memory_capacities"):
            config.pop(key, None)
        return config
    return {"num_envs": 2}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("physics,expected", [({}, "cpu"), ({"physics_backend": "cpu"}, "cpu"),
    ({"physics_backend": "gpu"}, "gpu"), ({"gpu_dynamics": True}, "gpu")])
def test_all_native_tasks_forward_canonical_physics(task, physics, expected, task_config):
    spec = options_for(task, task_config)
    obj = TASKS[task]({**spec, **physics})
    for source in (obj.config, obj.execution):
        assert source["physics_backend"] == expected
        assert source["gpu_dynamics"] is (expected == "gpu")
        assert isinstance(source["gpu_memory_capacities"], dict)
        assert bool(source["gpu_memory_capacities"]) is (expected == "gpu")
    assert obj.config["gpu_memory_capacities"] == obj.execution["gpu_memory_capacities"]


@pytest.mark.parametrize("task", TASKS)
def test_gpu_buffer_override_survives_task_configuration(task, task_config):
    spec = options_for(task, task_config)
    obj = TASKS[task]({**spec, "physics_backend": "gpu",
                      "gpu_memory_capacities": {"max_rigid_contact_count": 2**21}})
    assert obj.execution["gpu_memory_capacities"]["max_rigid_contact_count"] == 2**21


@pytest.mark.parametrize("task", TASKS)
def test_tasks_reject_conflicting_cpu_gpu_configuration(task, task_config):
    spec = options_for(task, task_config)
    with pytest.raises(ValueError, match="conflict"):
        TASKS[task]({**spec, "physics_backend": "gpu", "gpu_dynamics": False})


@pytest.mark.parametrize("task", TASKS)
def test_physics_choice_does_not_mutate_scene_or_reset_distribution(task, task_config):
    spec = options_for(task, task_config)
    before = deepcopy(spec)
    cpu = TASKS[task](spec)
    gpu = TASKS[task]({**spec, "physics_backend": "gpu"})
    assert spec == before
    if task == "elevator":
        assert cpu.scene == gpu.scene
        assert cpu.sample_reset(0, 42, {}) == gpu.sample_reset(0, 42, {})
    else:
        assert cpu.config["scene"] == gpu.config["scene"]
        for key in ("randomization", "smoothing", "outcome", "reward"):
            assert cpu.config.get(key) == gpu.config.get(key)
    assert cpu.metadata()["action"] == gpu.metadata()["action"]
    assert cpu.metadata()["observation"] == gpu.metadata()["observation"]


@pytest.mark.parametrize("backend,spec", [
    (PlaygroundBackend, {}), (TableButtonsBackend, {}),
    (PAPParallelBackend, json.loads((ROOT / "configs/rl_nodes/pap_simulation.json").read_text())["backend"]["config"]),
])
def test_legacy_backends_do_not_silently_ignore_gpu_request(backend, spec):
    with pytest.raises(ValueError, match="shared parallel"):
        backend({**spec, "physics_backend": "gpu"})
