"""RL deployment defaults without opening a simulator, model or training run."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pressb.experiments.config import apply_training_defaults, load_experiment, materialize, PROJECT_ROOT
from pressb.experiments.cli import main
from pressb.registry import resolve
from pressb.runtime.physics import normalize_physics_config


NATIVE_NAMES = ("pap", "pap_plus", "pap_plus_plus", "pap_recording", "elevator", "table_buttons",
                "playground", "cube_stack", "tabletop_cleanup", "fr3_pap")
ALIASES = (
    "pressb.tasks.pap.vector:create_backend",
    "pressb.tasks.pap.vector:PAPVectorBackend",
    "pressb.tasks.pap.recording:create_backend",
    "pressb.tasks.pap.recording:PAPRecordingVectorBackend",
    "pressb.tasks.elevator.native:create_backend",
    "pressb.tasks.elevator.native:NativeElevatorBackend",
)


def document(task=None, *, method="dsrl", mode="train", config=None):
    return {"schema_version": 1, "name": "defaults_test",
            "task": {**(task or {"plugin": "pap"}), "config": config or {}},
            "model": {"plugin": "walloss", "config": {}},
            "run": {"method": method, "mode": mode}}


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def task_config(experiment):
    return experiment["nodes"]["simulation"]["backend"]["config"]


@pytest.mark.parametrize("method", ["dsrl", "action_residual", "dppo"])
@pytest.mark.parametrize("task", [{"plugin": name} for name in NATIVE_NAMES]
                         + [{"factory": resolve("task", name)} for name in NATIVE_NAMES]
                         + [{"factory": name} for name in ALIASES])
def test_all_native_rl_methods_default_to_gpu(tmp_path, task, method):
    path = save(tmp_path / "input.json", document(task, method=method))
    result = load_experiment(path)
    assert task_config(result)["physics_backend"] == "gpu"
    assert "gpu_dynamics" not in task_config(result)
    if "elevator" in result["factories"]["task"] and method == "dsrl":
        assert task_config(result)["floor_sampling"] == "random"
        assert result["nodes"]["trainer"]["observation_state_encoding"] == "pose9"
    else:
        assert "floor_sampling" not in task_config(result)
        assert "observation_state_encoding" not in result["nodes"]["trainer"]


def test_implicit_trainer_method_and_mode_use_dsrl_defaults(tmp_path):
    data = document({"plugin": "elevator"})
    data["run"] = {}
    compiled = load_experiment(save(tmp_path / "input.json", data))
    assert task_config(compiled) == {"physics_backend": "gpu", "floor_sampling": "random"}
    assert compiled["nodes"]["trainer"]["observation_state_encoding"] == "pose9"


@pytest.mark.parametrize("config", [
    {"physics_backend": "cpu"}, {"gpu_dynamics": False},
    {"physics_backend": "cpu", "gpu_dynamics": False},
    {"physics_backend": "gpu"}, {"gpu_dynamics": True},
])
def test_explicit_physics_mode_or_alias_wins(tmp_path, config):
    data = document({"plugin": "elevator"}, config=config)
    data["run"]["observation_state_encoding"] = "raw"
    compiled = load_experiment(save(tmp_path / "input.json", data))
    assert task_config(compiled) == {**config, "floor_sampling": "random"}
    assert compiled["nodes"]["trainer"]["observation_state_encoding"] == "raw"


@pytest.mark.parametrize("config", [
    {"physics_backend": "gpu", "gpu_dynamics": False},
    {"physics_backend": "cpu", "gpu_dynamics": True},
    {"physics_backend": None}, {"gpu_dynamics": None},
])
def test_invalid_explicit_physics_is_not_silently_overridden(tmp_path, config):
    result = load_experiment(save(tmp_path / "input.json", document(config=config)))
    assert task_config(result) == config
    with pytest.raises(ValueError):
        normalize_physics_config(task_config(result))


def test_elevator_sampling_override_and_other_run_settings_are_preserved(tmp_path):
    data = document({"plugin": "elevator"}, config={"floor_sampling": "round_robin"})
    data["run"].update(observation_state_encoding="raw", rollout={"max_transitions": 1234},
                       learner={"batch_size": 17}, residual_warmup={"fixture": "unchanged"})
    compiled = load_experiment(save(tmp_path / "input.json", data))
    assert task_config(compiled)["floor_sampling"] == "round_robin"
    for key in ("observation_state_encoding", "rollout", "learner", "residual_warmup"):
        assert compiled["nodes"]["trainer"][key] == data["run"][key]


@pytest.mark.parametrize("task", [
    {"plugin": "piper_real"}, {"plugin": "pap_process"}, {"plugin": "elevator_remote"},
    {"factory": "pressb.runtime.real_robot:create_backend"},
    {"factory": "pressb.tasks.pap.backend:create_backend"},
    {"factory": "pressb.tasks.elevator.backend:create_backend"},
    {"factory": "pressb.tasks.elevator.legacy_native:create_backend"},
    {"factory": "example.custom:create_backend"},
])
def test_real_legacy_remote_and_external_factories_are_not_reconfigured(tmp_path, task):
    compiled = load_experiment(save(tmp_path / "input.json", document(task)))
    assert task_config(compiled) == {}
    assert "observation_state_encoding" not in compiled["nodes"]["trainer"]


@pytest.mark.parametrize("method", ["base", "dsrl", "action_residual", "dppo"])
def test_eval_recipe_without_explicit_physics_remains_unmodified(tmp_path, method):
    compiled = load_experiment(save(tmp_path / "input.json", document(mode="eval", method=method)))
    assert task_config(compiled) == {}


def test_defaults_apply_after_inheritance_and_do_not_leak_into_base_eval(tmp_path):
    base = document()
    base_path = save(tmp_path / "parent.json", base)
    child = {"extends": "parent.json", "name": "eval_child", "run": {"method": "base", "mode": "eval"}}
    child_path = save(tmp_path / "child.json", child)
    assert task_config(load_experiment(base_path)) == {"physics_backend": "gpu"}
    assert task_config(load_experiment(child_path)) == {}
    assert json.loads(base_path.read_text()) == base
    assert json.loads(child_path.read_text()) == child


def test_apply_defaults_is_pure_idempotent_and_materializes_effective_values(tmp_path):
    data = document({"plugin": "elevator"}, mode="eval")
    path = save(tmp_path / "input.json", data)
    before = load_experiment(path)
    before["nodes"]["trainer"]["mode"] = "train"
    original = deepcopy(before)
    effective = apply_training_defaults(before)
    assert before == original
    assert apply_training_defaults(effective) == effective
    paths = materialize(effective, tmp_path / "resolved")
    assert json.loads(paths["simulation"].read_text())["backend"]["config"] == {
        "physics_backend": "gpu", "floor_sampling": "random"}
    assert json.loads(paths["trainer"].read_text())["observation_state_encoding"] == "pose9"
    assert json.loads(path.read_text()) == data


def _fake_runner(monkeypatch):
    import pressb.runtime.trainer as module
    captured = []

    class Runner:
        def __init__(self, config, *, observer=None):
            captured.append(deepcopy(config))
            self.observer = observer

        def stop(self):
            pass

        def run(self):
            result = {"state": "completed"}
            if self.observer:
                self.observer.close(result)
            return result

    monkeypatch.setattr(module, "GenericTrainer", Runner)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    return captured


def test_cli_train_override_reapplies_defaults_before_writing_configs(tmp_path, monkeypatch, capsys):
    captured = _fake_runner(monkeypatch)
    data = document({"plugin": "elevator"}, mode="eval")
    data["run"]["output_dir"] = str(tmp_path / "run")
    path = save(tmp_path / "input.json", data)
    assert main(["train", "--config", str(path), "--resolved-dir", str(tmp_path / "resolved")]) == 0
    assert captured[0]["mode"] == "train"
    assert captured[0]["observation_state_encoding"] == "pose9"
    simulation = json.loads((tmp_path / "resolved/simulation.json").read_text())
    assert simulation["backend"]["config"]["physics_backend"] == "gpu"
    assert json.loads(path.read_text()) == data


@pytest.mark.parametrize("command", ["eval", "collect"])
def test_cli_eval_of_training_recipe_retains_its_physical_and_observation_profile(tmp_path, monkeypatch, command):
    captured = _fake_runner(monkeypatch)
    data = document({"plugin": "elevator"})
    data["run"]["output_dir"] = str(tmp_path / "run")
    path = save(tmp_path / "input.json", data)
    assert main([command, "--config", str(path), "--resume", str(tmp_path / "policy.pt"),
                 "--resolved-dir", str(tmp_path / "resolved")]) == 0
    assert captured[0]["mode"] == "eval"
    assert captured[0]["observation_state_encoding"] == "pose9"
    simulation = json.loads((tmp_path / "resolved/simulation.json").read_text())
    assert simulation["backend"]["config"]["physics_backend"] == "gpu"


def test_cli_base_collection_keeps_generic_simulation_default(tmp_path, monkeypatch):
    captured = _fake_runner(monkeypatch)
    data = document(method="base", mode="eval")
    data["run"]["output_dir"] = str(tmp_path / "run")
    path = save(tmp_path / "input.json", data)
    assert main(["collect", "--config", str(path), "--resolved-dir", str(tmp_path / "resolved")]) == 0
    assert captured[0]["mode"] == "eval"
    simulation = json.loads((tmp_path / "resolved/simulation.json").read_text())
    assert simulation["backend"]["config"] == {}


def test_resolver_stays_lazy_in_fresh_process(tmp_path):
    path = save(tmp_path / "input.json", document({"plugin": "elevator"}))
    script = """
import sys
from pressb.experiments.config import load_experiment
result = load_experiment(sys.argv[1])
assert result['nodes']['simulation']['backend']['config']['physics_backend'] == 'gpu'
assert 'torch' not in sys.modules and 'isaacsim' not in sys.modules
assert not any(name.startswith(('pressb.tasks.', 'pressb.models.')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, str(path)], check=True,
                   env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
