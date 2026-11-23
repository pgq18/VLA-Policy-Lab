"""RL entry defaults reach the backend; no Isaac, CUDA or sockets are opened."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pressb.experiments.config import load_experiment
from pressb.registry import resolve


ROOT = Path(__file__).resolve().parents[1]


def read_node(name):
    return json.loads((ROOT / "configs/rl_nodes" / name).read_text())


@pytest.mark.parametrize("script_name", ["serve_rl_simulation", "serve_rl_fast_simulation"])
@pytest.mark.parametrize("flags,expected", [([], True), (["--gpu-dynamics"], True),
                                            (["--no-gpu-dynamics"], False)])
def test_legacy_rl_server_forwards_gpu_default_and_explicit_cpu(tmp_path, monkeypatch, script_name, flags, expected):
    """Exercise main's parser, validation, constructor and cleanup together."""
    captures = {"backend": [], "closed": []}

    class App:
        def __init__(self, settings):
            captures["app"] = settings
            self.app = SimpleNamespace(post_quit=lambda code: captures.setdefault("quit", code))

        def close(self):
            captures["closed"].append("app")

    class Backend:
        def __init__(self, app, **kwargs):
            captures["backend"].append(kwargs)

        def close(self):
            captures["closed"].append("backend")

    class Service:
        def __init__(self, backend, **kwargs):
            pass

        def health(self, payload=None):
            return {}

        reset = reset_envs = step = close_run = health

    class Server:
        def __init__(self, address, handlers, **kwargs):
            captures["address"] = address
            captures["routes"] = set(handlers)

        def serve_forever(self, **kwargs):
            captures["served"] = True

        def server_close(self):
            captures["closed"].append("server")

    modules = {
        "isaacsim": {"SimulationApp": App},
        "pressb.tasks.elevator.collection": {"DATASET_ANTI_ALIASING": 3},
        "pressb.tasks.elevator.simulation": {
            "_file_identity": lambda path: {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()},
            "IsaacVectorBackend": Backend, "SimulationService": Service},
        "pressb.tasks.elevator.fast_simulation": {"FastIsaacVectorBackend": Backend,
                                                   "IndexedSimulationService": Service},
        "pressb.runtime.rpc": {"RPCServer": Server},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    script_path = ROOT / "scripts" / f"{script_name}.py"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location(f"_entry_test_{script_name}", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    # main only uses setdefault; retain those additions inside this test.
    monkeypatch.setattr(module.os, "environ", dict(module.os.environ))
    frozen = tmp_path / "outputs/rtx5090_eval_comparison/input"
    (frozen / "dataset_metadata/meta").mkdir(parents=True)
    (frozen / "config.json").write_text("{}")
    snapshot = frozen / "scene.usda"
    snapshot.write_text("#usda 1.0\n")
    (frozen / "dataset_metadata/meta/collection_metadata.json").write_text(json.dumps({
        "config": {}, "scene_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}))

    result = module.main(["--output", str(tmp_path / "new-run"), "--no-asset-bundle", *flags])
    assert result == 0
    assert len(captures["backend"]) == 1
    assert captures["backend"][0]["gpu_dynamics"] is expected
    assert captures["served"] and captures["quit"] == 0
    assert captures["closed"] == ["server", "backend", "app"]
    assert {"/health", "/reset", "/step", "/close_run"}.issubset(captures["routes"])


def test_canonical_elevator_dsrl_uses_random_gpu_pose9_without_budget_change():
    result = load_experiment(ROOT / "configs/experiments/elevator_vlajepa_dsrl.json")
    task = result["nodes"]["simulation"]["backend"]["config"]
    trainer = result["nodes"]["trainer"]
    assert task["floor_sampling"] == "random"
    assert task["physics_backend"] == "gpu"
    assert trainer["observation_state_encoding"] == "pose9"
    assert trainer["rollout"]["max_transitions"] == 400000
    assert trainer.get("residual_warmup") is None
    assert read_node("elevator_dsrl.json")["observation_state_encoding"] == "pose9"


@pytest.mark.parametrize("filename,mode,encoding,sampling", [
    ("elevator_vlajepa_dsrl_residual_warmup_100k.json", "cpu", "raw", "random"),
    ("elevator_vlajepa_dsrl_gpu_physx_warmup_100k.json", "gpu", "raw", "random"),
    ("elevator_vlajepa_dsrl_gpu_round_robin_warmup_100k.json", "gpu", "raw", "round_robin"),
    ("elevator_vlajepa_dsrl_gpu_pose9_random_warmup_100k.json", "gpu", "pose9", "random"),
])
def test_historical_ablations_keep_their_original_solver_and_encoding(filename, mode, encoding, sampling):
    result = load_experiment(ROOT / "configs/experiments" / filename)
    task = result["nodes"]["simulation"]["backend"]["config"]
    assert task["physics_backend"] == mode
    assert task.get("gpu_dynamics", mode == "gpu") is (mode == "gpu")
    assert task["floor_sampling"] == sampling
    assert result["nodes"]["trainer"]["observation_state_encoding"] == encoding
    assert result["nodes"]["trainer"]["rollout"]["max_transitions"] == 100000


def backend_factory(node):
    spec = node["backend"]
    return resolve("task", spec["plugin"]) if "plugin" in spec else spec["factory"]


@pytest.mark.parametrize("filename,task", [
    ("elevator_simulation.json", "elevator"), ("pap_simulation.json", "pap"),
    ("pap_vector_simulation.json", "pap"), ("pap_vector_fast_simulation.json", "pap"),
])
def test_standalone_rl_simulation_templates_choose_native_gpu(filename, task):
    node = read_node(filename)
    assert backend_factory(node) in {
        f"pressb.tasks.{task}.parallel:create_backend",
        f"pressb.tasks.{task}.vector:create_backend",
        f"pressb.tasks.{task}.native:create_backend",
    }
    assert node["backend"]["config"]["physics_backend"] == "gpu"
    if task == "elevator":
        assert node["backend"]["config"]["num_envs"] == 64
        assert node["backend"]["config"]["floor_sampling"] == "random"
        assert "endpoint" not in node["backend"]["config"]


def test_explicit_compatibility_templates_do_not_silently_request_gpu():
    remote = read_node("elevator_remote_simulation.json")
    assert backend_factory(remote) == "pressb.tasks.elevator.backend:create_backend"
    assert remote["backend"]["config"]["endpoint"].startswith("http://")
    assert "physics_backend" not in remote["backend"]["config"]
    assert "gpu_dynamics" not in remote["backend"]["config"]
    process = read_node("pap_process_cpu_simulation.json")
    assert backend_factory(process) == "pressb.tasks.pap.backend:create_backend"
    assert process["backend"]["config"]["physics_backend"] == "cpu"
