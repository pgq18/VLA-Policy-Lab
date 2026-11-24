"""Physical environment launch compatibility without opening hardware devices."""
import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from pressb.experiments.cli import main
from pressb.experiments.config import load_experiment, materialize, PROJECT_ROOT
from pressb.runtime import server


def experiment(path):
    data = dict(schema_version=1, name="physical", task={"plugin": "piper_real", "config": {}},
                model={"plugin": "walloss", "config": {}},
                nodes={"simulation": {"server": {"host": "0.0.0.0", "port": 8201}}},
                run={"method": "base", "mode": "eval"})
    path.write_text(json.dumps(data))
    return path


def test_physical_experiment_keeps_shared_environment_config_and_endpoint(tmp_path):
    config = load_experiment(experiment(tmp_path / "experiment.json"))
    assert config["factories"]["task"] == "pressb.runtime.real_robot:create_backend"
    assert config["nodes"]["simulation"]["backend"]["plugin"] == "piper_real"
    assert config["nodes"]["trainer"]["endpoints"]["simulation"] == "http://127.0.0.1:8201"
    paths = materialize(config, tmp_path / "resolved")
    assert set(paths) == {"inference", "simulation", "trainer"}
    assert json.loads(paths["simulation"].read_text())["backend"]["plugin"] == "piper_real"


@pytest.mark.parametrize("endpoint_flag", ["--simulation", "--environment"])
def test_real_robot_cli_uses_simulation_config_and_environment_endpoint_alias(
        tmp_path, monkeypatch, endpoint_flag):
    calls = []
    monkeypatch.setattr(server, "run_service", lambda kind, path: calls.append((kind, path)))
    config = experiment(tmp_path / "experiment.json")
    resolved = tmp_path / "resolved"
    assert main(["real_robot", "--config", str(config), "--resolved-dir", str(resolved),
                 endpoint_flag, "http://physical.test:8201"]) == 0
    assert calls == [("real_robot", resolved / "simulation.json")]
    run = json.loads((resolved / "trainer.json").read_text())
    assert run["endpoints"]["simulation"] == "http://physical.test:8201"
    assert "environment" not in run["endpoints"]


def test_real_robot_cli_discovery_and_dispatch_keep_dependencies_lazy(tmp_path):
    config = experiment(tmp_path / "experiment.json")
    source = textwrap.dedent("""
        import importlib.abc
        import pathlib
        import sys

        forbidden = ("torch", "isaacsim", "omni", "piper_sdk", "pyrealsense2",
                     "pressb.runtime.trainer", "pressb.runtime.real_robot")
        class RejectHardwareAndTraining(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
                    raise AssertionError("Unexpected import: " + fullname)
        sys.meta_path.insert(0, RejectHardwareAndTraining())
        from pressb.registry import resolve
        assert resolve("task", "piper_real") == "pressb.runtime.real_robot:create_backend"
        from pressb.runtime import server
        calls = []
        server.run_service = lambda kind, path: calls.append((kind, path.name))
        from pressb.experiments.cli import main
        assert main(["real_robot", "--config", sys.argv[1], "--resolved-dir", sys.argv[2]]) == 0
        assert calls == [("real_robot", "simulation.json")]
        assert not any(name in sys.modules for name in forbidden)
    """)
    result = subprocess.run([sys.executable, "-c", source, str(config), str(tmp_path / "resolved")],
                            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


class PhysicalBackend:
    def __init__(self, *, physical=True):
        self.physical = physical
        self.closed = False

    def describe(self):
        return dict(schema_version=2, node_type="simulation", task_id="physical-fixture", num_envs=1,
                    environment_type="real_robot" if self.physical else "simulation",
                    capabilities=["read_only_observation"] if self.physical else [],
                    execution={"actuation_enabled": False},
                    observation=dict(state=dict(names=["joint1"], units=["rad"]), cameras=[]),
                    action=dict(representation="joint_position", frame="robot_base", names=["joint1"],
                                units=["rad"], horizon=1, control_hz=30),
                    control_state_dim=0, task_features_dim=0, reset_mode="indexed",
                    scheduling="independent", discount=dict(gamma=.99, timebase_hz=30))

    def close(self):
        self.closed = True


@pytest.mark.parametrize("node_kind", ["real_robot", "simulation"])
def test_physical_backend_http_defaults_and_sigterm_close_hardware(tmp_path, monkeypatch, capsys, node_kind):
    backend = PhysicalBackend()
    node = tmp_path / "node.json"
    node.write_text(json.dumps({"backend": {"factory": "fixture:create_backend"}}))
    monkeypatch.setattr(server, "load_backend", lambda _: backend)
    calls = []

    class FakeHTTP:
        def __init__(self, address, handlers, *, token):
            assert "/observe" in handlers
            with pytest.raises(ValueError, match="Actuation is disabled"):
                handlers["/reset"]({})
            calls.append((address, handlers["/health"]({}), token))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            calls.append("http_closed")

        def serve_forever(self, *, poll_interval):
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)

    monkeypatch.setattr(server, "RPCServer", FakeHTTP)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    server.run_service(node_kind, node)
    assert calls[0][0] == ("127.0.0.1", 8101)
    assert calls[0][1]["node_type"] == "simulation"
    assert calls[0][1]["environment_type"] == "real_robot"
    assert calls[0][2] is None
    assert calls[1] == "http_closed"
    assert backend.closed
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm
    ready = json.loads(capsys.readouterr().out)
    assert ready["endpoint"] == "http://127.0.0.1:8101"


def test_real_robot_rejects_simulator_metadata_before_serving_and_closes(tmp_path, monkeypatch):
    backend = PhysicalBackend(physical=False)
    node = tmp_path / "node.json"
    node.write_text(json.dumps({"backend": {"factory": "fixture:create_backend"}}))
    monkeypatch.setattr(server, "load_backend", lambda _: backend)
    monkeypatch.setattr(server, "RPCServer", lambda *a, **kw: pytest.fail("Must not serve a simulator"))
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError, match="requires a physical task backend"):
        server.run_service("real_robot", node)
    assert backend.closed
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm


def test_real_robot_restores_sigterm_if_backend_loading_fails(tmp_path, monkeypatch):
    node = tmp_path / "node.json"
    node.write_text(json.dumps({"backend": {"factory": "fixture:create_backend"}}))
    def fail(_):
        raise RuntimeError("hardware unavailable")
    monkeypatch.setattr(server, "load_backend", fail)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError, match="hardware unavailable"):
        server.run_service("real_robot", node)
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm


@pytest.mark.parametrize("method,recipe", [("base", "eval"), ("dsrl", "dsrl")])
def test_fr3_physical_recipe_matches_policy_before_connect(tmp_path, monkeypatch, method, recipe):
    from pressb.robots.franka import real
    from pressb.runtime.contract import negotiate

    config = load_experiment(PROJECT_ROOT / f"configs/experiments/fr3_real_pi05_{recipe}.json")
    node = config["nodes"]["simulation"]
    node["backend"]["config"]["task"]["config"]["directory"] = str(tmp_path / "operator")
    assert config["factories"]["task"] == "pressb.runtime.real_robot:create_franka_backend"
    assert config["nodes"]["trainer"]["method"] == method
    monkeypatch.setattr(real, "_load_native", lambda: pytest.fail("Offline health opened the SDK"))
    backend = server.load_backend(node["backend"], kind="task")
    try:
        health = backend.describe()
        assert health["environment_type"] == "real_robot" and health["num_envs"] == 1
        assert health["execution"]["actuation_enabled"] is False
        assert not health["execution"]["return_to_initial"]["enabled"]
        assert "read_only_observation" in health["capabilities"]
        names = [f"joint{i}" for i in range(1, 8)] + ["gripper_closure"]
        units = ["rad"] * 7 + ["unitless"]
        # FR3 pi0.5 wire contract recorded alongside the trained checkpoint.
        model = dict(schema_version=2, node_type="inference", model_id="fr3-pap",
                     observation=dict(state=dict(names=names, units=units), cameras=["base", "wrist"]),
                     action=dict(representation="joint_position", frame="robot_base", names=names,
                                 units=units, horizon=16, control_hz=25), feature_dim=2048,
                     noise={"shape": [16, 32]},
                     capabilities=["encode", "decode", "features", "context_cache", "external_initial_noise"])
        assert negotiate(model, health, method, 16)["action_dim"] == 8
        assert backend.status()["robot"]["submitted_commands"] == 0
    finally:
        backend.close()
