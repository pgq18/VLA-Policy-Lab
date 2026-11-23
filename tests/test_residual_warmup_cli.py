"""Evaluation may reuse a training config without activating shadow training."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import sys

import pytest

from pressb.experiments.cli import main
from pressb.experiments.config import PROJECT_ROOT
import pressb.runtime.trainer as trainer_module


def _training_run():
    # These settings are intentionally training-only; evaluation must ignore
    # them before TrainerConfig validation and retain the selected RL policy.
    return {"method": "dsrl", "mode": "train",
            "residual_warmup": {"learner": {"residual_scale": [.02]}},
            "residual_initialize_from": "/unused-training-source.pt"}


def _capture_runner(monkeypatch):
    captured = []

    class Runner:
        def __init__(self, config, *, observer=None):
            self.config = deepcopy(config if isinstance(config, dict) else asdict(config))
            self.observer = observer
            captured.append(self.config)

        def stop(self):
            pass

        def run(self):
            status = {"state": "completed"}
            if self.observer:
                self.observer.close(status)
            return status

    monkeypatch.setattr(trainer_module, "GenericTrainer", Runner)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    return captured


@pytest.mark.parametrize("command", ["eval", "collect"])
def test_experiment_eval_discards_training_only_initialization_and_warmup(tmp_path, monkeypatch, command):
    captured = _capture_runner(monkeypatch)
    run = _training_run()
    run["output_dir"] = str(tmp_path / "run")
    document = {"schema_version": 1, "name": "warmup_cli", "task": {"plugin": "pap", "config": {}},
                "model": {"plugin": "walloss", "config": {}}, "run": run}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document))
    checkpoint = tmp_path / "trained-dsrl.pt"
    assert main([command, "--config", str(path), "--resume", str(checkpoint),
                 "--resolved-dir", str(tmp_path / "resolved")]) == 0
    assert captured[0]["method"] == "dsrl" and captured[0]["mode"] == "eval"
    assert captured[0]["checkpoint"]["resume"] == str(checkpoint)
    assert "residual_warmup" not in captured[0]
    assert "residual_initialize_from" not in captured[0]
    resolved = json.loads((tmp_path / "resolved/trainer.json").read_text())
    assert "residual_warmup" not in resolved and "residual_initialize_from" not in resolved
    assert json.loads(path.read_text()) == document  # Source training config stays intact.


@pytest.mark.parametrize("explicit_mode", [False, True])
def test_legacy_cli_eval_discards_training_only_fields(tmp_path, monkeypatch, explicit_mode):
    captured = _capture_runner(monkeypatch)
    path = tmp_path / "trainer.json"
    run = _training_run()
    if not explicit_mode:
        run["mode"] = "eval"
    path.write_text(json.dumps(run))
    specification = importlib.util.spec_from_file_location(
        "pressb_warmup_test_cli", PROJECT_ROOT / "scripts/run_rl_training.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    arguments = ["run_rl_training.py", "--config", str(path), "--resume", "/trained-dsrl.pt"]
    if explicit_mode:
        arguments.extend(["--mode", "eval"])
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    assert captured[0]["method"] == "dsrl" and captured[0]["mode"] == "eval"
    assert captured[0]["checkpoint"]["resume"] == "/trained-dsrl.pt"
    # Dataclass defaults may add the fields back as None; neither is active.
    assert captured[0].get("residual_warmup") is None
    assert captured[0].get("residual_initialize_from") is None
    assert json.loads(path.read_text()) == run
