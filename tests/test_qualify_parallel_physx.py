"""Offline checks for the real-physics qualifier's comparison, without Isaac."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("qualify_parallel_physx", SCRIPTS / "qualify_parallel_physx.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_run(directory, solver, *, delta=0., duplicate=False, commands="identical"):
    directory.mkdir()
    stages = {}
    for phase in ("hold_timeout", "success_replay_00_floor24"):
        success = phase != "hold_timeout"
        stages[phase] = [dict(env_id=i, terminated=success, truncated=not success,
                             info=dict(episode_physics_steps=2, success=success)) for i in range(2)]
    (directory / "report.json").write_text(json.dumps(dict(physics_backend_requested=solver,
        operations_passed=True, commands_sha256=commands, num_envs=2, stages=stages)))
    (directory / "config.json").write_text(json.dumps(dict(physics_backend=solver, num_envs=2,
        max_seconds=15, output=str(directory), gpu=0 if solver == "cpu" else 1)))
    state = np.zeros((8, 8)); state[:, 0] = delta
    np.savez(directory / "trajectory.npz", phase=np.array(["hold_timeout"] * 4 + ["success_replay_00_floor24"] * 4),
             env_id=np.array([0, 0, 1, 1] * 2), tick=np.array([1, 1 if duplicate else 2, 1, 2] * 2),
             state=state, joints=np.zeros((8, 8)))


def test_comparison_aligns_actual_ticks_without_requiring_bitwise_equality(tmp_path):
    fixture_run(tmp_path / "cpu", "cpu")
    fixture_run(tmp_path / "gpu", "gpu", delta=.001)
    assert module.compare(tmp_path / "cpu", tmp_path / "gpu", tmp_path / "result.json") == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["aligned_tick_samples"] == 8
    assert result["tcp_position_difference_m"]["max"] == pytest.approx(.001)
    assert result["passed"]


def test_comparison_reports_trajectory_drift_as_failure(tmp_path):
    fixture_run(tmp_path / "cpu", "cpu")
    fixture_run(tmp_path / "gpu", "gpu", delta=.003)
    assert module.compare(tmp_path / "cpu", tmp_path / "gpu", tmp_path / "result.json") == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert not result["checks"]["within_position_tolerance"]


@pytest.mark.parametrize("change", ["duplicate", "commands"])
def test_comparison_rejects_unmatched_inputs(tmp_path, change):
    fixture_run(tmp_path / "cpu", "cpu")
    fixture_run(tmp_path / "gpu", "gpu", **({"duplicate": True} if change == "duplicate" else {"commands": "different"}))
    with pytest.raises(RuntimeError, match="Duplicate|Replay commands differ"):
        module.compare(tmp_path / "cpu", tmp_path / "gpu", tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()
