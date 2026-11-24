import json
from pathlib import Path

import pytest

from pressb.runtime.checkpoint_reference import resolve_checkpoint


@pytest.mark.parametrize("overrides,error", [
    ({"state": "running"}, "completed"),
    ({"state": "stopped"}, "completed"),
    ({"state": "failed"}, "completed"),
    ({"mode": "eval"}, "dsrl training"),
    ({"method": None}, "dsrl training"),
    ({"method": "action_residual"}, "dsrl training"),
    ({"last_checkpoint": None}, "last_checkpoint"),
    ({"last_checkpoint": "https://sink.test/step.pt"}, "Only DPPO"),
])
def test_invalid_stage_source_is_rejected(tmp_path, overrides, error):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"state": "completed", "method": "dsrl",
                                  "last_checkpoint": "checkpoints/step.pt", **overrides}))
    with pytest.raises(ValueError, match=error):
        resolve_checkpoint(str(status), "dsrl")


def test_dppo_sink_and_missing_local_checkpoint(tmp_path):
    status = tmp_path / "status.json"
    url = "https://sink.test/run/step_000000029.pt"
    record = {"state": "completed", "mode": "train", "method": "dppo", "last_checkpoint": url}
    status.write_text(json.dumps(record))
    assert resolve_checkpoint(str(status), "dppo") == url
    assert resolve_checkpoint(url, "dppo") == url
    status.write_text(json.dumps({**record, "last_checkpoint": "checkpoints/missing.pt"}))
    with pytest.raises(FileNotFoundError, match="checkpoints/missing.pt"):
        resolve_checkpoint(str(status), "dppo")


def test_dppo_relative_checkpoint_resume_from_another_working_directory(tmp_path, monkeypatch):
    from test_dppo_training import make_runner

    monkeypatch.chdir(tmp_path)
    run = Path("outputs/first")
    first, _, _ = make_runner(run, steps=4)
    checkpoint = (tmp_path / first.run()["last_checkpoint"]).resolve()
    output = run.resolve()
    monkeypatch.chdir(tmp_path.parent)
    resumed, _, _ = make_runner(output, steps=8, checkpoint=output / "status.json")
    result = resumed.run()
    assert result["state"] == "completed" and result["transitions"] == 8
    assert resumed.config.checkpoint.resume == str(checkpoint)
