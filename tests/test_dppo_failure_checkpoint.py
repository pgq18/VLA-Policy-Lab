"""Interrupt real CPU DPPO execution and resume complete learner state."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from pressb.runtime.dppo_trainer import MasterAdamW
from pressb.runtime.expert_sync import state_sha256
from test_dppo_training import make_runner


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def lose_step_response(simulation, step_number):
    original = simulation.call

    def call(path, payload=None):
        result = original(path, payload)
        if path == "/step" and len(simulation.steps) == step_number:
            raise TimeoutError("response lost after action execution")
        return result

    simulation.call = call


def test_partial_rollout_failure_saves_and_resumes_full_learner_without_replay(tmp_path):
    runner, _, simulation = make_runner(tmp_path / "failed", steps=12)
    runner.config.checkpoint.interval = 100000
    lose_step_response(simulation, 4)
    with pytest.raises(TimeoutError, match="response lost"):
        runner.run()

    status = runner.health()
    assert status["state"] == "failed"
    assert status["execution_state"].startswith("unknown")
    assert len(simulation.steps) == 4  # No replay of the unknown action request.
    checkpoint = Path(status["last_checkpoint"])
    assert checkpoint.name.startswith("recovery_step_000000006_")
    payload = torch.load(checkpoint, weights_only=True)
    assert (payload["transitions"], payload["iterations"], payload["actor_updates"],
            payload["critic_updates"]) == (6, 1, 4, 4)
    assert payload["last_optimized_transitions"] == 4
    assert payload["recovery"]["confirmed_discarded_rollout_transitions"] == 2
    assert payload["discarded_rollout_transitions"] == 2
    assert payload["recovery"]["physics_request_in_flight"] is True
    assert payload["recovery"]["pending_physics_kind"] == "step"
    assert payload["recovery"]["pending_physics_env_ids"] == [0, 1]
    assert payload["recovery"]["unconfirmed_transitions"] is None
    assert payload["recovery"]["on_policy_rollout_discarded"] is True
    assert "records" not in payload and "replay" not in payload
    assert_tree_equal(payload["expert"], runner.expert.state_dict())
    assert_tree_equal(payload["value"], runner.value.state_dict())
    assert_tree_equal(payload["actor_optimizer"], runner.actor_optimizer.state_dict())
    assert_tree_equal(payload["value_optimizer"], runner.value_optimizer.state_dict())
    assert_tree_equal(payload["rng"], runner.rng.bit_generator.state)
    assert torch.equal(payload["torch_rng"], torch.get_rng_state())
    assert not list(checkpoint.parent.glob("step_*.pt"))
    assert not (checkpoint.parent / "latest.json").exists()
    assert json.loads((checkpoint.parent / "recovery_latest.json").read_text())["path"] == checkpoint.name

    resumed, inference, new_simulation = make_runner(tmp_path / "resumed", steps=12, checkpoint=checkpoint)
    resumed.config.checkpoint.interval = 100000
    original_negotiate = resumed._negotiate
    loaded = []

    def negotiate():
        original_negotiate()
        assert resumed.transitions == 6 and resumed.iterations == 1
        assert_tree_equal(resumed.actor_optimizer.state_dict(), payload["actor_optimizer"])
        assert_tree_equal(resumed.value_optimizer.state_dict(), payload["value_optimizer"])
        assert_tree_equal(resumed.expert.state_dict(), payload["expert"])
        loaded.append(True)

    resumed._negotiate = negotiate
    result = resumed.run()
    assert loaded and result["state"] == "completed" and result["transitions"] == 12
    assert result["actor_updates"] > 4 and result["critic_updates"] > 4
    assert state_sha256(resumed.expert.state_dict()) != state_sha256(payload["expert"])
    routes = [route for route, _ in inference.calls]
    assert routes.index("/weights_commit") < routes.index("/sample_chain")
    for item in new_simulation.resets[:2]:
        eid = item["env_ids"][0]
        assert item["seed"] == resumed._seed(eid, payload["resume_episode_serial"][eid], "reset")
        assert payload["resume_episode_serial"][eid] > payload["episode_serial"][eid]
    assert {row["request_id"] for row in simulation.steps}.isdisjoint(
        row["request_id"] for row in new_simulation.steps)
    assert Path(result["last_checkpoint"]).name == "step_000000012.pt"
    assert result["discarded_rollout_transitions"] == 2
    final = torch.load(result["last_checkpoint"], weights_only=True)
    assert final["discarded_rollout_transitions"] == 2


def test_second_interruption_does_not_count_previous_discarded_rollout_twice(tmp_path):
    first, _, sim = make_runner(tmp_path / "first", steps=12)
    first.config.checkpoint.interval = 100000
    lose_step_response(sim, 4)
    with pytest.raises(TimeoutError):
        first.run()
    second, _, sim2 = make_runner(tmp_path / "second", steps=12, checkpoint=first.last_checkpoint)
    second.config.checkpoint.interval = 100000
    lose_step_response(sim2, 1)
    with pytest.raises(TimeoutError):
        second.run()
    checkpoint = torch.load(second.last_checkpoint, weights_only=True)
    assert checkpoint["transitions"] == 6
    assert checkpoint["last_optimized_transitions"] == 4
    assert checkpoint["recovery"]["confirmed_discarded_rollout_transitions"] == 0
    assert checkpoint["discarded_rollout_transitions"] == 2


def test_lost_initial_reset_skips_unknown_episode_seed_when_resuming(tmp_path):
    runner, _, simulation = make_runner(tmp_path / "failed", steps=4)
    runner.config.checkpoint.interval = 100000
    original = simulation.call

    def call(path, payload=None):
        result = original(path, payload)
        if path == "/reset":
            raise TimeoutError("lost reset acknowledgement")
        return result

    simulation.call = call
    with pytest.raises(TimeoutError, match="reset acknowledgement"):
        runner.run()
    assert len(simulation.resets) == 1 and not simulation.steps
    path = runner.health()["last_checkpoint"]
    payload = torch.load(path, weights_only=True)
    assert payload["recovery"]["pending_physics_env_ids"] == [0]
    assert payload["resume_episode_serial"][0] == 1
    resumed, _, new_simulation = make_runner(tmp_path / "resumed", steps=4, checkpoint=path)
    result = resumed.run()
    assert result["state"] == "completed"
    assert new_simulation.resets[0]["seed"] == resumed._seed(0, 1, "reset")
    assert new_simulation.resets[0]["seed"] != simulation.resets[0]["seed"]


def test_completed_update_survives_lost_publication_acknowledgement(tmp_path):
    runner, client, _ = make_runner(tmp_path / "failed", steps=8)
    runner.config.checkpoint.interval = 100000
    original = client.call

    def lose_ack(route, payload=None):
        result = original(route, payload)
        if route == "/weights_commit":
            raise TimeoutError("commit succeeded but acknowledgement lost")
        return result

    client.call = lose_ack
    with pytest.raises(TimeoutError, match="acknowledgement lost"):
        runner.run()
    path = runner.health()["last_checkpoint"]
    payload = torch.load(path, weights_only=True)
    assert payload["recovery"]["publication_state"] == "pending"
    assert payload["policy_version"] == 0  # Last acknowledged version only.
    assert client.service.health()["action_expert"]["policy_version"] == 1
    assert payload["transitions"] == payload["last_optimized_transitions"] == 4
    assert payload["recovery"]["confirmed_discarded_rollout_transitions"] == 0
    assert payload["iterations"] == 1 and payload["actor_updates"] == payload["critic_updates"] == 4
    assert state_sha256(payload["expert"]) == state_sha256(client.service.backend.expert.state_dict())

    resumed, new_client, _ = make_runner(tmp_path / "resumed", steps=8, checkpoint=path)
    result = resumed.run()
    assert result["state"] == "completed" and result["iterations"] == 2
    assert result["actor_updates"] == result["critic_updates"] == 8
    sample_versions = [body["expected_policy_version"] for route, body in new_client.calls if route == "/sample_chain"]
    assert sample_versions == [1, 1]  # Restored expert was published on the fresh receiver.


def test_mid_optimizer_failure_never_labels_partial_update_as_recovery(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path / "failed", steps=12)
    original = MasterAdamW.step
    calls = []

    def fail_mid_update(optimizer, max_grad_norm):
        result = original(optimizer, max_grad_norm)
        calls.append(True)
        if len(calls) == 6:
            raise RuntimeError("optimizer failed after a partial mutation")
        return result

    monkeypatch.setattr(MasterAdamW, "step", fail_mid_update)
    with pytest.raises(RuntimeError, match="partial mutation"):
        runner.run()
    checkpoint = Path(runner.health()["last_checkpoint"])
    assert checkpoint.name == "step_000000004.pt"
    assert runner.health()["recovery_checkpoint"]["state"] == "skipped"
    assert not list(checkpoint.parent.glob("recovery*.pt"))
    assert not (checkpoint.parent / "recovery_latest.json").exists()
    saved = torch.load(checkpoint, weights_only=True)
    assert state_sha256(saved["expert"]) != state_sha256(runner.expert.state_dict())
    assert saved["actor_updates"] == 4 and saved["iterations"] == 1
    monkeypatch.setattr(MasterAdamW, "step", original)
    resumed, _, _ = make_runner(tmp_path / "resumed", steps=12, checkpoint=checkpoint)
    assert resumed.run()["state"] == "completed"


def test_recovery_storage_failure_preserves_original_transport_error(tmp_path, monkeypatch):
    runner, _, simulation = make_runner(tmp_path / "failed", steps=8)
    runner.config.checkpoint.interval = 100000
    lose_step_response(simulation, 1)

    def fail_save(*args, **kwargs):
        raise OSError("checkpoint storage full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(TimeoutError, match="response lost"):
        runner.run()
    assert runner.health()["error_type"] == "TimeoutError"
    recovery = runner.health()["recovery_checkpoint"]
    assert recovery["state"] == "failed" and recovery["error_type"] == "OSError"
    assert runner.health()["last_checkpoint"] is None
    assert len(simulation.steps) == 1


def test_failed_interval_save_keeps_bucket_and_recovery_separate(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path / "failed", steps=4)
    original = torch.save
    calls = []

    def fail_once(payload, path, *args, **kwargs):
        calls.append(str(path))
        if len(calls) == 1:
            raise OSError("interval write failed")
        return original(payload, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", fail_once)
    with pytest.raises(OSError, match="interval write failed"):
        runner.run()
    assert runner._last_checkpoint_bucket == 0
    path = runner.health()["last_checkpoint"]
    payload = torch.load(path, weights_only=True)
    assert payload["checkpoint_interval_bucket"] == 0
    assert "recovery" in payload and payload["transitions"] == 4
    assert not (Path(path).parent / "latest.json").exists()
    resumed, _, _ = make_runner(tmp_path / "resumed", steps=4, checkpoint=path)
    result = resumed.run()
    assert Path(result["last_checkpoint"]).name == "step_000000004.pt"
    assert resumed._last_checkpoint_bucket == 1


def test_recovery_resume_recalculates_regular_save_interval(tmp_path):
    first, _, simulation = make_runner(tmp_path / "failed", steps=24)
    first.config.checkpoint.interval = 2
    lose_step_response(simulation, 6)
    with pytest.raises(TimeoutError, match="response lost"):
        first.run()

    recovery = first.health()["last_checkpoint"]
    assert Path(recovery).name.startswith("recovery_step_000000010_")
    resumed, _, _ = make_runner(tmp_path / "resumed", steps=24, checkpoint=recovery)
    resumed.config.checkpoint.interval = 10
    resumed.run()
    files = sorted(path.name for path in (tmp_path / "resumed/checkpoints").glob("step_*.pt"))
    # The last regular save was at 8; recovery at 10 did not fulfill the new
    # 10-transition milestone. Save it after the next completed rollout.
    assert files == ["step_000000014.pt", "step_000000022.pt", "step_000000024.pt"]


def test_failure_checkpoint_can_stream_to_configured_storage_and_resume(tmp_path):
    from test_checkpoint_sink import sink, config
    with sink(tmp_path / "storage") as (_, endpoint):
        runner, _, simulation = make_runner(tmp_path / "failed", steps=8)
        runner.config.checkpoint.interval = 100000
        runner.config.checkpoint_sink = config(endpoint)
        lose_step_response(simulation, 3)
        with pytest.raises(TimeoutError, match="response lost"):
            runner.run()
        path = runner.health()["last_checkpoint"]
        assert path.startswith(endpoint) and "recovery_step_" in path
        assert not list((tmp_path / "failed").rglob("*.pt"))
        resumed, _, _ = make_runner(tmp_path / "resumed", steps=8)
        resumed.config.checkpoint.resume = path
        resumed.config.checkpoint_sink = config(endpoint)
        assert resumed.run()["state"] == "completed"
