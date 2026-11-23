"""Real replica RPC tests; no accelerator or pretrained model is required."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from pressb.runtime.client import NodeClient
from pressb.runtime.expert_sync import publish_expert, state_sha256, tree_from_wire
from pressb.runtime.inference import InferenceService
from pressb.runtime.inference_pool import InferencePool, InferencePoolError
from test_expert_sync import ToyBackend, chain_request, observation, running_node, upload_payload


@contextmanager
def replicas(*, count=2, cache_size=8, ttl=600., clock=None, backend_type=ToyBackend,
             transform=None, transport="http"):
    services = [InferenceService(backend_type(), cache_size=cache_size, cache_ttl=ttl,
                                 batch_size=4, action_expert={}, **({"clock": clock} if clock else {}))
                for _ in range(count)]
    with ExitStack() as stack:
        endpoints = [stack.enter_context(running_node(transform(i, service) if transform else service, transport))
                     for i, service in enumerate(services)]
        clients = [NodeClient(url, timeout=3, token="test-token") for url in endpoints]
        pool = InferencePool(clients, **({"clock": clock} if clock else {}))
        try:
            yield pool, services
        finally:
            pool.close()
            for service in services:
                service.close()


class HandlerOverride:
    def __init__(self, service, route, replacement):
        self.service, self.route, self.replacement = service, route, replacement

    @property
    def handlers(self):
        return {**self.service.handlers, self.route: self.replacement}


def encoded_request(count, *, base=False):
    return {"observations": [observation(i + .125) for i in range(count)],
            "seeds": list(range(10, 10 + count)), "include_base_actions": base}


@pytest.mark.parametrize("count", [1, 3, 8, 13])
def test_parallel_chain_preserves_input_order_seed_and_conditioning(count):
    reference = InferenceService(ToyBackend(), cache_size=16, batch_size=4, action_expert={})
    request = chain_request(count=count)
    expected = reference.action_expert.sample_chain(request)
    with replicas() as (pool, workers):
        with running_node(pool, "http") as endpoint:
            client = NodeClient(endpoint, timeout=3, token="test-token")
            actual = client.call("/sample_chain", request)
            client.close()
        assert len(actual["items"]) == count
        assert actual["policy_version"] == 0
        assert actual["weights_sha256"] == expected["weights_sha256"]
        assert sum(row["items"] for row in actual["replica_timings"]) == count
        assert all(row["seconds"] >= 0 for row in actual["replica_timings"])
        for actual_row, expected_row in zip(actual["items"], expected["items"], strict=True):
            assert actual_row == expected_row
            assert torch.equal(tree_from_wire(actual_row["chain"]), tree_from_wire(expected_row["chain"]))
        health = pool.health()
        assert health["encoded_count"] == count == health["decoded_count"]
        assert health["parallel_inference"]["replicas"] == 2
        assert health["parallel_inference"]["sharding"] == "contiguous_balanced"
        assert workers[0].encoded_count == (count + 1) // 2
        assert workers[1].encoded_count == count // 2
        assert not pool.contexts
    reference.close()


def test_worker_calls_really_overlap_and_keep_contiguous_partitions():
    gate = threading.Barrier(2)

    class ConcurrentBackend(ToyBackend):
        def encode_batch(self, observations, seeds):
            gate.wait(timeout=2)
            self.seen = [row["state"][0] for row in observations]
            return super().encode_batch(observations, seeds)

    with replicas(backend_type=ConcurrentBackend) as (pool, workers):
        result = pool.sample_chain(chain_request(count=8))
        assert len(result["items"]) == 8
        assert workers[0].backend.seen == pytest.approx([0, .1, .2, .3])
        assert workers[1].backend.seen == pytest.approx([.4, .5, .6, .7])


def test_encode_decode_release_routing_and_ttl_are_observation_owned():
    clock = [0.]
    with replicas(cache_size=3, ttl=5., clock=lambda: clock[0]) as (pool, workers):
        result = pool.encode(encoded_request(6, base=True))
        rows = result["items"]
        keys = [row["context_id"] for row in rows]
        assert all(key.startswith(pool.generation + ".") for key in keys)
        assert [key.split(".")[1] for key in keys] == ["0"] * 3 + ["1"] * 3
        order = [5, 0, 4, 1, 3, 2]
        decoded = pool.decode({"items": [{"context_id": keys[i]} for i in order]})["items"]
        assert [row["context_id"] for row in decoded] == [keys[i] for i in order]
        for row, index in zip(decoded, order, strict=True):
            assert row["actions"] == rows[index]["base_actions"]
        clock[0] = 4.9
        pool.decode({"items": [{"context_id": keys[0]}]})
        assert pool.release({"context_ids": [keys[4], keys[4], "foreign"]}) == {"released": 1}
        assert pool.release({"context_ids": [keys[4]]}) == {"released": 0}
        clock[0] = 5.
        assert pool.health()["cache"]["size"] == 0
        with pytest.raises(ValueError, match="expired"):
            pool.decode({"items": [{"context_id": keys[0]}]})
        assert pool.health()["ready"]  # Local invalid input does not poison workers.
        assert pool.release({"context_ids": keys}) == {"released": 0}


def test_worker_cache_eviction_and_atomic_encode_retirement():
    with replicas(cache_size=2) as (pool, workers):
        old = pool.encode(encoded_request(4))["items"]
        keys = [row["context_id"] for row in old]
        payload = encoded_request(1)
        payload["release_context_ids"] = [keys[2], "unknown", keys[2]]
        new = pool.encode(payload)
        assert new["released_context_ids"] == [keys[2], "unknown"]
        # Worker 0 gets the replacement and evicts its own oldest entry;
        # worker 1 gets a release-only request and must keep its other entry.
        for key in (keys[0], keys[2]):
            with pytest.raises(ValueError, match="expired"):
                pool.decode({"items": [{"context_id": key}]})
        for key in (keys[1], keys[3], new["items"][0]["context_id"]):
            assert len(pool.decode({"items": [{"context_id": key}]})["items"]) == 1
        assert [len(worker.contexts) for worker in workers] == [2, 1]


def test_one_shot_infer_keeps_cache_and_preserves_external_noise_order():
    with replicas() as (pool, _):
        cached = pool.encode(encoded_request(5, base=True))["items"]
        original = list(pool.contexts)
        payload = encoded_request(5)
        payload.pop("include_base_actions")
        payload["initial_noises"] = [np.full((3, 2), i).tolist() for i in range(5)]
        expected = InferenceService(ToyBackend()).infer(payload)
        actual = pool.infer(payload)
        assert actual["items"] == expected["items"]
        assert list(pool.contexts) == original
        assert len(pool.decode({"items": [{"context_id": row["context_id"]} for row in cached]})["items"]) == 5


def test_binary_png_public_payload_is_losslessly_bridged_to_http_workers():
    from test_rl_nodes_inference import Backend, observation as image_observation
    services = [InferenceService(Backend(horizon=3, action_dim=2, latent_dim=2), cache_size=4)
                for _ in range(2)]
    payload = {"observations": [image_observation(i) for i in range(3)], "seeds": [4, 5, 6]}
    expected = InferenceService(Backend(horizon=3, action_dim=2, latent_dim=2)).infer(deepcopy(payload))
    for row in payload["observations"]:
        image = row["images"]["up"]
        image.update(encoding="png_bytes", data=base64.b64decode(image["data"]))
    with ExitStack() as stack:
        clients = [NodeClient(stack.enter_context(running_node(service, "http")), token="test-token")
                   for service in services]
        pool = InferencePool(clients)
        try:
            actual = pool.infer(payload)
            assert actual["items"] == expected["items"]
            assert all(isinstance(row["images"]["up"]["data"], bytes) for row in payload["observations"])
        finally:
            pool.close()
            for service in services:
                service.close()


def test_two_weight_publications_update_every_replica_and_clear_old_contexts():
    with replicas() as (pool, workers):
        with running_node(pool, "http") as endpoint:
            client = NodeClient(endpoint, timeout=3, token="test-token")
            initial = client.call("/sample_chain", chain_request(count=5))
            for version in (1, 2):
                cached = client.call("/encode", encoded_request(2))["items"]
                state = {"bias": torch.tensor([version * .1, version * -.2])}
                committed = publish_expert(client, state, version - 1, chunk_bytes=128)
                assert committed["replicas_committed"] == 2
                assert committed["policy_version"] == version
                assert all(worker.action_expert.weights.sha256 == state_sha256(state) for worker in workers)
                health = client.call("/health")
                assert health["ready"] and health["action_expert"]["policy_version"] == version
                assert health["cache"]["size"] == 0
                with pytest.raises(ValueError, match="expired"):
                    pool.decode({"items": [{"context_id": cached[0]["context_id"]}]})
                updated = client.call("/sample_chain", chain_request(version=version, count=5))
                assert updated["items"][0]["actions"] != initial["items"][0]["actions"]
                assert all(row["policy_version"] == version for row in updated["items"])
            client.close()


def test_real_dppo_trainer_runs_three_updates_through_the_rpc_pool(tmp_path):
    from test_dppo_training import ToyBackend as TrainingBackend, make_runner
    runner, _, simulation = make_runner(tmp_path, steps=12)
    original = state_sha256(runner.expert.state_dict())
    with replicas(backend_type=TrainingBackend) as (pool, workers):
        with running_node(pool, "http") as endpoint:
            runner.inference = NodeClient(endpoint, timeout=3, token="test-token")
            status = runner.run()
        assert status["state"] == "completed" and status["transitions"] == 12
        assert status["policy_version"] == 3
        assert status["actor_updates"] > 0 and status["critic_updates"] > 0
        current = state_sha256(runner.expert.state_dict())
        assert current != original == json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])["expert_sha256"]
        assert all(worker.action_expert.weights.sha256 == current for worker in workers)
        assert pool.health()["action_expert"]["weights_sha256"] == current
        assert simulation.closed_run_id == runner.run_id
        assert len(list((tmp_path / "checkpoints").glob("step_*.pt"))) == 3


@pytest.mark.parametrize("lose_ack", [False, True])
def test_failed_replica_commit_locks_entire_pool_without_mixed_policy_samples(lose_ack):
    def transform(index, service):
        if index == 0:
            return service
        original = service.action_expert.handlers["/weights_commit"]

        def fail(payload):
            if lose_ack:
                original(payload)
            raise RuntimeError("injected lost commit ACK" if lose_ack else "injected commit failure")
        return HandlerOverride(service, "/weights_commit", fail)

    with replicas(transform=transform) as (pool, workers):
        with running_node(pool, "http") as endpoint:
            client = NodeClient(endpoint, timeout=3, token="test-token")
            with pytest.raises(Exception, match="pool locked"):
                publish_expert(client, {"bias": torch.tensor([.2, .3])}, 0, chunk_bytes=128)
            client.close()
        assert workers[0].action_expert.weights.version == 1
        assert workers[1].action_expert.weights.version == (1 if lose_ack else 0)
        assert pool.version == 0  # Never advertises a successful partial commit.
        with pytest.raises(InferencePoolError, match="locked"):
            pool.sample_chain(chain_request(count=2))
        with pytest.raises(InferencePoolError, match="locked"):
            pool.health()


def test_upload_barrier_blocks_sampling_and_abort_keeps_original_experts():
    with replicas() as (pool, workers):
        payload, blob = upload_payload({"bias": torch.tensor([.2, .3])})
        begun = pool.weights_begin(payload)
        assert pool.health()["ready"] is False
        with pytest.raises(ValueError, match="publication is in progress"):
            pool.sample_chain(chain_request(count=2))
        with pytest.raises(ValueError, match="incomplete"):
            pool.weights_commit({"upload_id": begun["upload_id"]})
        with pytest.raises(ValueError, match="sequential"):
            pool.weights_append({"upload_id": begun["upload_id"], "offset": 1,
                                 "data": base64.b64encode(blob).decode()})
        assert pool.weights_abort({"upload_id": begun["upload_id"]})["aborted"]
        assert pool.health()["ready"]
        assert all(worker.action_expert.weights.version == 0 for worker in workers)


def test_external_replica_update_is_detected_before_new_sampling():
    with replicas() as (pool, workers):
        workers[1].action_expert.weights.version = 1
        before = [worker.encoded_count for worker in workers]
        with pytest.raises(InferencePoolError, match="different action expert"):
            pool.sample_chain(chain_request(count=2))
        assert [worker.encoded_count for worker in workers] == before


@pytest.mark.parametrize("fault", ["seed", "state", "version", "count", "noise"])
def test_invalid_request_does_not_partially_execute_or_poison_pool(fault):
    with replicas(cache_size=2) as (pool, workers):
        payload = chain_request(count=3)
        if fault == "seed":
            payload["seeds"][-1] = True
        elif fault == "state":
            payload["observations"][-1]["state"][0] = float("nan")
        elif fault == "version":
            payload["expected_policy_version"] = 2
        elif fault == "count":
            payload = chain_request(count=5)
        else:
            payload = encoded_request(3)
            payload.pop("include_base_actions")
            payload["initial_noises"] = [[[0.]]] * 3
        with pytest.raises(ValueError):
            (pool.infer if fault == "noise" else pool.sample_chain)(payload)
        assert all(worker.encoded_count == 0 for worker in workers)
        assert pool.health()["ready"]


def test_close_only_closes_pool_connections_not_externally_owned_workers():
    with replicas() as (pool, workers):
        pool.close()
        pool.close()
        with pytest.raises(InferencePoolError, match="closed"):
            pool.health()
        assert not any(worker.backend.closed for worker in workers)


def test_replica_identity_mismatch_rejected_before_any_observation():
    def transform(index, service):
        if index:
            service.metadata["model_id"] = "wrong-checkpoint"
        return service
    with pytest.raises(ValueError, match="share one model"):
        with replicas(transform=transform):
            pytest.fail("inconsistent pool was accepted")


def test_configured_cli_serves_and_exits_without_stopping_worker_nodes(tmp_path):
    workers = [InferenceService(ToyBackend(), cache_size=4, batch_size=4, action_expert={}) for _ in range(2)]
    with ExitStack() as stack:
        endpoints = [stack.enter_context(running_node(worker, "http")) for worker in workers]
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        config = {"workers": [{"endpoint": endpoint, "token_env": "TEST_WORKER_TOKEN", "timeout": 3}
                              for endpoint in endpoints],
                  "server": {"host": "127.0.0.1", "port": port, "transport": "http", "token_env": "TEST_POOL_TOKEN"},
                  "service": {"max_batch_size": 4}}
        path = tmp_path / "pool.json"
        path.write_text(json.dumps(config))
        command = [sys.executable, "scripts/serve_inference_pool.py", "--config", str(path)]
        log = tmp_path / "pool.log"
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TEST_WORKER_TOKEN": "test-token", "TEST_POOL_TOKEN": "pool-secret"},
                stdout=stream, stderr=subprocess.STDOUT)
        client = None
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                client = NodeClient(f"http://127.0.0.1:{port}", token="pool-secret", timeout=.5)
                try:
                    if client.call("/health")["ready"]:
                        break
                except Exception:
                    client.close()
                    client = None
                    if process.poll() is not None:
                        pytest.fail(log.read_text())
                    time.sleep(.02)
            assert client is not None, log.read_text()
            assert len(client.call("/sample_chain", chain_request(count=3))["items"]) == 3
            changed = {"bias": torch.tensor([.2, -.3])}
            assert publish_expert(client, changed, 0)["replicas_committed"] == 2
        finally:
            if client is not None:
                client.close()
            process.terminate()
            process.wait(timeout=5)
        assert process.returncode == 0, log.read_text()
        assert "pool-secret" not in log.read_text() and "test-token" not in log.read_text()
        assert all(worker.health()["ready"] and worker.action_expert.weights.version == 1 for worker in workers)


@pytest.mark.parametrize("worker_transport,public_transport", [("http", "ws"), ("ws", "http")])
def test_websocket_pool_uses_the_same_node_client_protocol(worker_transport, public_transport):
    pytest.importorskip("websockets.asyncio.server")
    with replicas(transport=worker_transport) as (pool, _):
        with running_node(pool, public_transport) as endpoint:
            client = NodeClient(endpoint, timeout=3, token="test-token")
            health = client.call("/health")
            assert health["schema_version"] == 2 and health["parallel_inference"]["replicas"] == 2
            result = client.call("/sample_chain", chain_request(count=3))
            assert len(result["items"]) == 3
            client.close()
