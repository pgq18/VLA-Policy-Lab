"""Exercise a sharded simulator through its public RPC contract and real worker services."""
from contextlib import closing
from copy import deepcopy
import threading

import pytest

from pressb.runtime.simulation import SimulationService
from pressb.runtime.simulation_pool import SimulationPool
from test_rl_nodes_contract_simulation import Backend, step_payload


class ShardBackend(Backend):
    def __init__(self, index, envs=2):
        super().__init__()
        self.index = index
        self.count = [0] * envs
        self.reset_seeds = []
        self.reset_options = []

    def describe(self):
        return {**super().describe(), "num_envs": len(self.count), "initial_reset": "all_envs",
                "scheduling": "all_active", "equal_execution_horizon": True,
                "capabilities": ["per_env_reset_seeds"]}

    def observation(self, i):
        row = super().observation(i)
        row["state"][1] = float(10 * self.index + i)
        return row

    def reset(self, env_ids, seed, options, *, env_seeds=None):
        self.reset_seeds.append((seed, env_seeds))
        self.reset_options.append(deepcopy(options))
        return super().reset(env_ids, seed, options)


class Client:
    def __init__(self, index, envs=2):
        self.service = SimulationService(ShardBackend(index, envs))
        self.hook = None
        self.closed = False

    def call(self, path, payload=None):
        if self.hook:
            self.hook(path)
        return self.service.handlers[path](payload)

    def close(self):
        self.closed = True


def reset_request(ids, *, rid="reset"):
    return {"run_id": "test", "request_id": rid, "env_ids": ids, "seed": 515, "options": {}}


def step_request(result):
    return {**step_payload(result), "run_id": "test"}


def test_unequal_shards_parallel_step_order_seeds_and_indexed_reset():
    clients = [Client(0, 2), Client(1, 3)]
    with closing(SimulationPool(clients)) as pool:
        assert pool.health()["num_envs"] == 5
        gate = threading.Barrier(2)

        def both_workers(path):
            if path in ("/step", "/reset"):
                gate.wait(timeout=3)

        for client in clients:
            client.hook = both_workers
        order = [2, 1, 4, 0, 3]
        request = reset_request(order)
        request["env_seeds"] = {str(i): 100 + i for i in order}
        request["options"] = {"by_env": {"1": {"color": "red"}, "3": {"color": "blue"}}}
        reset = pool.reset(request)
        assert [r["env_id"] for r in reset["items"]] == order
        assert [r["observation"]["state"][1] for r in reset["items"]] == [10., 1., 12., 0., 11.]
        left, right = [client.service.backend for client in clients]
        assert left.reset_seeds[0][1] == {0: 100, 1: 101}
        assert right.reset_seeds[0][1] == {0: 102, 1: 103, 2: 104}
        assert left.reset_options[0]["by_env"] == {"1": {"color": "red"}}
        assert right.reset_options[0]["by_env"] == {"1": {"color": "blue"}}
        for row in reset["items"]:
            mapping = row["info"]["simulation_shard"]
            assert mapping["global_env_id"] == row["env_id"]
            assert mapping["shard"] == int(row["env_id"] >= 2)
            assert mapping["local_env_id"] == row["env_id"] - 2 * mapping["shard"]
            assert mapping["requested_seed"] == mapping["worker_seed"] == 100 + row["env_id"]
        step = pool.step(step_request(reset))
        assert [r["env_id"] for r in step["items"]] == order
        assert [r["episode_id"] for r in step["items"]] == [r["episode_id"] for r in reset["items"]]
        assert {r["env_id"] for r in step["items"] if r["terminated"]} == {0, 2}
        calls_left = deepcopy(left.calls)
        untouched_slots = deepcopy(clients[0].service.slots)
        for client in clients:
            client.hook = None
        partial = pool.reset(reset_request([2], rid="partial"))
        assert [r["env_id"] for r in partial["items"]] == [2]
        assert partial["items"][0]["episode_id"] != reset["items"][0]["episode_id"]
        assert left.calls == calls_left
        assert clients[0].service.slots == untouched_slots
        assert right.count == [0, 1, 1]
        closed = pool.close_run({"run_id": "test"})
        assert closed["closed"]
        assert all(client.service.run_id is None for client in clients)
    assert all(client.closed for client in clients)


def test_shared_seed_is_reproducible_and_distinct_across_shards():
    def collect():
        clients = [Client(0), Client(1)]
        with closing(SimulationPool(clients)) as pool:
            pool.reset(reset_request([0, 1, 2, 3]))
            return [client.service.backend.reset_seeds[0][0] for client in clients]

    first, second = collect(), collect()
    assert first == second
    assert first[0] == 515
    assert first[1] != first[0]


def test_requests_are_idempotent_and_global_validation_precedes_physics():
    clients = [Client(0), Client(1)]
    with closing(SimulationPool(clients, request_cache_size=1)) as pool:
        request = reset_request([3, 0, 1, 2])
        reset = pool.reset(request)
        assert pool.reset(deepcopy(request)) == reset
        with pytest.raises(ValueError):
            pool.reset({**request, "seed": 516})
        step = step_request(reset)
        for invalid in (step["items"][:-1], [*step["items"][:-1], step["items"][0]]):
            with pytest.raises(ValueError):
                pool.step({**step, "items": invalid})
        assert all(client.service.backend.count == [0, 0] for client in clients)
        response = pool.step(step)
        assert pool.step(deepcopy(step)) == response
        with pytest.raises(ValueError):
            pool.reset(request)
        assert all(client.service.backend.count == [1, 1] for client in clients)


def test_uncertain_response_after_one_shard_advanced_is_never_retried():
    clients = [Client(0), Client(1)]
    with closing(SimulationPool(clients)) as pool:
        reset = pool.reset(reset_request([0, 1, 2, 3]))
        request = step_request(reset)
        after_left_step = threading.Event()
        original = clients[0].call

        def left(path, payload=None):
            result = original(path, payload)
            if path == "/step":
                after_left_step.set()
            return result

        def right_failure(path):
            if path == "/step":
                assert after_left_step.wait(3)
                raise ConnectionError("unknown worker execution state")

        clients[0].call = left
        clients[1].hook = right_failure
        with pytest.raises(RuntimeError):
            pool.step(request)
        assert clients[0].service.backend.count == [1, 1]
        for action in (lambda: pool.step(request), lambda: pool.reset(reset_request([0], rid="r2"))):
            with pytest.raises(RuntimeError):
                action()
        assert clients[0].service.backend.count == [1, 1]
        assert clients[1].service.backend.count == [0, 0]
    assert all(client.service.run_id for client in clients)


def test_pool_rejects_incompatible_or_preowned_workers():
    a, b = Client(0), Client(1)
    b.service.metadata["task_id"] = "other-task"
    with pytest.raises((ValueError, RuntimeError)):
        SimulationPool([a, b])
    a, b = Client(0), Client(1)
    b.service.reset(reset_request([0, 1]))
    with pytest.raises((ValueError, RuntimeError)):
        SimulationPool([a, b])
    assert b.service.run_id == "test"


def test_pool_shutdown_releases_healthy_workers_for_the_next_pool():
    clients = [Client(0), Client(1)]
    with closing(SimulationPool(clients)) as first:
        first.reset(reset_request([0, 1, 2, 3]))
        assert all(client.service.run_id for client in clients)
    replacement_clients = [Client(0), Client(1)]
    for new, previous in zip(replacement_clients, clients, strict=True):
        new.service = previous.service
    with closing(SimulationPool(replacement_clients)) as second:
        result = second.reset(reset_request([0, 1, 2, 3]))
        assert len(result["items"]) == 4


def test_public_http_pool_with_two_http_workers():
    from contextlib import ExitStack
    from pressb.runtime.client import NodeClient
    from test_expert_sync import running_node

    with ExitStack() as stack:
        endpoints = [stack.enter_context(running_node(Client(i).service, "http")) for i in range(2)]
        workers = [NodeClient(url, token="test-token", timeout=5) for url in endpoints]
        pool = stack.enter_context(closing(SimulationPool(workers)))
        url = stack.enter_context(running_node(pool, "http"))
        client = stack.enter_context(closing(NodeClient(url, token="test-token", timeout=5)))
        assert client.call("/health")["num_envs"] == 4
        order = [2, 3, 0, 1]
        reset = client.call("/reset", reset_request(order))
        result = client.call("/step", step_request(reset))
        assert [r["env_id"] for r in result["items"]] == order
        status = client.call("/status")
        assert status["run_id"] == "test"
        assert {int(key) for key in status["environments"]} == set(order)
        assert client.call("/close_run", {"run_id": "test"})["closed"]
