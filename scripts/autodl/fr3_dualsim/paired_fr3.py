"""Route one FR3 trainer to two dedicated, unmodified simulation nodes."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time

from pressb.tasks.fr3_pap.configuration import load_config, sample_objects


class PairedFR3Client:
    def __init__(self, clients, task_config):
        self.clients = clients
        self.cfg = load_config(task_config)
        self.executor = ThreadPoolExecutor(max_workers=len(clients))
        self.last_call_timing = {}
        self.faulted = False
        self.verified_layouts = [0] * len(clients)
        try:
            health = [c.call("/health") for c in clients]
            if len(health) != 2 or any(h["num_envs"] != 64 for h in health):
                raise ValueError("FR3 dual-sim requires two 64-environment workers")
            self.size = health[0]["num_envs"]
            self.total = self.size * len(clients)
            statuses = [c.call("/status") for c in clients]
            if any(s.get("run_id") or s.get("fault") or s.get("environments") for s in statuses):
                raise RuntimeError("Simulation worker is owned or faulted")
            identities = deepcopy(health)
            for h in identities:
                for key in ("gpu_ids", "worker_logs"):
                    h["execution"].pop(key, None)
            if identities[0] != identities[1]:
                raise ValueError(
                    "Simulation workers differ in task, renderer, or execution semantics"
                )
            self.meta = deepcopy(health[0])
            self.meta["num_envs"] = self.total
            self.meta["execution"].update(
                workers=2,
                camera_count=sum(h["execution"]["camera_count"] for h in health),
                gpu_ids=[gpu for h in health for gpu in h["execution"]["gpu_ids"]],
                worker_logs=[h["execution"]["worker_logs"] for h in health],
            )
            self.meta["simulation_shards"] = {
                "sizes": [self.size] * 2,
                "reset_layouts": "original sampler with global env_id",
            }
        except BaseException:
            self.close()
            raise

    def close(self):
        self.executor.shutdown(wait=True)
        for c in self.clients:
            c.close()

    def _parts(self, items, get_id):
        parts = [[] for _ in self.clients]
        ids = [get_id(item) for item in items]
        if len(set(ids)) != len(ids) or any(
            type(i) is not int or not 0 <= i < self.total for i in ids
        ):
            raise ValueError("Invalid global environment IDs")
        for item, eid in zip(items, ids):
            parts[eid // self.size].append(item)
        return parts

    def _dispatch(self, route, calls):
        def send(index, payload):
            t = time.perf_counter()
            result = self.clients[index].call(route, payload)
            return result, time.perf_counter() - t

        futures = {i: self.executor.submit(send, i, p) for i, p in calls.items()}
        try:
            replies = {i: f.result() for i, f in futures.items()}
        except BaseException:
            self.faulted = (
                True  # A sibling may have advanced. The trainer must restart from a checkpoint.
            )
            raise
        self.last_call_timing.update(
            {f"shard{i}_seconds": elapsed for i, (_, elapsed) in replies.items()}
        )
        return {i: r for i, (r, _) in replies.items()}

    def _per_env_options(self, options, eid):
        defaults = {k: v for k, v in options.items() if k != "by_env"}
        by_env = options.get("by_env", {})
        return {**defaults, **by_env.get(str(eid), by_env.get(eid, {}))}

    def _layout(self, eid, seed, options):
        return sample_objects(self.cfg, eid, seed, options.get("objects", {}))

    def _options(self, index, ids, seed, options, seeds=None):
        defaults = {k: v for k, v in options.items() if k != "by_env"}
        if index == 0 and not options.get("by_env"):
            return defaults
        indexed = {}
        for eid in ids:
            values = self._per_env_options(options, eid)
            if index:
                effective_seed = seeds[eid] if seeds is not None else seed
                values["objects"] = self._layout(eid, effective_seed, values)
            indexed[str(eid % self.size)] = values
        return {**defaults, "by_env": indexed}

    def _rows(self, replies, field, order, *, seed=None, options=None, seeds=None, subset=False):
        result = {}
        for index, reply in replies.items():
            for row in reply[field]:
                local = row["env_id"]
                if type(local) is not int or not 0 <= local < self.size:
                    raise ValueError("Invalid local environment in simulation response")
                eid = index * self.size + local
                if eid in result:
                    raise ValueError("Duplicate simulation response")
                mapped = {**row, "env_id": eid}
                if seed is not None:
                    info = row["info"]
                    actual_seed = seeds[eid] if seeds is not None else seed
                    expected = self._layout(eid, actual_seed, self._per_env_options(options, eid))
                    if (
                        info["env_id"] != local
                        or info["seed"] != actual_seed
                        or info["objects"] != expected
                    ):
                        raise RuntimeError(
                            f"Worker {index} returned a different layout for global env {eid}"
                        )
                    mapped["info"] = {**info, "env_id": eid}
                    self.verified_layouts[index] += 1
                result[eid] = mapped
        if (subset and not set(result).issubset(order)) or (
            not subset and set(result) != set(order)
        ):
            raise ValueError("Simulation returned unexpected global environment IDs")
        return [result[eid] for eid in order if eid in result]

    def _reset(self, p):
        ids, seed, options = p["env_ids"], p["seed"], p.get("options", {})
        seeds = {int(k): v for k, v in p["env_seeds"].items()} if "env_seeds" in p else None
        calls = {}
        for i, part in enumerate(self._parts(ids, lambda e: e)):
            if part:
                item = {
                    **p,
                    "env_ids": [e % self.size for e in part],
                    "options": self._options(i, part, seed, options, seeds),
                }
                if seeds is not None:
                    item["env_seeds"] = {str(e % self.size): seeds[e] for e in part}
                calls[i] = item
        replies = self._dispatch("/reset", calls)
        return self._rows(replies, "items", ids, seed=seed, options=options, seeds=seeds)

    def _step(self, p):
        order = [x["env_id"] for x in p["items"]]
        limit, seed, opts = p["max_active_envs"], p["reset_seed"], p["reset_options"]
        if not 0 <= limit <= self.total:
            raise ValueError("Invalid global active limit")
        calls = {}
        for i, part in enumerate(self._parts(p["items"], lambda item: item["env_id"])):
            if part:
                ids = [x["env_id"] for x in part]
                calls[i] = {
                    **p,
                    "items": [{**x, "env_id": x["env_id"] % self.size} for x in part],
                    "reset_options": self._options(i, ids, seed, opts),
                    "max_active_envs": self.size if limit == self.total else 0,
                }
        replies = self._dispatch("/step_and_reset", calls)
        steps = self._rows(replies, "items", order)
        done = [row["env_id"] for row in steps if row["terminated"] or row["truncated"]]
        chosen = done[: max(0, limit - (len(order) - len(done)))]
        resets = self._rows(replies, "reset_items", done, seed=seed, options=opts, subset=True)
        if 0 < limit < self.total and chosen:
            # Close to the global transition limit both workers first report endings.
            # Reset exactly the globally selected rows without advancing physics.
            resets = self._reset(
                {
                    "run_id": p["run_id"],
                    "request_id": p["request_id"] + "-reset",
                    "env_ids": chosen,
                    "seed": seed,
                    "options": opts,
                }
            )
        if [row["env_id"] for row in resets] != chosen:
            raise RuntimeError("Workers reset a different set of environments")
        return {"items": steps, "reset_items": resets}

    def call(self, route, payload=None):
        self.last_call_timing = {}
        if self.faulted and route != "/close_run":
            raise RuntimeError("Simulation pool is faulted")
        if route == "/health":
            return deepcopy(self.meta)
        if route == "/close_run":
            replies = self._dispatch(route, {i: payload for i in range(len(self.clients))})
            return {
                "closed": all(r.get("closed") is True for r in replies.values()),
                "restart_required": any(r.get("restart_required") for r in replies.values()),
            }
        if route == "/reset":
            return {"items": self._reset(payload)}
        if route == "/step_and_reset":
            return self._step(payload)
        raise ValueError("Unsupported operation for this fused FR3 training run: " + route)
