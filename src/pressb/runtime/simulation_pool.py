"""Synchronous indexed reset/step over dedicated simulation services."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from pathlib import Path
import threading
import uuid

from .contract import fingerprint, validate_metadata
from .simulation import SimulationService


def _worker_contract(metadata):
    result = deepcopy(metadata)
    for key in ("protocol_version", "ready", "num_envs"):
        result.pop(key, None)
    for key in ("gpu_ids", "worker_logs", "camera_count"):
        result.get("execution", {}).pop(key, None)
    return result


class _SimulationShards:
    def __init__(self, clients):
        if len(clients) < 2:
            raise ValueError("Simulation pool requires at least two dedicated workers")
        self.clients = list(clients)
        self.executor = ThreadPoolExecutor(max_workers=len(clients), thread_name_prefix="simulation-shard")
        self.worker_run = uuid.uuid4().hex
        self.episodes = {}
        try:
            health = self._parallel("/health", {i: {} for i in range(len(clients))})
            statuses = self._parallel("/status", {i: {} for i in range(len(clients))})
            self.offsets, self.placement, topology = [], [], []
            common = _worker_contract(health[0])
            for i, row in health.items():
                validate_metadata(row, "simulation")
                if (row.get("ready") is False or statuses[i].get("run_id") is not None
                        or statuses[i].get("fault") or statuses[i].get("environments")):
                    raise ValueError(f"Simulation shard {i} is faulted or already owned")
                if _worker_contract(row) != common:
                    raise ValueError(f"Simulation shard {i} task or execution metadata differs")
                start, count = len(self.placement), row["num_envs"]
                self.offsets.append(start)
                self.placement.extend((i, local) for local in range(count))
                execution = row.get("execution", {})
                topology.append(dict(shard=i, first_env_id=start, num_envs=count,
                    gpu_ids=execution.get("gpu_ids", []), worker_logs=execution.get("worker_logs")))
            metadata = common
            metadata["num_envs"] = len(self.placement)
            metadata["capabilities"] = [cap for cap in metadata.get("capabilities", [])
                                        if cap in ("per_env_reset_seeds", "terminal_state_only")]
            mode = "explicit env_seeds unchanged; shared seed unchanged on shard 0, BLAKE2b-derived on later shards"
            identity = dict(worker_task_config_sha256=metadata.get("task_config_sha256"),
                            shard_sizes=[row["num_envs"] for row in topology], reset_seed_mapping=mode)
            metadata["task_config_sha256"] = fingerprint(identity)
            implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            worker_revision = metadata.get("source_revision")
            metadata["source_revision"] = "sha256:" + fingerprint([worker_revision, implementation])
            metadata["simulation_pool"] = dict(shards=topology, **identity,
                worker_source_revision=worker_revision, implementation_sha256=implementation)
            execution = metadata.setdefault("execution", {})
            execution["workers"] = len(clients)
            execution["gpu_ids"] = [gpu for shard in topology for gpu in shard["gpu_ids"]]
            if all("camera_count" in row.get("execution", {}) for row in health.values()):
                execution["camera_count"] = sum(row["execution"]["camera_count"] for row in health.values())
            self.metadata = metadata
        except BaseException:
            self.close()
            raise

    def describe(self):
        return self.metadata

    def status(self):
        return {"worker_run_id": self.worker_run, "worker_episodes": deepcopy(self.episodes)}

    def _parallel(self, route, payloads):
        futures = {i: self.executor.submit(self.clients[i].call, route, payload) for i, payload in payloads.items()}
        results, errors = {}, []
        for i, future in futures.items():
            try:
                results[i] = future.result()
                if not isinstance(results[i], dict):
                    raise TypeError("response must be an object")
            except Exception as error:
                errors.append(f"shard {i}: {type(error).__name__}: {error}")
        if errors:
            raise RuntimeError(f"Simulation {route} failed; " + ", ".join(errors))
        return results

    def _request(self):
        return {"run_id": self.worker_run, "request_id": uuid.uuid4().hex}

    def _join(self, route, payloads, ids, seeds=None):
        responses = self._parallel(route, payloads)
        rows = {}
        for shard, response in responses.items():
            local_ids = (payloads[shard]["env_ids"] if route == "/reset" else
                         [item["env_id"] for item in payloads[shard]["items"]])
            items = response.get("items")
            SimulationService._rows(items, local_ids)
            for row in items:
                local = row["env_id"]
                eid = self.offsets[shard] + local
                episode, step = row.get("episode_id"), row.get("step_id")
                if not isinstance(episode, str) or not episode or type(step) is not int:
                    raise RuntimeError(f"Simulation shard {shard} returned invalid episode identity")
                if route == "/reset":
                    if step != 0:
                        raise RuntimeError("Worker reset did not return step zero")
                else:
                    prior = self.episodes[eid]
                    if episode != prior["episode_id"] or step != prior["step_id"] + 1:
                        raise RuntimeError("Worker returned a stale or inconsistent episode/step")
                self.episodes[eid] = {"episode_id": episode, "step_id": step}
                mapping = dict(shard=shard, local_env_id=local, global_env_id=eid, worker_episode_id=episode)
                if seeds is not None:
                    mapping["requested_seed"], mapping["worker_seed"] = seeds[eid]
                rows[eid] = {**row, "env_id": eid, "info": {**row.get("info", {}), "simulation_shard": mapping}}
        return [rows[eid] for eid in ids]

    def reset(self, env_ids, seed, options, *, env_seeds=None):
        options = dict(options)
        by_env = options.pop("by_env", {})
        requests, seeds = {}, {}
        for eid in env_ids:
            shard, local = self.placement[eid]
            if shard not in requests:
                # PAP draws a layout with SeedSequence([seed, local_env_id]).
                # Separate shared seeds to avoid duplicating the first world's layouts.
                shared = seed
                if shard and env_seeds is None:
                    raw = hashlib.blake2b(f"{seed}:{shard}".encode(), digest_size=8).digest()
                    shared = int.from_bytes(raw, "little") & ((1 << 63) - 1)
                requests[shard] = {**self._request(), "env_ids": [], "seed": shared, "options": dict(options)}
            request = requests[shard]
            request["env_ids"].append(local)
            if str(eid) in by_env:
                request["options"].setdefault("by_env", {})[str(local)] = by_env[str(eid)]
            if env_seeds is not None:
                request.setdefault("env_seeds", {})[str(local)] = env_seeds[eid]
            seeds[eid] = (seed, request["seed"]) if env_seeds is None else (env_seeds[eid], env_seeds[eid])
        return self._join("/reset", requests, env_ids, seeds)

    def step(self, commands, terminal_observation_mode="full"):
        requests = {}
        ids = []
        for command in commands:
            eid = command["env_id"]
            ids.append(eid)
            shard, local = self.placement[eid]
            if shard not in requests:
                requests[shard] = {**self._request(), "items": [], "terminal_observation_mode": terminal_observation_mode}
            requests[shard]["items"].append({**command, **self.episodes[eid], "env_id": local})
        return self._join("/step", requests, ids)

    def close_run(self):
        owned = {self.placement[eid][0] for eid in self.episodes}
        results = self._parallel("/close_run", {i: {"run_id": self.worker_run} for i in owned})
        for row in results.values():
            if row.get("closed") is not True:
                raise RuntimeError("Worker did not acknowledge closing its run")
            if row.get("restart_required") and not self.metadata.get("requires_restart_after_active_close"):
                raise RuntimeError("Worker requires an unexpected simulator restart")
        self.episodes.clear()
        self.worker_run = uuid.uuid4().hex

    def close(self):
        self.executor.shutdown(wait=True)
        for client in self.clients:
            client.close()


class SimulationPool(SimulationService):
    """One ownership/idempotency boundary over immutable contiguous shard ranges."""

    def __init__(self, clients, *, request_cache_size=2):
        if type(request_cache_size) is not int or request_cache_size < 1:
            raise ValueError("request_cache_size must be a positive integer")
        self.lock = threading.RLock()
        super().__init__(_SimulationShards(clients), request_cache_size=request_cache_size)

    @property
    def handlers(self):
        return {"/health": self.health, "/status": self.status, "/reset": self.reset,
                "/step": self.step, "/close_run": self.close_run}

    def health(self, payload=None):
        with self.lock:
            return {**super().health(payload), "ready": not bool(self.fault)}

    def status(self, payload=None):
        with self.lock:
            return super().status(payload)

    def reset(self, payload):
        with self.lock:
            options = payload.get("options") if isinstance(payload, dict) else None
            if isinstance(options, dict) and "by_env" in options:
                indexed = options["by_env"]
                if not isinstance(indexed, dict):
                    raise ValueError("options.by_env must be a mapping")
                canonical = {}
                for key, value in indexed.items():
                    s = str(key)
                    if (not s.isascii() or not s.isdecimal() or s != str(int(s)) or not 0 <= int(s) < self.metadata["num_envs"]
                            or s in canonical or not isinstance(value, dict)):
                        raise ValueError("Invalid or duplicate global environment in options.by_env")
                    canonical[s] = value
                payload = {**payload, "options": {**options, "by_env": canonical}}
            return super().reset(payload)

    def step(self, payload):
        with self.lock:
            return super().step(payload)

    def close_run(self, payload):
        with self.lock:
            return super().close_run(payload)

    def close(self):
        with self.lock:
            try:
                if self.run_id is not None and not self.fault:
                    self.close_run({"run_id": self.run_id})
            finally:
                self.fault = self.fault or "Simulation pool closed"
                super().close()
