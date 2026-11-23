"""Concurrent inference replicas behind the ordinary schema-v2 node protocol.

Each worker owns its accelerator and model. The pool owns only RPC clients and
context routing. A request is split into contiguous blocks, evaluated in
parallel, and restored to input order. Weight updates form a barrier: every
worker must acknowledge the same new version and parameter digest before any
further inference is accepted. An uncertain/partial commit locks the pool;
restarting synchronized workers is required rather than serving mixed policies.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import base64
import math
import threading
import time
import uuid

from .contract import finite_array, validate_metadata, validate_observation
from .inference import InferenceService


class InferencePoolError(RuntimeError):
    pass


def _identity(health):
    result = deepcopy(health)
    for name in ("protocol_version", "ready", "cache", "microbatch_size",
                 "encoded_count", "decoded_count", "last_timing"):
        result.pop(name, None)
    if isinstance(result.get("batching"), dict):
        result["batching"].pop("microbatch_size", None)
    if isinstance(result.get("action_expert"), dict):
        result["action_expert"].pop("policy_version", None)
        result["action_expert"].pop("weights_sha256", None)
    return result


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _http_images(payload):
    """Retain the PNG bytes when a binary public request reaches an HTTP worker."""
    if "observations" not in payload:
        return payload
    observations = []
    for row in payload["observations"]:
        images = row.get("images", {})
        changed = {name: {**value, "encoding": "png_base64",
                          "data": base64.b64encode(value["data"]).decode("ascii")}
                   for name, value in images.items() if value.get("encoding") == "png_bytes"}
        observations.append({**row, "images": {**images, **changed}} if changed else row)
    return {**payload, "observations": observations}


class InferencePool:
    """Aggregate independent, identically configured inference services.

    Workers must be dedicated to this pool; external clients must not mutate
    their weights or context caches. ``close`` closes the pool's connections,
    not the externally managed worker processes. TTLs are conservatively
    measured from dispatch, never extended by decode or delayed responses.
    """

    def __init__(self, clients, *, max_batch_size=None, clock=time.monotonic):
        if not isinstance(clients, (tuple, list)) or len(clients) < 2:
            raise ValueError("Inference pool requires at least two worker clients")
        self.clients, self.clock = list(clients), clock
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=len(clients), thread_name_prefix="inference-replica")
        self._closed = False
        self._failed = None
        self._pending = None
        self.contexts = OrderedDict()
        self.generation = uuid.uuid4().hex
        try:
            health = self._parallel({i: ("/health", {}) for i in range(len(clients))})
            first = health[0]
            validate_metadata(first, node_type="inference")
            self._identity = _identity(first)
            self.metadata = deepcopy(first)
            self.metadata.pop("protocol_version", None)
            self.has_expert = isinstance(first.get("action_expert"), dict)
            self.capacities, self.ttls = [], []
            for index in range(len(clients)):
                row = health[index]
                validate_metadata(row, node_type="inference")
                if row.get("ready") is not True or _identity(row) != self._identity:
                    raise ValueError("Inference workers must be ready and share one model/sampler identity")
                cache = row.get("cache", {})
                capacity, ttl = cache.get("capacity"), cache.get("ttl_seconds")
                if (type(capacity) is not int or capacity < 1 or type(ttl) not in (int, float)
                        or not math.isfinite(ttl) or ttl <= 0):
                    raise ValueError("Workers must advertise bounded context capacity and TTL")
                self.capacities.append(capacity)
                self.ttls.append(ttl)
            self.cache_size = sum(self.capacities)
            limit = min(self.capacities) * len(clients)
            if max_batch_size is None:
                max_batch_size = limit
            if type(max_batch_size) is not int or not 1 <= max_batch_size <= limit:
                raise ValueError("max_batch_size exceeds balanced worker capacity")
            self.max_batch_size = max_batch_size
            self.version = first.get("action_expert", {}).get("policy_version")
            self.weights_sha256 = first.get("action_expert", {}).get("weights_sha256")
            if self.has_expert and (type(self.version) is not int or self.version < 0
                                    or not _digest(self.weights_sha256)):
                raise ValueError("Workers must advertise a valid expert version and digest")
            self._validate_health(health)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _fields(payload, allowed, required=()):
        InferenceService._fields(payload, set(allowed), set(required))

    def _ensure_open(self, *, inference=False):
        if self._closed:
            raise InferencePoolError("Inference pool is closed")
        if self._failed is not None:
            raise InferencePoolError("Inference pool is locked after a worker failure: " + self._failed)
        if inference and self._pending is not None:
            raise ValueError("Weight publication is in progress; inference waits for all replicas")

    def _fail(self, message):
        self._failed = str(message)
        raise InferencePoolError("Inference pool locked; restart synchronized workers: " + self._failed)

    def _call(self, index, path, payload):
        client = self.clients[index]
        if str(getattr(client, "endpoint", "")).startswith(("http://", "https://")):
            payload = _http_images(payload)
        return client.call(path, payload)

    def _parallel(self, calls):
        # Wait for every submitted operation, including on failure. A commit
        # must not remain in a background thread after the pool reports failure.
        futures = {i: self._executor.submit(self._call, i, path, payload)
                   for i, (path, payload) in calls.items()}
        results, errors = {}, []
        for index, future in futures.items():
            try:
                result = future.result()
                if not isinstance(result, dict):
                    raise TypeError("Worker returned a non-object")
                results[index] = result
            except Exception as error:
                errors.append(f"worker {index}: {type(error).__name__}: {error}")
        if errors:
            self._fail("; ".join(errors))
        return results

    def _validate_health(self, rows, *, version=None, weights=None):
        version = self.version if version is None else version
        weights = self.weights_sha256 if weights is None else weights
        for index, row in rows.items():
            if row.get("ready") is not True or _identity(row) != self._identity:
                self._fail(f"worker {index} model/sampler identity changed")
            if self.has_expert:
                expert = row.get("action_expert", {})
                if expert.get("policy_version") != version or expert.get("weights_sha256") != weights:
                    self._fail(f"worker {index} has a different action expert version or digest")
        return rows

    def _check_workers(self):
        return self._validate_health(self._parallel({i: ("/health", {}) for i in range(len(self.clients))}))

    def _purge(self):
        now = self.clock()
        for key in [key for key, row in self.contexts.items() if row[2] <= now]:
            del self.contexts[key]

    def _parts(self, count):
        size, remainder = divmod(count, len(self.clients))
        start = 0
        result = []
        for index in range(len(self.clients)):
            end = start + size + (index < remainder)
            if end > start:
                result.append((index, start, end))
            start = end
        return result

    def _observations(self, payload, *, path):
        allowed = {"observations", "seeds"}
        required = set(allowed)
        if path == "/sample_chain":
            allowed |= {"expected_policy_version", "deterministic"}
            required.add("expected_policy_version")
        elif path == "/infer":
            allowed.add("initial_noises")
        else:
            allowed |= {"include_base_actions", "release_context_ids"}
        self._fields(payload, allowed, required)
        observations, seeds = payload["observations"], payload["seeds"]
        if not isinstance(observations, list) or not 1 <= len(observations) <= self.max_batch_size:
            raise ValueError("observations must contain 1..max_batch_size entries")
        if not isinstance(seeds, list) or len(seeds) != len(observations) or any(
                type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("seeds must contain one valid integer per observation")
        for row in observations:
            validate_observation(row, self.metadata["observation"])
        if path == "/sample_chain":
            if not self.has_expert:
                raise ValueError("Workers do not support action expert optimization")
            if type(payload["expected_policy_version"]) is not int or payload["expected_policy_version"] != self.version:
                raise ValueError("Stale policy version for inference pool")
            if type(payload.get("deterministic", False)) is not bool:
                raise ValueError("deterministic must be boolean")
        elif path == "/infer" and "initial_noises" in payload:
            values = payload["initial_noises"]
            if not isinstance(values, list) or len(values) != len(observations):
                raise ValueError("initial_noises must match observations")
            for value in values:
                if value is not None:
                    finite_array(value, tuple(self.metadata["noise"]["shape"]), "initial_noise")
        elif path == "/encode" and type(payload.get("include_base_actions", False)) is not bool:
            raise ValueError("include_base_actions must be boolean")
        return observations, seeds

    def _keys(self, keys, *, field):
        if (not isinstance(keys, list) or len(keys) > self.cache_size
                or any(not isinstance(key, str) or not key for key in keys)):
            raise ValueError(f"{field} must be a bounded list of context identifiers")
        return list(dict.fromkeys(keys))

    def health(self, payload=None):
        with self._lock:
            self._ensure_open()
            self._fields(payload or {}, set())
            health = self._check_workers()
            self._purge()
            result = deepcopy(self.metadata)
            result.update(ready=self._pending is None,
                cache={"size": len(self.contexts), "capacity": self.cache_size,
                       "ttl_seconds": min(self.ttls), "eviction": "per_worker_oldest_created",
                       "ttl_refresh_on_decode": False},
                encoded_count=sum(row.get("encoded_count", 0) for row in health.values()),
                decoded_count=sum(row.get("decoded_count", 0) for row in health.values()),
                parallel_inference={"replicas": len(self.clients), "sharding": "contiguous_balanced",
                    "max_request_observations": self.max_batch_size,
                    "weight_barrier": "all_replicas_same_version_and_digest",
                    "publication_in_progress": self._pending is not None})
            if self.has_expert:
                result["action_expert"].update(policy_version=self.version, weights_sha256=self.weights_sha256)
            return result

    def _stateless(self, path, payload):
        with self._lock:
            started = self.clock()
            self._ensure_open(inference=True)
            observations, seeds = self._observations(payload, path=path)
            self._check_workers()
            calls = {}
            for index, start, end in self._parts(len(observations)):
                part = {key: value for key, value in payload.items() if key != "protocol_version"}
                part.update(observations=observations[start:end], seeds=seeds[start:end])
                if "initial_noises" in part:
                    part["initial_noises"] = part["initial_noises"][start:end]
                calls[index] = (path, part)
            responses = self._parallel(calls)
            items = []
            for index, start, end in self._parts(len(observations)):
                result = responses[index]
                if not isinstance(result.get("items"), list) or len(result["items"]) != end - start:
                    self._fail(f"worker {index} returned the wrong number of inference rows")
                if path == "/sample_chain":
                    if (result.get("policy_version") != self.version or result.get("weights_sha256") != self.weights_sha256
                            or any(row.get("policy_version") != self.version for row in result["items"])):
                        self._fail("A sampled chain used a different expert version")
                items.extend(result["items"])
            self._check_workers()
            result = {"items": items, "timing": {"total_seconds": self.clock() - started},
                      "replicas_used": len(calls),
                      "replica_timings": [{"worker": index, "items": end - start,
                          "seconds": responses[index].get("timing", {}).get("total_seconds")}
                          for index, start, end in self._parts(len(observations))]}
            if path == "/sample_chain":
                result.update(policy_version=self.version, weights_sha256=self.weights_sha256)
            return result

    def infer(self, payload):
        return self._stateless("/infer", payload)

    def sample_chain(self, payload):
        return self._stateless("/sample_chain", payload)

    def encode(self, payload):
        with self._lock:
            started = self.clock()
            self._ensure_open(inference=True)
            observations, seeds = self._observations(payload, path="/encode")
            release = self._keys(payload.get("release_context_ids", []), field="release_context_ids")
            self._purge()
            self._check_workers()
            releases = {}
            for key in release:
                if key in self.contexts:
                    index, inner, _ = self.contexts[key]
                    releases.setdefault(index, []).append(inner)
            calls = {index: ("/release", {"context_ids": keys}) for index, keys in releases.items()}
            parts = self._parts(len(observations))
            for index, start, end in parts:
                part = {"observations": observations[start:end], "seeds": seeds[start:end]}
                if "include_base_actions" in payload:
                    part["include_base_actions"] = payload["include_base_actions"]
                if index in releases:
                    part["release_context_ids"] = releases[index]
                calls[index] = ("/encode", part)
            dispatched = self.clock()
            responses = self._parallel(calls)
            self._check_workers()
            for key in release:
                self.contexts.pop(key, None)
            items = []
            for index, start, end in parts:
                values = responses[index].get("items")
                if not isinstance(values, list) or len(values) != end - start:
                    self._fail(f"worker {index} returned the wrong number of encoded rows")
                existing = [key for key, row in self.contexts.items() if row[0] == index]
                for key in existing[:max(0, len(existing) + len(values) - self.capacities[index])]:
                    del self.contexts[key]
                for row in values:
                    inner = row.get("context_id")
                    if not isinstance(inner, str) or not inner:
                        self._fail(f"worker {index} returned an invalid context identifier")
                    key = f"{self.generation}.{index}.{inner}"
                    if key in self.contexts:
                        self._fail("Worker reused a live context identifier")
                    self.contexts[key] = (index, inner, dispatched + self.ttls[index])
                    items.append({**row, "context_id": key})
            result = {"items": items, "timing": {"total_seconds": self.clock() - started}}
            if "release_context_ids" in payload:
                result["released_context_ids"] = release
            return result

    def decode(self, payload):
        with self._lock:
            self._ensure_open(inference=True)
            self._fields(payload, {"items"}, {"items"})
            items = payload["items"]
            if not isinstance(items, list) or not 1 <= len(items) <= self.cache_size:
                raise ValueError("items must contain 1..cache_size entries")
            self._purge()
            calls, positions = {}, {}
            for position, row in enumerate(items):
                InferenceService.decode_item_noise(row, self.metadata)
                key = row["context_id"]
                if not isinstance(key, str) or key not in self.contexts:
                    raise ValueError("context is missing or expired; encode the current observation again")
                index, inner, _ = self.contexts[key]
                calls.setdefault(index, ("/decode", {"items": []}))[1]["items"].append({**row, "context_id": inner})
                positions.setdefault(index, []).append((position, key, inner))
            if any(len(value[1]["items"]) > self.capacities[index] for index, value in calls.items()):
                raise ValueError("decode request exceeds one worker's capacity")
            self._check_workers()
            responses = self._parallel(calls)
            self._check_workers()
            result = [None] * len(items)
            for index, entries in positions.items():
                rows = responses[index].get("items")
                if not isinstance(rows, list) or len(rows) != len(entries):
                    self._fail("Worker returned the wrong number of decoded rows")
                for row, (position, key, inner) in zip(rows, entries, strict=True):
                    if row.get("context_id") != inner:
                        self._fail("Worker returned a different context's decoded action")
                    result[position] = {**row, "context_id": key}
            return {"items": result}

    def release(self, payload):
        with self._lock:
            self._ensure_open(inference=True)
            self._fields(payload, {"context_ids"}, {"context_ids"})
            keys = self._keys(payload["context_ids"], field="context_ids")
            self._purge()
            calls = {}
            for key in keys:
                if key in self.contexts:
                    index, inner, _ = self.contexts[key]
                    calls.setdefault(index, ("/release", {"context_ids": []}))[1]["context_ids"].append(inner)
            responses = self._parallel(calls)
            for key in keys:
                self.contexts.pop(key, None)
            return {"released": sum(row.get("released", 0) for row in responses.values())}

    def weights_begin(self, payload):
        with self._lock:
            self._ensure_open(inference=True)
            self._fields(payload, {"expected_version", "new_version", "size", "sha256", "weights_sha256"},
                         {"expected_version", "new_version", "size", "sha256", "weights_sha256"})
            if not self.has_expert:
                raise ValueError("Workers do not support action expert optimization")
            if type(payload["expected_version"]) is not int or payload["expected_version"] != self.version:
                raise ValueError("Stale policy version for inference pool")
            if type(payload["new_version"]) is not int or payload["new_version"] != self.version + 1:
                raise ValueError("new_version must increment policy_version by exactly one")
            if type(payload["size"]) is not int or not 0 < payload["size"] <= 4 * 1024**3:
                raise ValueError("Weight upload size is invalid")
            if not _digest(payload["sha256"]) or not _digest(payload["weights_sha256"]):
                raise ValueError("Weight digests must be SHA256 values")
            self._check_workers()
            responses = self._parallel({i: ("/weights_begin", payload) for i in range(len(self.clients))})
            for index, row in responses.items():
                if (row.get("offset") != 0 or not isinstance(row.get("upload_id"), str)
                        or not row["upload_id"] or row.get("policy_version") != self.version
                        or row.get("weights_sha256") != self.weights_sha256):
                    self._fail(f"worker {index} did not acknowledge weight staging")
            self._pending = {"upload_id": uuid.uuid4().hex, "workers": {i: row["upload_id"] for i, row in responses.items()},
                "offset": 0, "size": payload["size"], "version": payload["new_version"], "weights": payload["weights_sha256"]}
            return {"upload_id": self._pending["upload_id"], "offset": 0,
                    "policy_version": self.version, "weights_sha256": self.weights_sha256}

    def _upload(self, payload, fields):
        self._ensure_open()
        self._fields(payload, fields, fields)
        if self._pending is None or payload.get("upload_id") != self._pending["upload_id"]:
            raise ValueError("Unknown pool weight upload")
        return self._pending

    def weights_append(self, payload):
        with self._lock:
            pending = self._upload(payload, {"upload_id", "offset", "data"})
            if type(payload["offset"]) is not int or payload["offset"] != pending["offset"]:
                raise ValueError("Weight chunks must arrive once in sequential order")
            chunk = base64.b64decode(payload["data"], validate=True)
            if not 0 < len(chunk) <= 8 * 1024**2 or pending["offset"] + len(chunk) > pending["size"]:
                raise ValueError("Invalid weight chunk size")
            end = pending["offset"] + len(chunk)
            replies = self._parallel({i: ("/weights_append", {**payload, "upload_id": worker})
                                      for i, worker in pending["workers"].items()})
            if any(row.get("offset") != end or row.get("upload_id") != pending["workers"][i]
                   for i, row in replies.items()):
                self._fail("A worker did not acknowledge the complete weight chunk")
            pending["offset"] = end
            return {"upload_id": pending["upload_id"], "offset": end}

    def weights_commit(self, payload):
        with self._lock:
            pending = self._upload(payload, {"upload_id"})
            if pending["offset"] != pending["size"]:
                raise ValueError("Cannot commit an incomplete weight upload")
            replies = self._parallel({i: ("/weights_commit", {"upload_id": worker})
                                      for i, worker in pending["workers"].items()})
            if any(row.get("committed") is not True or row.get("policy_version") != pending["version"]
                   or row.get("weights_sha256") != pending["weights"] for row in replies.values()):
                self._fail("A worker did not commit the requested action expert")
            health = self._parallel({i: ("/health", {}) for i in range(len(self.clients))})
            self._validate_health(health, version=pending["version"], weights=pending["weights"])
            self.version, self.weights_sha256 = pending["version"], pending["weights"]
            self._pending = None
            self.contexts.clear()
            return {"committed": True, "policy_version": self.version,
                    "weights_sha256": self.weights_sha256, "replicas_committed": len(self.clients)}

    def weights_abort(self, payload):
        with self._lock:
            pending = self._upload(payload, {"upload_id"})
            replies = self._parallel({i: ("/weights_abort", {"upload_id": worker})
                                      for i, worker in pending["workers"].items()})
            if any(row.get("aborted") is not True or row.get("policy_version") != self.version
                   or row.get("weights_sha256") != self.weights_sha256 for row in replies.values()):
                self._fail("A worker did not abort the staged update")
            self._pending = None
            return {"aborted": True, "policy_version": self.version, "weights_sha256": self.weights_sha256}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.contexts.clear()
            self._executor.shutdown(wait=True, cancel_futures=True)
            for client in self.clients:
                client.close()

    @property
    def handlers(self):
        result = {"/health": self.health, "/encode": self.encode, "/decode": self.decode,
                  "/release": self.release, "/infer": self.infer}
        if self.has_expert:
            result.update({"/sample_chain": self.sample_chain, "/weights_begin": self.weights_begin,
                           "/weights_append": self.weights_append, "/weights_commit": self.weights_commit,
                           "/weights_abort": self.weights_abort})
        return result
