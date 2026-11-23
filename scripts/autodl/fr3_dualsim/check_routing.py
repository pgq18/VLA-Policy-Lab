import json, sys
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from paired_fr3 import PairedFR3Client
from pressb.tasks.fr3_pap.configuration import load_config, sample_objects

raw = json.loads(Path(sys.argv[1]).read_text())["backend"]["config"]
raw["scene_config"] = sys.argv[2]
cfg = load_config(raw)
health = json.loads(Path(sys.argv[3]).read_text())
health["num_envs"] = 64
health["execution"]["camera_count"] = 128
barrier = Barrier(2, timeout=5)


class Worker:
    def __init__(self, index):
        self.index = index
        self.fail = False
        self.done = set()

    def close(self):
        pass

    def reset(self, eid, seed, opts):
        custom = opts.get("by_env", {}).get(str(eid), {})
        args = {**opts, **custom}
        return {
            "env_id": eid,
            "episode_id": uuid4().hex,
            "step_id": 0,
            "info": {
                "env_id": eid,
                "seed": seed,
                "objects": sample_objects(cfg, eid, seed, args.get("objects", {})),
            },
        }

    def call(self, route, p=None):
        if route == "/health":
            v = deepcopy(health)
            v["execution"]["gpu_ids"] = [self.index]
            return v
        if route == "/status":
            return {"run_id": None, "fault": None, "environments": {}}
        if route == "/close_run":
            return {"closed": True}
        if route == "/reset":
            return {
                "items": [
                    self.reset(i, p.get("env_seeds", {}).get(str(i), p["seed"]), p["options"])
                    for i in p["env_ids"]
                ]
            }
        barrier.wait()
        if self.fail:
            raise RuntimeError("worker failed")
        steps = [
            dict(
                env_id=i["env_id"],
                episode_id=i["episode_id"],
                step_id=i["step_id"] + 1,
                terminated=i["env_id"] in self.done,
                truncated=False,
            )
            for i in p["items"]
        ]
        done = [r["env_id"] for r in steps if r["terminated"]]
        reset = done[: max(0, p["max_active_envs"] - len(steps) + len(done))]
        return {
            "items": steps,
            "reset_items": [self.reset(i, p["reset_seed"], p["reset_options"]) for i in reset],
        }


workers = [Worker(0), Worker(3)]
pool = PairedFR3Client(workers, raw)
try:
    envs = list(range(128))
    seed = 38421
    options = {}
    env_seeds = {str(i): seed + i for i in envs}
    p = dict(
        env_ids=envs, seed=seed, run_id="test", request_id="a", options=options, env_seeds=env_seeds
    )
    rows = pool.call("/reset", p)["items"]
    assert [r["info"]["objects"] for r in rows] == [sample_objects(cfg, i, seed + i) for i in envs]
    assert all(rows[i]["info"]["objects"] != rows[64 + i]["info"]["objects"] for i in range(64))
    inputs = [
        dict(env_id=i, episode_id=r["episode_id"], step_id=r["step_id"], actions=[])
        for i, r in enumerate(rows)
    ]
    workers[0].done = {1, 63}
    workers[1].done = {0, 36}
    q = dict(run_id="test", request_id="step", items=inputs, reset_seed=11, reset_options=options)
    for limit, expected in [(128, [1, 63, 64, 100]), (126, [1, 63]), (0, [])]:
        result = pool.call("/step_and_reset", {**q, "max_active_envs": limit})
        assert [x["env_id"] for x in result["items"]] == envs and [
            x["episode_id"] for x in result["items"]
        ] == [r["episode_id"] for r in rows]
        assert [x["env_id"] for x in result["reset_items"]] == expected
        assert all(
            x["info"]["objects"] == sample_objects(cfg, x["env_id"], 11)
            for x in result["reset_items"]
        )
    workers[1].fail = True
    try:
        pool.call("/step_and_reset", {**q, "max_active_envs": 128})
        raise AssertionError("missing worker error")
    except RuntimeError:
        assert pool.faulted
    assert pool.call("/close_run", {"run_id": "test"})["closed"]
    print(
        "128 global layouts and IDs match; concurrent dispatch, fused resets, capped and final resets and worker failure verified."
    )
finally:
    pool.close()
