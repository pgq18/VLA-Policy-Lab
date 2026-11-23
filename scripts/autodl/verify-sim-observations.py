"""Run the deployed task recipe with two clones and save both real camera views."""
import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parent
PROJECT = DEPLOY.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from pressb.control.pose_protocol import pose8_to_pose9
from pressb.experiments.config import load_experiment
from pressb.runtime.contract import decode_image
from pressb.runtime.server import load_backend

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("task", choices=("pap_plus","pap_plus_plus","fr3_pap","elevator"))
parser.add_argument("gpu", type=int, choices=range(4))
parser.add_argument("output", type=Path)
parser.add_argument("--single-render-tick",action="store_true",help="Use Kit's single tick-rate RTX camera scheduler")
parser.add_argument("--inspect-render-settings",action="store_true",help="Log actual/requested Kit setting values if its renderer assertion fails")
parser.add_argument("--renderer-profile",choices=("rl_fast","calibrated"),help="Select a PressB renderer profile for the PAP task")
args = parser.parse_args()
sys.argv = [sys.argv[0]]
if args.single_render_tick:
    sys.argv.append("--/rtx/hydra/supportMultiTickRate=false")
os.chdir(PROJECT)
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
spec = load_experiment(DEPLOY/"configs"/f"{args.task}.json")["nodes"]["simulation"]["backend"]
cfg = spec["config"]
cfg["num_envs"] = 2
if args.task.startswith("pap_plus"):
    if args.renderer_profile:
        cfg["renderer_profile"] = args.renderer_profile
    cfg["python"] = sys.executable
    cfg["gpu_ids"] = [args.gpu]
else:
    cfg["gpu"] = args.gpu
cfg["output" if args.task == "elevator" else "log_dir"] = str(out/"worker")
report = {"task":args.task,"gpu":args.gpu,"interpreter":sys.executable,"renderer_profile":cfg.get("renderer_profile"),"stages":{},"status":"starting"}
backend = None
if args.inspect_render_settings:
    from pressb.runtime.isaac_parallel import IsaacParallelWorld
    capture = IsaacParallelWorld._capture
    def traced_capture(engine, *args, **kwargs):
        try:
            return capture(engine, *args, **kwargs)
        except RuntimeError:
            diff = {k:{"requested":v,"actual":engine.renderer_settings.get(k)}
                    for k,v in engine.requested_renderer_settings.items() if engine.renderer_settings.get(k) != v}
            (out/"render-settings.json").write_text(json.dumps(diff,indent=2)+"\n")
            raise
    IsaacParallelWorld._capture = traced_capture

def record():
    tmp = out/"report.tmp"
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    tmp.replace(out/"report.json")

def images(rows, phase, meta):
    measurements = []
    for row in rows:
        frames = row["observation"]["images"]
        names = meta["observation"]["cameras"]
        assert set(frames) == set(names), (frames.keys(),names)
        digests = set()
        for name, data in frames.items():
            frame = decode_image(data)
            expected_size = tuple(cfg.get("render_resolution", [cfg.get("camera_resolution")] * 2))
            assert frame.size == expected_size, (frame.size, expected_size)
            pixels = np.asarray(frame)
            # Catch uniform, almost black, or identical camera streams.
            assert pixels.ndim == 3 and pixels.shape[2] == 3 and pixels.std() > 1
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            digests.add(digest)
            target = out/f"{phase}-{row['env_id']}-{name}.png"
            frame.save(target)
            measurements.append(dict(env_id=row["env_id"],camera=name,size=list(frame.size),pixel_sha256=digest,
                                     pixel_std=float(pixels.std()),file=target.name))
        assert len(digests) == len(names)
    return measurements

try:
    record()
    backend = load_backend(spec, kind="task")
    meta = backend.describe()
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    assert meta["task_id"] == args.task and meta["num_envs"] == 2
    proof=meta["execution"]
    assert proof["physics_backend_verified"] and proof["gpu_dynamics"]
    spec_action=meta["action"]
    horizon=spec_action["horizon"]
    assert horizon == (7 if args.task=="elevator" else 16),horizon
    report.update(physics_backend=proof["physics_backend"],action_horizon=horizon,action_width=len(spec_action["names"]))
    rows=backend.reset([0,1],42,{})
    report["stages"]["reset_images"]=images(rows,"initial",meta)
    record()
    actions=[]
    for row in rows:
        measured=np.asarray(row["observation"]["state"],dtype=float)
        assert np.isfinite(measured).all()
        chunk=np.repeat(measured[None,:],horizon,axis=0)
        if args.task=="elevator":
            chunk=pose8_to_pose9(chunk)
        if "lower" in spec_action:
            chunk=np.clip(chunk,spec_action["lower"],spec_action["upper"])
        actions.append(dict(env_id=row["env_id"],actions=chunk.tolist()))
    stepped=backend.step(actions)
    report["stages"]["stepped_images"]=images(stepped,"stepped",meta)
    ticks=[r["info"]["physics_steps"] for r in stepped]
    assert ticks == [horizon*backend.execution["physics_ticks_per_action"]]*2,ticks
    report["stages"]["world_ticks_per_env"]=ticks
    before=backend.snapshot([1])
    clock=backend.engine.world.current_time
    backend.reset([0],43,{})
    assert before == backend.snapshot([1]) and clock == backend.engine.world.current_time
    report["stages"]["isolated_indexed_reset"]=True
    report["status"]="passed"
except Exception:
    report["status"]="failed"
    report["error"]=traceback.format_exc()
    raise
finally:
    record()
    if backend is not None:
        backend.close()
