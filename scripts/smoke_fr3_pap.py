#!/usr/bin/env python3
"""Check two FR3 environments, action steps and zero-tick isolated object reset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def check_render_poses(backend, eid):
    """Compare the reset USD poses used for RGB to independent robot FK and PhysX."""
    from pxr import Usd, UsdGeom

    engine = backend.engine
    def world_pose(path):
        prim = engine.stage.GetPrimAtPath(engine.relocate(eid, path))
        return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

    state = engine.robot.states()[eid]
    scene = backend.task.config['scene']
    wrist = scene['cameras']['wrist']
    root = world_pose(engine.template.robot_path)
    expected = root @ engine.robot.kin.fk(state[:7], link=wrist['parent_link']) @ wrist['link_from_camera_usd']
    actual = world_pose(engine.template.camera_paths['wrist'])
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    bowl = world_pose(backend.task.scene['bowl_path'])[:3, 3]
    physical = backend.snapshot([eid])[str(eid)]['objects']['bowl']['position_world']
    np.testing.assert_allclose(bowl, physical, atol=2e-6, rtol=0)
    return {'wrist_translation_error_m': float(np.linalg.norm(actual[:3,3]-expected[:3,3])),
            'bowl_translation_error_m': float(np.linalg.norm(bowl-physical))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--scene-config", default="configs/fr3_pap.json")
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from pressb.tasks.fr3_pap.backend import create_backend
    report = {"passed": False, "simulation_only": True}
    backend = None
    try:
        backend = create_backend(dict(scene_config=args.scene_config, gpu=args.gpu, num_envs=2,
            horizon=2, physics_backend="gpu", cpu_threads=4, png_workers=2,
            render_resolution=[640,480], log_dir=str(output/"isaac_logs"),
            terminate_on_success=False, randomization={"enabled":False,"bowl_enabled":False}))
        rows = backend.reset([0,1], 20261116, {})
        # The Isaac 6 adapter explicitly publishes USD poses. Isaac 5 can
        # render correct poses from the fast cache while USD is still stale.
        check_usd = 'reset_pose_publication' in backend.describe()['execution']
        if check_usd:
            report['initial_render_alignment'] = [check_render_poses(backend, i) for i in (0, 1)]
        actions = [dict(env_id=row["env_id"], actions=[row["observation"]["state"]]*2) for row in rows]
        stepped = backend.step(actions)
        before = backend.snapshot([1])
        clock = backend.engine.world.current_time
        new_state = list(rows[0]['observation']['state'])
        new_state[0] += .2
        backend.reset([0], 20261117, {"initial_state": new_state, "objects": {
            "cube":{"center_xy":[.52,.14],"yaw_deg":20.},
            "bowl":{"center_xy":[.44,-.14]}}})
        after = backend.snapshot([1])
        moved = backend.snapshot([0])["0"]
        if check_usd:
            report['indexed_render_alignment'] = [check_render_poses(backend, i) for i in (0, 1)]
        report.update(metadata=backend.describe(), peer_unchanged=before == after,
            reset_world_time_unchanged=clock == backend.engine.world.current_time,
            step_env_ids=[row["env_id"] for row in stepped], selected_state=moved,
            peer_before=before, peer_after=after)
        report["passed"] = bool(report["peer_unchanged"] and report["reset_world_time_unchanged"]
            and report["step_env_ids"] == [0,1]
            and abs(moved["objects"]["cube"]["position_world"][0]-.52)<1e-6
            and abs(moved["objects"]["bowl"]["position_world"][1]+.14)<1e-6
            and report["metadata"]["execution"]["camera_projection_groups"] == 2)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        # Isaac fast_shutdown can exit the interpreter during close.
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"passed":report["passed"],"report":str(output/"report.json"),
                          "error":report.get("error")}),flush=True)
        if backend is not None:
            backend.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
