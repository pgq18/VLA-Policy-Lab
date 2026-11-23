#!/usr/bin/env python3
"""Qualify shared CPU/GPU physics with identical elevator commands.

Run each solver in a fresh Isaac process, then compare the saved trajectories.
No model, learner, remote service or physical robot is used. Results are written
before Kit shutdown, which may terminate the Python process directly.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import uuid
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smoke_elevator_task_backend import DEFAULT_REPLAY, DEFAULT_EPISODE, load_replay, save_images


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Trace:
    """Read-only physics observer: no render and no additional world tick."""
    def __init__(self):
        self.phase = "startup"
        self.rows = []
        self.before_reset = None

    def after_tick(self, backend, ids, outcomes):
        snapshots = backend.snapshot(ids)
        for eid in ids:
            snapshot = snapshots[str(eid)]
            outcome = outcomes[eid]
            info = outcome.get("info", {})
            self.rows.append(dict(phase=self.phase, env_id=eid, tick=int(backend.ticks[eid]),
                world_time=float(backend.engine.world.current_time), state=snapshot["state"],
                joints=snapshot["joints"], joint_velocity=snapshot["joint_velocity"],
                button_positions=snapshot["button_positions"],
                button_velocities=snapshot["button_velocities"],
                max_target_travel=float(info.get("max_target_travel_m", 0.)),
                max_target_force=float(info.get("max_target_contact_force_n", 0.)),
                success=bool(outcome.get("success", False))))

    def after_step(self, backend, rows):
        # For fused step/reset, this hook occurs after step but before reset.
        self.before_reset = dict(world_time=float(backend.engine.world.current_time),
                                 world_tick=int(backend.engine.world.current_time_step_index),
                                 slots=backend.snapshot())

    def save(self, path):
        if self.rows:
            arrays = {name: np.asarray([row[name] for row in self.rows]) for name in self.rows[0]}
            require(all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fiu"),
                    "Recorded physics contains nonfinite values")
            np.savez_compressed(path, **arrays)
        return len(self.rows)


class PanelLayoutAudit:
    """Reapply the old reset qualification against shared-engine tensor state."""
    def __init__(self, backend):
        from pxr import Usd, UsdPhysics
        from pressb.tasks.elevator.dataset_scene import _world_matrix
        self.backend = backend
        self.fixed, self.anchors = {}, {}
        engine = backend.engine
        for eid, prefix in enumerate(engine.prefixes):
            paths = (prefix + "/Piper", prefix + "/GlobalCamera", prefix + "/GlobalCamera/CameraLink/ColorCamera")
            self.fixed[eid] = {p: _world_matrix(engine.stage, p) for p in paths}
            anchors = []
            for prim in Usd.PrimRange(engine.stage.GetPrimAtPath(prefix)):
                if not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                for side, relation, attribute in ((0, joint.GetBody0Rel(), joint.GetLocalPos0Attr()),
                                                 (1, joint.GetBody1Rel(), joint.GetLocalPos1Attr())):
                    if not relation.GetTargets():
                        anchors.append(dict(joint_path=str(prim.GetPath()), world_side=side,
                            initial=np.asarray(attribute.Get(), dtype=float),
                            panel=str(prim.GetPath()).startswith(prefix + "/Panel/")))
            require(len(anchors) == 13, "Expected one fixed robot anchor and twelve button springs")
            self.anchors[eid] = anchors

    def check(self, ids):
        from pressb.tasks.elevator.dataset_scene import validate_panel_layout
        backend, task, engine = self.backend, self.backend.task, self.backend.engine
        position, orientation = task.buttons.get_world_poses()
        result = []
        for eid in ids:
            spec = backend.reset_specs[eid]
            source_x, source_y = task.source_offsets[eid]
            delta = np.array([spec["offset_x_m"]-source_x, spec["offset_y_m"]-source_y, 0.])
            anchors = [dict(joint_path=a["joint_path"], world_side=a["world_side"],
                world_position=(a["initial"] + (delta if a["panel"] else 0.)).tolist()) for a in self.anchors[eid]]
            rows = task.button_rows[eid]
            buttons = {floor: SimpleNamespace(get_world_pose=lambda idx=int(rows[floor-24]):
                       (position[idx], orientation[idx])) for floor in range(24, 36)}
            env = SimpleNamespace(prefix=engine.prefixes[eid], offset=engine.offsets[eid], config=task.scene,
                global_camera_path=engine.prefixes[eid] + "/GlobalCamera/CameraLink/ColorCamera",
                fixed_rig_transforms=self.fixed[eid], world_joint_anchors=anchors,
                panel_offset_x_m=spec["offset_x_m"], panel_offset_y_m=spec["offset_y_m"], buttons=buttons,
                button_info={floor: {"body_center": task.rests[eid, floor-24]} for floor in range(24, 36)})
            result.append({"env_id": eid, **validate_panel_layout(engine.world, env)})
        return result


def run(args, backend_factory=None):
    from pressb.control.pose_protocol import pose8_to_pose9
    from pressb.runtime.parallel import SharedWorldBackend
    from pressb.runtime.simulation import SimulationService

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [load_replay(args.replay_log, episode) for episode in (args.replay_episode_id or [DEFAULT_EPISODE])]
    chunks, layout, source = cases[0]
    digest = hashlib.sha256()
    for action, case_layout, case_source in cases:
        digest.update(np.asarray(action, dtype="<f8").tobytes())
        digest.update(json.dumps(case_layout, sort_keys=True).encode())
        require(args.max_seconds >= case_source["source_sim_seconds"], "Time limit is shorter than source replay")
    commands_hash = digest.hexdigest()
    config = dict(output=str(output / "simulation"), gpu=args.gpu, num_envs=args.num_envs,
        max_seconds=args.max_seconds, physics_backend=args.physics_backend,
        smoothing_window=3, single_gamma=.99, camera_resolution=224,
        render_subframes=1, light_settle_captures=1, light_settle_subframes=2,
        ik_iterations=16, log_actions=True, png_workers=4, cpu_threads=8)
    write_json(output / "config.json", config)
    report = dict(schema_version=1, task="elevator", physics_backend_requested=args.physics_backend,
        num_envs=args.num_envs, status="running", operations_passed=False,
        replay_source=source, replay_sources=[case[2] for case in cases], commands_sha256=commands_hash,
        scope="Fixed recorded action replay/contact/reset qualification, not policy success-rate evaluation",
        checks={}, stages={}, indexed_reset=[], layout_checks=[], images={})
    backend = service = None
    layout_audit = None
    trace = Trace()
    started = time.perf_counter()
    serial, run_id = 0, "qualify-physx-" + uuid.uuid4().hex
    ids = list(range(args.num_envs))

    def request(**kwargs):
        nonlocal serial
        serial += 1
        return dict(run_id=run_id, request_id=str(serial), **kwargs)

    def reset(selected, seed, options):
        rows = service.reset(request(env_ids=selected, seed=seed, options=options))["items"]
        if layout_audit is not None:
            report["layout_checks"].extend(layout_audit.check(selected))
        return rows

    def commands(rows, chunk=None, mixed=False):
        result = []
        for row in rows:
            action = (chunk if chunk is not None and (not mixed or row["env_id"] == 0)
                      else pose8_to_pose9(np.repeat(np.asarray(row["observation"]["state"])[None], 7, axis=0)).tolist())
            result.append(dict(env_id=row["env_id"], episode_id=row["episode_id"],
                               step_id=row["step_id"], actions=action))
        return result

    def compact(row):
        return {key: deepcopy(row[key]) for key in ("env_id", "episode_id", "step_id", "reward", "discount",
                                                    "terminated", "truncated", "info")}

    def rollout(rows, phase, replay_chunks=None):
        trace.phase = phase
        results = {}
        for index in range(math.ceil(args.max_seconds * 120 / 28)):
            chunk = replay_chunks[index] if replay_chunks is not None and index < len(replay_chunks) else None
            rows = service.step(request(items=commands(rows, chunk)))["items"]
            for row in rows:
                if row["terminated"] or row["truncated"]:
                    results[row["env_id"]] = row
            rows = [row for row in rows if not (row["terminated"] or row["truncated"])]
            if not rows:
                break
        require(not rows, f"{phase} exceeded its episode budget")
        final = [results[i] for i in sorted(results)]
        report["stages"][phase] = [compact(row) for row in final]
        report["images"][phase] = save_images(final, output / phase)
        return final

    try:
        if backend_factory is None:
            from pressb.tasks.elevator.parallel import create_backend
            backend_factory = create_backend
        backend = backend_factory(config)
        require(isinstance(backend, SharedWorldBackend), "Task bypassed the common backend")
        service = SimulationService(backend)
        metadata = service.health()
        report["metadata"] = metadata
        effective = metadata["execution"].get("gpu_dynamics")
        if effective is None:
            label = metadata["execution"].get("physics_backend", "").lower()
            effective = True if "gpu" in label else False if "cpu" in label else None
        require(effective is (args.physics_backend == "gpu"), "Effective solver differs from requested solver")
        report["checks"]["effective_solver_matches"] = True
        backend.add_observer(trace)
        layout_audit = PanelLayoutAudit(backend)
        rows = reset(ids, args.seed, {})
        report["images"]["initial"] = save_images(rows, output / "initial")
        holds = rollout(rows, "hold_timeout")
        require(all(r["truncated"] and not r["terminated"] and not r["info"]["success"] for r in holds),
                "Hold commands did not terminate by timeout")
        require(all(r["info"]["episode_physics_steps"] == round(args.max_seconds * 120) for r in holds),
                "Hold timeout did not match configured physical duration")
        report["checks"]["exact_hold_timeout"] = True

        ready = []
        for subset in ([0], ids[1:]):
            peers = [i for i in ids if i not in subset]
            clock = float(backend.engine.world.current_time)
            tick = int(backend.engine.world.current_time_step_index)
            before = backend.snapshot(peers)
            slots = deepcopy({i: service.slots[i] for i in peers})
            rows = reset(subset, args.seed + 1, layout)
            require(before == backend.snapshot(peers), "Indexed reset changed peer physical state")
            require(clock == backend.engine.world.current_time and tick == backend.engine.world.current_time_step_index,
                    "Indexed reset advanced the physics clock")
            require(all(service.slots[i] == slots[i] for i in peers), "Indexed reset changed peer episode identity")
            ready.extend(rows)
            report["indexed_reset"].append(dict(reset_env_ids=subset, peer_env_ids=peers,
                zero_ticks=True, peer_state_identical=True, peer_protocol_identical=True))
        report["checks"]["indexed_reset_isolation"] = True
        for index, (case_chunks, case_layout, _) in enumerate(cases):
            rows = sorted(ready, key=lambda x: x["env_id"]) if index == 0 else reset(ids, args.seed+10+index, case_layout)
            phase = "success_replay" if len(cases) == 1 else f"success_replay_{index:02d}_floor{case_layout['floor']}"
            replay = rollout(rows, phase, replay_chunks=case_chunks)
            require(all(r["info"]["success"] and r["terminated"] and not r["truncated"] for r in replay),
                    f"Recorded trajectory for floor {case_layout['floor']} did not press target in every clone")
            require(all(r["info"]["pressed_floors"] == [case_layout["floor"]] and
                        r["info"]["max_target_travel_m"] >= .0015 and
                        r["info"]["max_target_contact_force_n"] > .02 for r in replay),
                    "Success lacked target-only physical contact and button travel")
        report["checks"]["physical_contact_success"] = True

        # Exercise a real contact termination and indexed reset while peers are
        # still executing holds. The hook snapshot is immediately before reset.
        rows = reset(ids, args.seed + 2, layout)
        trace.phase = "mixed_fused"
        mixed_done = False
        for index in range(math.ceil(args.max_seconds * 120 / 28)):
            chunk = chunks[index] if index < len(chunks) else None
            response = service.step_and_reset(request(items=commands(rows, chunk, mixed=True),
                terminal_observation_mode="state_only_if_no_bootstrap", reset_seed=args.seed+3,
                reset_options=layout, max_active_envs=args.num_envs))
            advanced, fresh = response["items"], response["reset_items"]
            if fresh:
                ended = [r for r in advanced if r["terminated"] or r["truncated"]]
                require([r["env_id"] for r in fresh] == [0] and len(ended) == 1 and ended[0]["info"]["success"],
                        "Mixed rollout did not isolate a target press to env0")
                peers = ids[1:]
                before = trace.before_reset
                require({str(i): before["slots"][str(i)] for i in peers} == backend.snapshot(peers),
                        "Fused reset changed the live peer state")
                require(before["world_time"] == backend.engine.world.current_time and
                        before["world_tick"] == backend.engine.world.current_time_step_index,
                        "Fused reset advanced physics")
                require(fresh[0]["episode_id"] != ended[0]["episode_id"] and fresh[0]["step_id"] == 0,
                        "Fused reset did not provide a fresh episode identity")
                require(ended[0].get("terminal_observation_omitted") is True and
                        fresh[0]["info"]["episode_physics_steps"] == 0,
                        "Fused reset mixed terminal and reset records")
                report["layout_checks"].extend(layout_audit.check([0]))
                report["stages"]["mixed_fused"] = [compact(row) for row in advanced]
                report["checks"]["fused_reset_isolation"] = True
                report["images"]["fused_reset"] = save_images(fresh, output / "fused_reset")
                mixed_done = True
                break
            rows = advanced
        require(mixed_done, "Mixed contact/fused reset was not exercised")
        report["checks"]["legacy_layout_invariants"] = True
        report["checks"]["finite_tick_trajectories"] = True
        report["operations_passed"] = True
        report["status"] = "passed"
        report["close_run"] = service.close_run({"run_id": run_id})
        require(not report["close_run"]["restart_required"], "Shared backend unexpectedly requires restart")
    except BaseException as error:
        report.update(status="failed", operations_passed=False,
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        try:
            report["trace_rows"] = trace.save(output / "trajectory.npz")
        except Exception as error:
            report.update(status="failed", operations_passed=False,
                          trace_error=f"{type(error).__name__}: {error}")
        report["wall_seconds"] = time.perf_counter() - started
        if backend is not None:
            report["timing"] = deepcopy(backend.timings)
        # Final operation result precedes close; do not depend on post-Kit code.
        write_json(output / "report.json", report)
        print(json.dumps({"output": str(output), "operations_passed": report["operations_passed"],
                          "physics_backend": args.physics_backend, "error": report.get("error")}), flush=True)
        if backend is not None:
            backend.close()
    return 0 if report["operations_passed"] else 1


def compare(cpu, gpu, output, *, position_tolerance_m=.002, joint_tolerance_rad=.02,
            terminal_tolerance_ticks=4):
    """Compare matching observed ticks; no interpolation across terminations."""
    inputs = [Path(cpu).resolve(), Path(gpu).resolve()]
    reports = [json.loads((p / "report.json").read_text()) for p in inputs]
    require([r["physics_backend_requested"] for r in reports] == ["cpu", "gpu"], "Provide CPU then GPU run")
    require(reports[0]["commands_sha256"] == reports[1]["commands_sha256"], "Replay commands differ")
    require(reports[0]["num_envs"] == reports[1]["num_envs"], "Environment counts differ")
    configs = [json.loads((p / "config.json").read_text()) for p in inputs]
    for cfg in configs:
        cfg.pop("physics_backend"); cfg.pop("output"); cfg.pop("gpu", None)
    require(configs[0] == configs[1], "Non-solver run configuration differs")
    traces = [np.load(p / "trajectory.npz", allow_pickle=False) for p in inputs]
    def keyed(trace):
        keys = list(zip(trace["phase"].tolist(), trace["env_id"].tolist(), trace["tick"].tolist()))
        require(len(keys) == len(set(keys)), "Duplicate phase/env/tick in trace")
        return {key: i for i, key in enumerate(keys)}
    maps = [keyed(t) for t in traces]
    common = sorted(set(maps[0]) & set(maps[1]))
    require(common, "No aligned physics ticks")
    indices = [[m[k] for k in common] for m in maps]
    positions = np.linalg.norm(traces[0]["state"][indices[0], :3] - traces[1]["state"][indices[1], :3], axis=-1)
    joints = np.abs(traces[0]["joints"][indices[0], :6] - traces[1]["joints"][indices[1], :6])
    terminal_differences = []
    phases = [key for key in reports[0]["stages"] if key == "hold_timeout" or key.startswith("success_replay")]
    require(phases == [key for key in reports[1]["stages"] if key == "hold_timeout" or key.startswith("success_replay")],
            "Compared runs have different completed replay stages")
    for phase in phases:
        left, right = (r["stages"].get(phase, []) for r in reports)
        require(len(left) == len(right) == reports[0]["num_envs"], "Replay stage omitted an environment outcome")
        for a, b in zip(left, right, strict=True):
            require(a["env_id"] == b["env_id"], "Terminal ordering differs")
            for row, mapping in zip((a, b), maps):
                ticks = sorted(key[2] for key in mapping if key[0] == phase and key[1] == row["env_id"])
                require(ticks == list(range(1, row["info"]["episode_physics_steps"]+1)),
                        "Physics trajectory has missing or unexpected ticks")
            terminal_differences.append(dict(stage=phase, env_id=a["env_id"],
                ticks=abs(a["info"]["episode_physics_steps"] - b["info"]["episode_physics_steps"]),
                same_outcome=all(a[k] == b[k] for k in ("terminated", "truncated"))
                    and a["info"]["success"] == b["info"]["success"]))
    checks = dict(both_operation_suites_passed=all(r["operations_passed"] for r in reports),
        same_outcomes=all(r["same_outcome"] for r in terminal_differences),
        within_terminal_tolerance=all(r["ticks"] <= terminal_tolerance_ticks for r in terminal_differences),
        within_position_tolerance=bool(np.max(positions) <= position_tolerance_m),
        within_joint_tolerance=bool(np.max(joints) <= joint_tolerance_rad))
    result = dict(schema_version=1, passed=all(checks.values()), checks=checks,
        sources=[str(p) for p in inputs], commands_sha256=reports[0]["commands_sha256"],
        deployment_fields_excluded_from_configuration_match=["output", "gpu"],
        aligned_tick_samples=len(common), unmatched_samples=[len(m)-len(common) for m in maps],
        tcp_position_difference_m=dict(max=float(np.max(positions)), rms=float(np.sqrt(np.mean(positions**2)))),
        arm_joint_difference_rad=dict(max=float(np.max(joints)), rms=float(np.sqrt(np.mean(joints**2)))),
        tolerances=dict(tcp_max_position_m=position_tolerance_m, arm_max_joint_rad=joint_tolerance_rad,
                        terminal_physics_ticks=terminal_tolerance_ticks), terminal_differences=terminal_differences,
        limits="Tolerance-based fixed-trajectory check; not bitwise physics equality or learned-policy equivalence.")
    for trace in traces:
        trace.close()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), "Comparison output already exists")
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run")
    command.add_argument("--physics-backend", choices=("cpu", "gpu"), required=True)
    command.add_argument("--gpu", type=int, default=0)
    command.add_argument("--num-envs", type=int, default=2)
    command.add_argument("--max-seconds", type=float, default=2.)
    command.add_argument("--seed", type=int, default=20260994)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--replay-log", type=Path, default=DEFAULT_REPLAY)
    command.add_argument("--replay-episode-id", action="append", help="Repeat to qualify several recorded episodes in one World")
    command = commands.add_parser("compare")
    command.add_argument("--cpu", type=Path, required=True)
    command.add_argument("--gpu", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--position-tolerance-m", type=float, default=.002)
    command.add_argument("--joint-tolerance-rad", type=float, default=.02)
    command.add_argument("--terminal-tolerance-ticks", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.num_envs < 2 or args.gpu < 0 or args.seed < 0:
            parser.error("Require >=2 environments and nonnegative GPU/seed")
        if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 120 or abs(args.max_seconds*120-round(args.max_seconds*120)) > 1e-7:
            parser.error("max-seconds must be positive and lie on a 120 Hz tick")
        return run(args)
    if any(not math.isfinite(x) or x <= 0 for x in (args.position_tolerance_m, args.joint_tolerance_rad)) or args.terminal_tolerance_ticks < 0:
        parser.error("Require positive numerical tolerances and nonnegative terminal tolerance")
    return compare(args.cpu, args.gpu, args.output, position_tolerance_m=args.position_tolerance_m,
                   joint_tolerance_rad=args.joint_tolerance_rad, terminal_tolerance_ticks=args.terminal_tolerance_ticks)


if __name__ == "__main__":
    raise SystemExit(main())
