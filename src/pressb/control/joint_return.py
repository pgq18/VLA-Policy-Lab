"""Paced joint return along a robot-supplied, validated alignment plan.

This motion is outside policy transitions. The robot adapter supplies its
existing planner and command projection; this module does not perform IK or
claim obstacle avoidance. Arm return preserves the gripper command; a separate
gripper initialization can follow once the arm has reached its target.
"""
from __future__ import annotations

from collections import deque
import math
import time

import numpy as np


def plan_joint_return(start, target, *, limits, speed_rad_s=.2, min_duration_s=5.,
                      max_flange_speed_m_s=.05):
    """Retiming of the recorder's quintic joint path with the same checks.

Keep the reference 0.4 rad/s² acceleration constraint. Robot-provided joint,
workspace and FK checks are applied to the same <=0.01 rad path samples;
return-specific timing need not alter the recorder's startup configuration.
"""
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    if (start.ndim != 1 or not len(start) or target.shape != start.shape
            or not np.isfinite(start).all() or not np.isfinite(target).all()):
        raise ValueError("Joint return planning requires matching finite joint vectors")
    for name, value in (("speed_rad_s", speed_rad_s), ("min_duration_s", min_duration_s),
                        ("max_flange_speed_m_s", max_flange_speed_m_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    velocity = np.asarray(limits.velocity, dtype=float)
    if velocity.shape != start.shape or not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError("Invalid robot joint velocity limits")
    delta = np.abs(target-start)
    count = max(101, int(np.ceil(delta.max()/.01))+1)
    xyz = np.array([limits.check(q, "return path") for q in np.linspace(start, target, count)])
    if xyz.shape != (count, 3) or not np.isfinite(xyz).all():
        raise ValueError("Return FK must provide finite flange XYZ positions")
    derivative = np.linalg.norm(np.diff(xyz, axis=0), axis=1)*(count-1)
    components = {"minimum": float(min_duration_s),
                  "joint_velocity": float(np.max(1.875*delta/np.minimum(velocity, speed_rad_s))),
                  "joint_acceleration": float(np.sqrt(5.773503*delta.max()/.4)),
                  "flange_velocity": 1.875*float(derivative.max())/max_flange_speed_m_s}
    return {"start_joints": start.tolist(), "target_joints": target.tolist(),
            "duration_s": max(components.values()), "duration_components_s": components,
            "path_samples": count, "max_joint_speed_rad_s": speed_rad_s,
            "max_joint_acceleration_rad_s2": .4, "max_flange_speed_m_s": max_flange_speed_m_s,
            "obstacle_collision_precheck": False}


class JointReturnTimeout(TimeoutError):
    """Convergence failure with the final measured/applied trajectory samples."""

    def __init__(self, report):
        self.report = report
        super().__init__(
            f"Return did not reach the initial pose; J{report['worst_joint']} error "
            f"{report['max_joint_error_rad']:.6f} rad exceeds tolerance "
            f"{report['tolerance_rad']:.6f} rad after {report['wall_seconds']:.2f} s")


class GripperResetTimeout(TimeoutError):
    def __init__(self, report):
        self.report = {"status": "failed", "gripper_reset": report}
        super().__init__(f"Gripper reset did not reach its initial state: target={report['target']:.6f}, "
                         f"measured={report['measured']:.6f}, tolerance={report['tolerance']:.6f}")


def reset_gripper(initial_command, target, *, read_state, command, control_hz,
                  progress=lambda *_: None, command_ready=lambda: True,
                  tolerance=.025, timeout_s=5., stable_s=.2,
                  clock=time.monotonic, sleep=time.sleep):
    """Hold arm targets, submit the gripper target once, then await feedback.

The last coordinate uses the caller's gripper units. Completion requires fresh
feedback within tolerance and no pending/in-flight device command for stable_s.
The existing robot adapter continues owning all SDK commands and limits.
"""
    hold = np.asarray(initial_command, dtype=float).copy()
    if hold.ndim != 1 or len(hold) < 2 or not np.isfinite(hold).all():
        raise ValueError("Gripper reset requires a finite arm+gripper command")
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
        raise ValueError("Gripper reset target must be finite")
    for name, value in (("control_hz", control_hz), ("tolerance", tolerance),
                        ("timeout_s", timeout_s), ("stable_s", stable_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if stable_s >= timeout_s:
        raise ValueError("Gripper stable_s must be less than timeout_s")
    progress(hold.copy())
    initial = np.asarray(read_state(), dtype=float)
    if initial.shape != hold.shape or not np.isfinite(initial).all():
        raise RuntimeError("Invalid initial gripper feedback")
    initial_gripper = float(initial[-1])
    hold[-1] = target
    began = clock()
    applied = np.asarray(command(hold), dtype=float).copy()
    if applied.shape != hold.shape or not np.isfinite(applied).all():
        raise RuntimeError("Gripper reset produced an invalid applied command")
    stable_since = None
    while True:
        tick = clock()
        progress(applied.copy())
        measured = np.asarray(read_state(), dtype=float)
        if measured.shape != hold.shape or not np.isfinite(measured).all():
            raise RuntimeError("Invalid measured state during gripper reset")
        ready = bool(command_ready())
        within = abs(float(measured[-1]) - target) <= tolerance
        stable_since = (tick if stable_since is None else stable_since) if within and ready else None
        report = {"target": target, "measured": float(measured[-1]), "initial": initial_gripper,
                  "tolerance": tolerance, "wall_seconds": tick-began, "stable_s": stable_s,
                  "command_ready": ready, "command_count": 1,
                  "already_at_initial": abs(initial_gripper-target) <= tolerance,
                  "last_applied_action": applied.tolist(), "measured_state": measured.tolist()}
        if stable_since is not None and tick-stable_since >= stable_s:
            return {**report, "status": "completed"}
        if tick-began >= timeout_s:
            raise GripperResetTimeout({**report, "status": "failed"})
        sleep(max(0., tick + 1/control_hz - clock()))


def execute_joint_return(start, target_joints, *, duration_s, control_hz,
                         read_state, command, progress=lambda *_: None,
                         tolerance_rad=.05, settle_timeout_s=5.,
                         clock=time.monotonic, sleep=time.sleep):
    """Quintic joint interpolation, then measured convergence to the target."""
    start = np.asarray(start, dtype=float).copy()
    target = np.asarray(target_joints, dtype=float).copy()
    if (start.ndim != 1 or target.shape != (len(start) - 1,)
            or not np.isfinite(start).all() or not np.isfinite(target).all()):
        raise ValueError("Return requires finite arm joints and one preserved gripper coordinate")
    for name, value in (("duration_s", duration_s), ("control_hz", control_hz),
                        ("tolerance_rad", tolerance_rad), ("settle_timeout_s", settle_timeout_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    began, commands = clock(), 0
    last = start.copy()
    samples = deque(maxlen=min(32, max(2, math.ceil(control_hz))))
    while True:
        tick = clock()
        progress(last.copy())  # Includes explicit abort checks, even while settling.
        fraction = min(1., max(0., (tick - began) / duration_s))
        blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
        request = np.r_[start[:-1] + blend * (target - start[:-1]), start[-1]]
        last = np.asarray(command(request), dtype=float)
        if last.shape != start.shape or not np.isfinite(last).all():
            raise RuntimeError("Return command produced invalid applied state")
        commands += 1
        sleep(max(0., tick + 1 / control_hz - clock()))
        measured = np.asarray(read_state(), dtype=float)
        if measured.shape != start.shape or not np.isfinite(measured).all():
            raise RuntimeError("Invalid measured state during return")
        errors = measured[:-1] - target
        worst = int(np.argmax(np.abs(errors)))
        error = float(abs(errors[worst]))
        elapsed = clock() - began
        samples.append({"elapsed_s": elapsed, "measured_joints_rad": measured[:-1].tolist(),
                        "applied_joints_rad": last[:-1].tolist()})
        if fraction >= 1 and error <= tolerance_rad:
            progress(last.copy())
            return {"status": "completed", "wall_seconds": elapsed,
                    "control_steps": commands, "start_joints_rad": start[:-1].tolist(),
                    "target_joints_rad": target.tolist(),
                    "measured_joints_rad": measured[:-1].tolist(), "max_joint_error_rad": error,
                    "joint_errors_rad": errors.tolist(), "worst_joint": worst + 1,
                    "tolerance_rad": tolerance_rad, "planned_duration_s": duration_s,
                    "gripper_command": float(start[-1]), "last_applied_action": last.tolist(),
                    "last_samples": list(samples)}
        if elapsed > duration_s + settle_timeout_s:
            raise JointReturnTimeout({"status": "failed", "wall_seconds": elapsed,
                "control_steps": commands, "start_joints_rad": start[:-1].tolist(),
                "target_joints_rad": target.tolist(),
                "measured_joints_rad": measured[:-1].tolist(), "joint_errors_rad": errors.tolist(),
                "worst_joint": worst + 1, "max_joint_error_rad": error,
                "tolerance_rad": tolerance_rad, "planned_duration_s": duration_s,
                "settle_timeout_s": settle_timeout_s,
                "gripper_command": float(start[-1]), "last_applied_action": last.tolist(),
                "last_samples": list(samples)})
