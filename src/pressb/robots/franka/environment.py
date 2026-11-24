"""FR3 episode boundaries with an optional, operator-abortable joint return."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from pressb.control.joint_return import (
    GripperResetTimeout, execute_joint_return, plan_joint_return, reset_gripper,
)
from pressb.runtime.contract import finite_array, fingerprint
from pressb.runtime.real_robot import RealRobotBackend


def _return_settings(value):
    if not isinstance(value, dict):
        raise ValueError("return_to_initial must be an object")
    defaults = dict(speed_rad_s=.2, min_duration_s=5., max_flange_speed_m_s=.05,
                    tolerance_rad=.05, already_at_initial_tolerance_rad=.01, settle_timeout_s=5.,
                    gripper_tolerance=.025, gripper_timeout_s=5., gripper_stable_s=.2)
    if set(value) - set(defaults) - {"enabled", "joints_rad", "gripper_closure", "source"}:
        raise ValueError("Unknown return_to_initial setting")
    config = {"enabled": False, "gripper_closure": None, **defaults, **deepcopy(value)}
    if type(config["enabled"]) is not bool:
        raise ValueError("return_to_initial.enabled must be boolean")
    for key in defaults:
        number = config[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"return_to_initial.{key} must be finite and positive")
    if config["speed_rad_s"] > .3:
        raise ValueError("FR3 return speed exceeds 0.3 rad/s")
    if config["already_at_initial_tolerance_rad"] > config["tolerance_rad"]:
        raise ValueError("Return skip tolerance exceeds convergence tolerance")
    if config["gripper_stable_s"] >= config["gripper_timeout_s"]:
        raise ValueError("Gripper stable interval must be shorter than its timeout")
    closure = config["gripper_closure"]
    if closure is not None and (type(closure) not in (int, float) or not 0 <= closure <= 1):
        raise ValueError("gripper_closure must be null or in [0, 1]")
    if config["enabled"] or "joints_rad" in config:
        config["joints_rad"] = finite_array(config.get("joints_rad"), (7,), "joints_rad", dtype=float).tolist()
    return config


class FrankaReturnBackend(RealRobotBackend):
    """Run enabled recovery before operator readiness or the final run close."""

    def __init__(self, config, **kwargs):
        config = dict(config)
        self.return_config = _return_settings(config.pop("return_to_initial", {}))
        self._return_pending = False
        super().__init__(config, **kwargs)
        if self.return_config["enabled"]:
            try:
                for hook in ("begin_return", "finish_return", "fail_return", "check_abort"):
                    if not callable(getattr(self.task, hook, None)):
                        raise ValueError(f"FR3 return_to_initial requires the operator task hook {hook}")
                target = self.return_config["joints_rad"]
                if self.width != 8 or np.any(target < np.asarray(self.action["lower"][:-1])) or np.any(
                        target > np.asarray(self.action["upper"][:-1])):
                    raise ValueError("Return joints must be within the FR3 action bounds")
            except BaseException:
                self.close()
                raise
        self.metadata["execution"]["return_to_initial"] = deepcopy(self.return_config)
        self.metadata["task_config_sha256"] = fingerprint({
            "task": self.metadata["task_config_sha256"], "return_to_initial": self.return_config,
        })

    def _progress(self, command):
        self._check()
        self.task.check_abort()
        self._last_command = np.array(command, dtype=float, copy=True)
        self._last_activity = self._clock()

    def _return_arm(self):
        cfg = self.return_config
        target = np.asarray(cfg["joints_rad"])
        start = finite_array(self.robot.read_state(), (8,), "measured FR3 state", dtype=float).copy()
        previous = self.robot.last_command
        if previous is not None:
            previous = finite_array(previous, (8,), "last FR3 command", dtype=float)
            start[-1] = previous[-1]
        self._progress(start)
        error = float(np.max(np.abs(start[:-1] - target)))
        tolerance = cfg["already_at_initial_tolerance_rad"]
        if error <= tolerance and (previous is None or np.max(np.abs(previous[:-1] - target)) <= tolerance):
            return dict(status="completed", already_at_initial=True, control_steps=0, wall_seconds=0.,
                        measured_joints_rad=start[:-1].tolist(), target_joints_rad=target.tolist(),
                        max_joint_error_rad=error, tolerance_rad=cfg["tolerance_rad"],
                        already_at_initial_tolerance_rad=tolerance, gripper_command=float(start[-1]),
                        last_applied_action=(start if previous is None else previous).tolist())
        plan = plan_joint_return(
            start[:-1], target, limits=self.robot.limits, speed_rad_s=cfg["speed_rad_s"],
            min_duration_s=cfg["min_duration_s"], max_flange_speed_m_s=cfg["max_flange_speed_m_s"],
        )
        result = execute_joint_return(
            start, target, duration_s=plan["duration_s"], control_hz=self.hz, command=self.robot.command_arm,
            read_state=self.robot.read_state, tolerance_rad=cfg["tolerance_rad"],
            settle_timeout_s=cfg["settle_timeout_s"], progress=self._progress, clock=self._clock, sleep=self._sleep,
        )
        return {**result, "already_at_initial": False, "plan": plan}

    def _return_arm_and_gripper(self):
        cfg = self.return_config
        began = self._clock()
        arm = self._return_arm()
        if cfg["gripper_closure"] is None:
            return arm
        try:
            hand = reset_gripper(
                arm["last_applied_action"], cfg["gripper_closure"], control_hz=self.hz,
                read_state=self.robot.read_state, command=self.robot.command, progress=self._progress,
                command_ready=self.robot.gripper_command_idle, tolerance=cfg["gripper_tolerance"],
                timeout_s=cfg["gripper_timeout_s"], stable_s=cfg["gripper_stable_s"],
                clock=self._clock, sleep=self._sleep,
            )
        except GripperResetTimeout as error:
            error.report = {**arm, **error.report, "arm_return_wall_seconds": arm["wall_seconds"],
                            "wall_seconds": self._clock() - began}
            raise
        return {**arm, "arm_already_at_initial": arm["already_at_initial"],
                "already_at_initial": arm["already_at_initial"] and hand["already_at_initial"],
                "arm_return_wall_seconds": arm["wall_seconds"], "wall_seconds": self._clock() - began,
                "gripper_reset": hand, "gripper_command": cfg["gripper_closure"],
                "control_steps": arm["control_steps"] + hand["command_count"],
                "last_applied_action": hand["last_applied_action"]}

    def _run_return(self, *, seed=None, options=None):
        self._check()
        if not self.actuation_enabled:
            raise RuntimeError("Actuation is disabled; FR3 return requires explicit actuation")
        try:
            self.task.check_abort()
            if not self._return_pending:
                self.task.begin_return(self.return_config["joints_rad"], reset_seed=seed, reset_options=options)
            self._connect()
            # The RPC server serializes requests. Keep the shared hardware
            # watchdog active throughout motion, including blocked SDK calls.
            self._last_activity = self._clock()
            self._active = True
            report = self._return_arm_and_gripper()
            self._progress(report["last_applied_action"])
            self.task.finish_return(report)
            self._return_pending = False
            return report
        except BaseException as error:
            self._fail(error)
            self.task.fail_return(error)
            raise
        finally:
            self._active = False

    def reset(self, env_ids, seed, options=None):
        if not self.return_config["enabled"]:
            return super().reset(env_ids, seed, options)
        if self._active or env_ids != [0] or type(env_ids[0]) is not int:
            raise ValueError("FR3 return requires an inactive env_ids=[0]")
        if type(seed) is not int or seed < 0 or (options is not None and not isinstance(options, dict)):
            raise ValueError("FR3 return requires a nonnegative seed and object options")
        report = self._run_return(seed=seed, options={} if options is None else options)
        rows = super().reset(env_ids, seed, options)
        rows[0]["info"].update(return_to_initial=report, reset_semantics="FR3 return then operator readiness")
        return rows

    def _queue_return(self):
        try:
            self.task.begin_return(self.return_config["joints_rad"])
        except BaseException as error:
            self._fail(error)
            raise
        self._return_pending = True

    def step(self, commands):
        rows = super().step(commands)
        row = rows[0]
        if self.return_config["enabled"] and (row["terminated"] or row["truncated"]):
            self._queue_return()
            row["info"].update(
                return_to_initial={"status": "pending", "target_joints_rad": self.return_config["joints_rad"].copy()},
                task_status=self.task.status(),
            )
        return rows

    def end_episode(self, env_ids):
        if self.return_config["enabled"] and not self._active:
            raise ValueError("FR3 end_episode requires an active episode")
        super().end_episode(env_ids)
        if self.return_config["enabled"]:
            self._queue_return()

    def close_run(self):
        if self._return_pending:
            self._run_return()
        super().close_run()
