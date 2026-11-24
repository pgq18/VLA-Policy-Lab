"""Physical FR3 arm and Franka Hand using the station's direct libfranka controller."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib import import_module
import math
import threading

import numpy as np

from pressb.runtime.contract import finite_array
from .gripper import GripperWorker
from .safety import SafetyLimits, checked_state


# Station controller URDF and capture-time native limits, in radians. The wire
# contract is available offline. Check it against the live controller on connect.
_JOINT_LOWER = (-2.9007400166666666, -1.8360900166666667, -2.9007400166666666,
                -3.077020016666667, -2.87630335, 0.43982265, -3.05083335)
_JOINT_UPPER = (2.9007400166666666, 1.8360900166666667, 2.9007400166666666,
                -0.11693708333333333, 2.87630335, 4.62163335, 3.05083335)
_HAND_TRAVEL_M = .08
# The station Hand's calibrated maximum can exceed nominal travel by tens
# of microns. The worker treats aperture differences within 1 mm as settled.
_HAND_WIDTH_TOLERANCE_M = .001


@dataclass(frozen=True)
class FrankaHardwareConfig:
    robot_ip: str = "172.16.0.2"
    actuation_enabled: bool = False
    control_hz: float = 25.
    max_joint_step: float = .05
    alignment_tolerance: float = .15
    enforce_realtime: bool = True
    gripper_speed: float = .1

    def __post_init__(self):
        if not isinstance(self.robot_ip, str) or not self.robot_ip.strip():
            raise ValueError("robot_ip must be a nonempty address")
        if type(self.actuation_enabled) is not bool or type(self.enforce_realtime) is not bool:
            raise ValueError("actuation_enabled and enforce_realtime must be booleans")
        for name, maximum in (("control_hz", math.inf), ("max_joint_step", .1),
                              ("alignment_tolerance", .3), ("gripper_speed", .1)):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 < value <= maximum):
                raise ValueError(f"{name} must be finite and in (0, {maximum}]")


def _load_native():
    try:
        return import_module("pressb.robots.franka._franka_native")
    except ImportError as error:
        raise ImportError(
            "Build the PressB FR3 backend on the robot host with the device interpreter: "
            "python scripts/franka/build_native.py --libfranka-dir ~/libfranka/build"
        ) from error


class FrankaHardware:
    """A single physical arm/Hand. Only an explicit command or hold starts control."""

    def __init__(self, config=None):
        self.config = FrankaHardwareConfig(**({} if config is None else config))
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._hardware = self._hand = self._limits = None
        self._max_width = None
        self._connected = self._started = False
        self._last_command = None
        self._submitted = 0
        self._fault = None
        self._device = {}
        self._feedback = {}

    def describe(self):
        names = [f"joint{i}" for i in range(1, 8)] + ["gripper_closure"]
        units = ["rad"] * 7 + ["unitless"]
        return {
            "robot_type": "franka_fr3", "actuation_enabled": self.config.actuation_enabled,
            "state": {"names": names, "units": units},
            "action": {"representation": "joint_position", "frame": "robot_base",
                       "names": names.copy(), "units": units.copy(), "control_hz": self.config.control_hz,
                       "lower": [*_JOINT_LOWER, 0.], "upper": [*_JOINT_UPPER, 1.]},
            "config": asdict(self.config), "transport": "direct libfranka", "arm_servo_hz": 1000,
            "gripper_coordinate": "0=open, 1=closed; width = calibrated maximum * (1 - closure)",
        }

    def _check_open(self):
        if self._stop_requested.is_set():
            raise RuntimeError("FR3 session stopped or closed; restart required")

    @contextmanager
    def _operation(self, *, connecting=False):
        with self._lock:
            self._check_open()
            if not connecting and not self._connected:
                raise RuntimeError("FR3 is disconnected")
            try:
                yield
                self._check_open()
            except BaseException as error:
                self._fault = f"{type(error).__name__}: {error}"
                self.close()
                raise

    def connect(self):
        with self._operation(connecting=True):
            if self._connected:
                return
            native = _load_native()
            if not getattr(native, "gripper_stop_concurrency_fixed", False):
                raise RuntimeError("FR3 requires the native SDK with the concurrent gripper stop fix")
            hardware = self._hardware = native.Hardware(self.config.robot_ip, self.config.enforce_realtime)
            urdf = hardware.urdf()
            limits = SafetyLimits.from_urdf(urdf, native.Kinematics(urdf), native.Limits())
            if (not np.allclose(limits.lower, _JOINT_LOWER, atol=1e-8, rtol=0)
                    or not np.allclose(limits.upper, _JOINT_UPPER, atol=1e-8, rtol=0)):
                raise RuntimeError("Live FR3 joint limits differ from the advertised station controller limits")
            self._limits = limits
            hardware.set_limits(limits.lower.tolist(), limits.upper.tolist())
            checked_state(hardware, stationary=True)
            width, maximum, grasped = hardware.gripper_state()
            if not np.isfinite([width, maximum]).all() or maximum <= 0:
                raise RuntimeError("Invalid calibrated Franka Hand feedback")
            self._max_width = min(_HAND_TRAVEL_M, float(maximum))
            self._device = {"libfranka_version": native.libfranka_version,
                            "joint_lower": limits.lower.tolist(), "joint_upper": limits.upper.tolist(),
                            "workspace_lower": limits.xyz_lower.tolist(), "workspace_upper": limits.xyz_upper.tolist(),
                            "gripper_max_width_m": self._max_width, "reported_gripper_max_width_m": float(maximum)}
            self._hand = GripperWorker(hardware, self._max_width, self.config.gripper_speed,
                                       float(width), bool(grasped), command_error=native.CommandError)
            self._connected = True

    def _sample(self):
        arm = checked_state(self._hardware)
        if self._started and not arm["running"]:
            raise RuntimeError("FR3 arm control stopped unexpectedly")
        hand = self._hand.snapshot()
        width = float(hand["width"])
        if not math.isfinite(width):
            raise RuntimeError("Nonfinite Franka Hand width")
        closure = float(np.clip(1. - width / self._max_width, 0., 1.))
        self._feedback = {"arm_timestamp_ns": arm["flange_timestamp_ns"], "arm_control_running": arm["running"],
                          "gripper": hand}
        return np.array([*arm["q"], closure], dtype=np.float64)

    def read_state(self):
        with self._operation():
            return self._sample()

    def _check_actuation(self):
        if not self.config.actuation_enabled:
            raise RuntimeError("FR3 actuation is disabled")

    def command(self, vector):
        return self._command(vector, move_hand=True)

    def command_arm(self, vector):
        """Submit only the arm part of a return tick, preserving the Hand's current intent."""
        return self._command(vector, move_hand=False)

    def _command(self, vector, *, move_hand):
        self._check_actuation()
        with self._operation():
            target = finite_array(vector, (8,), "FR3 command", dtype=np.float64).copy()
            if not 0 <= target[-1] <= 1:
                raise ValueError("FR3 gripper_closure must be in [0,1]")
            return self._send(target, self._sample(), move_hand=move_hand)

    def _check_hand_calibration(self):
        sample = self._feedback["gripper"]
        maximum = float(sample["max_width"])
        tolerance = _HAND_WIDTH_TOLERANCE_M
        # Preserve raw feedback for read-only diagnosis. Before actuation a
        # corrupt calibration (the station once returned 160.5 mm) must never
        # turn a clipped observation into a physical 80 mm aperture request.
        initial_max = self._device["reported_gripper_max_width_m"]
        if (not 0 < maximum <= _HAND_TRAVEL_M + tolerance
                or initial_max > _HAND_TRAVEL_M + tolerance
                or abs(min(maximum, _HAND_TRAVEL_M) - self._max_width) > tolerance
                or not -tolerance <= sample["width"] <= self._max_width + tolerance):
            raise RuntimeError("Franka Hand calibration disagrees with physical travel; inspect width and maximum before actuation")

    def _send(self, target, measured, *, move_hand):
        self._check_hand_calibration()
        if not move_hand:
            target[-1] = measured[-1] if self._last_command is None else self._last_command[-1]
        limits = self._limits
        q = measured[:7]
        # The runtime transports float32. Restore exact bound values if rounding
        # an advertised limit to the wire dtype produced the adjacent float.
        raw = target[:7]
        for bound in (limits.lower, limits.upper):
            rounded = raw == bound.astype(np.float32)
            raw[rounded] = bound[rounded]
        limits.check(raw, "requested FR3 target")
        delta = raw - q
        if not self._started and np.max(np.abs(delta)) > self.config.alignment_tolerance:
            raise ValueError("First FR3 command exceeds the measured startup alignment tolerance")
        raw[:] = np.clip(raw, q - self.config.max_joint_step, q + self.config.max_joint_step)
        limits.check(raw, "projected FR3 target")
        self._check_open()
        if not self._started:
            self._hardware.start(q.tolist(), 0.)
            self._started = True
            self._check_open()
        self._hardware.set_target(raw.tolist())
        self._check_open()
        if move_hand:
            self._hand.submit(float(target[-1]))
        self._last_command = target.copy()
        self._submitted += 1
        return target

    def hold(self):
        self._check_actuation()
        with self._operation():
            measured = self._sample()
            return self._send(measured.copy(), measured, move_hand=False)

    @property
    def limits(self):
        """Live joint, velocity and FK workspace limits for path validation."""
        if not self._connected:
            raise RuntimeError("FR3 is disconnected")
        return self._limits

    @property
    def last_command(self):
        target = self._last_command
        return None if target is None else target.copy()

    def gripper_command_idle(self):
        with self._operation():
            return not self._hand.snapshot()["busy"]

    def status(self):
        return {"connected": self._connected, "actuation_enabled": self.config.actuation_enabled,
                "stopped": self._stop_requested.is_set(), "fault": self._fault,
                "submitted_commands": self._submitted, "device": deepcopy(self._device),
                "last_feedback": deepcopy(self._feedback),
                "last_applied_action": None if self._last_command is None else self._last_command.tolist()}

    def stop(self):
        self.close()

    def close(self):
        # Prevent an in-flight call from starting the next sub-operation while
        # the watchdog waits for its SDK I/O. Cleanup is retryable on failure.
        self._stop_requested.set()
        with self._lock:
            self._connected = False
            try:
                if self._hardware is not None:
                    self._hardware.stop_control()
                    self._started = False
            finally:
                if self._hand is not None:
                    self._hand.close()
                    self._hand = None
            self._hardware = self._limits = None
