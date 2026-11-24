"""Joint, workspace and stationarity checks for the physical FR3.

Adapted from the Franka station follower for PressB.
"""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np


def arm_vector(value: Any) -> np.ndarray:
    q = np.asarray(value, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("Expected seven finite Franka joint angles")
    return q


@dataclass
class SafetyLimits:
    model: Any
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    xyz_lower: np.ndarray
    xyz_upper: np.ndarray

    @classmethod
    def from_urdf(cls, urdf: str, model: Any, native: Any) -> "SafetyLimits":
        joints: dict[int, tuple[float, float]] = {}
        for joint in ET.fromstring(urdf).iter("joint"):  # nosec B314: trusted robot-supplied model
            match = re.search(r"(?:^|_)joint_?([1-7])$", joint.get("name", ""))
            limit = joint.find("limit")
            if match and joint.get("type") == "revolute" and limit is not None:
                index = int(match[1])
                if index in joints:
                    raise ValueError(f"Robot URDF has duplicate limits for Franka joint {index}")
                bounds = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
                if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                    raise ValueError(f"Invalid joint limits for J{index} in Franka URDF")
                joints[index] = bounds
        if set(joints) != set(range(1, 8)):
            raise ValueError("Robot URDF must identify the seven Franka joint limits")

        def native_vector(name: str, size: int) -> np.ndarray:
            value = np.asarray(getattr(native, name), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Expected {size} finite values in native Franka {name} limits")
            return value

        result = cls(
            model=model,
            lower=np.maximum([joints[i][0] for i in range(1, 8)], native_vector("lower", 7)),
            upper=np.minimum([joints[i][1] for i in range(1, 8)], native_vector("upper", 7)),
            velocity=native_vector("velocity", 7) - 0.5,
            xyz_lower=native_vector("xyz_lower", 3) + 0.05,
            xyz_upper=native_vector("xyz_upper", 3) - 0.05,
        )
        if np.any(result.lower >= result.upper):
            raise ValueError("Invalid joint limits from Franka URDF")
        if np.any(result.velocity <= 0) or np.any(result.xyz_lower >= result.xyz_upper):
            raise ValueError("Invalid Franka velocity or workspace limits after applying margins")
        return result

    def check(self, value: Any, label: str = "target") -> np.ndarray:
        q = arm_vector(value)
        invalid = np.where((q < self.lower) | (q > self.upper))[0]
        if len(invalid):
            i = int(invalid[0])
            raise ValueError(
                f"Unsafe {label}: J{i + 1}={q[i]:.4f} outside [{self.lower[i]:.4f}, {self.upper[i]:.4f}] rad"
            )
        xyz = np.asarray(self.model.position(q.tolist()), dtype=float)
        if (
            xyz.shape != (3,)
            or not np.isfinite(xyz).all()
            or np.any(xyz < self.xyz_lower)
            or np.any(xyz > self.xyz_upper)
        ):
            raise ValueError(f"Unsafe {label}: flange position {xyz.tolist()} outside allowed workspace")
        return xyz


def checked_state(backend: Any, stationary: bool = False) -> dict[str, Any]:
    state = backend.state()
    if state.get("errors"):
        raise RuntimeError(f"Franka reports errors: {state['errors']}")
    arm_vector(state["q"])
    dq = arm_vector(state["dq"])
    age = (time.monotonic_ns() - state["flange_timestamp_ns"]) / 1e9
    if not -0.1 <= age <= 0.5:
        raise RuntimeError("Franka state is stale")
    if stationary and np.max(np.abs(dq)) > 0.05:
        raise RuntimeError("Franka is already moving")
    return state
