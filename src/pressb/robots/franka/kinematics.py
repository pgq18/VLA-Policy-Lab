"""URDF-derived FR3 kinematics and the dataset's normalized gripper closure.

Arm coordinates are radians. The dataset closure is zero when open and one
when closed; it is not a finger displacement or a width in metres. The default
FK frame is the controller's flange, ``link8``. Use ``hand_tcp`` explicitly
for the standard Franka hand grasp frame (103.4 mm beyond the hand frame).
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def closure_to_width(closure, maximum_width=.08):
    value = np.asarray(closure, dtype=float)
    if not np.isfinite(maximum_width) or maximum_width <= 0:
        raise ValueError("Maximum gripper width must be finite and positive")
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError("Gripper closure must be within [0, 1]")
    return maximum_width * (1 - value)


def width_to_closure(width, maximum_width=.08):
    value = np.asarray(width, dtype=float)
    if not np.isfinite(maximum_width) or maximum_width <= 0:
        raise ValueError("Maximum gripper width must be finite and positive")
    if not np.isfinite(value).all() or np.any(value < -1e-5) or np.any(value > maximum_width + 1e-5):
        raise ValueError("Measured gripper width exceeds configured travel")
    return np.clip(1 - value / maximum_width, 0, 1)


class Fr3Kinematics:
    """Forward kinematics from the same seven joint origins as the controller."""

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path)
        robot = ET.parse(self.urdf_path).getroot()
        self.root_link = "base"
        self.links = {node.get("name") for node in robot.findall("link")}
        self.joints = []
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        by_name = {node.get("name"): node for node in robot.findall("joint")}
        if any(name not in by_name for name in self.joint_names):
            raise ValueError("Expected the FR3 controller's seven joint1…joint7 names")
        self.lower = np.array([float(by_name[name].find("limit").get("lower")) for name in self.joint_names])
        self.upper = np.array([float(by_name[name].find("limit").get("upper")) for name in self.joint_names])
        self.velocity_limits = np.array([float(by_name[name].find("limit").get("velocity")) for name in self.joint_names])
        for node in robot.findall("joint"):
            origin = node.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, :3] = Rotation.from_euler("xyz", self._vector(origin.get("rpy", "0 0 0"))).as_matrix()
                transform[:3, 3] = self._vector(origin.get("xyz", "0 0 0"))
            axis_node = node.find("axis")
            axis = self._vector(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0")
            if np.linalg.norm(axis) == 0:
                raise ValueError("Joint axis cannot be zero")
            self.joints.append((node.get("name"), node.get("type"), node.find("parent").get("link"),
                                node.find("child").get("link"), transform, axis / np.linalg.norm(axis)))

    @staticmethod
    def _vector(text):
        value = np.fromstring(text, sep=" ")
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Malformed URDF xyz/rpy/axis vector")
        return value

    def link_transforms(self, q, gripper_width=0.):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("FR3 arm state must contain seven finite radians")
        if not np.isfinite(gripper_width) or not 0 <= gripper_width <= .08 + 1e-6:
            raise ValueError("FR3 hand width must lie in [0, .08] metres")
        values = dict(zip(self.joint_names, q))
        values.update(finger_joint1=gripper_width/2, finger_joint2=gripper_width/2)
        result = {self.root_link: np.eye(4)}
        pending = list(self.joints)
        while pending:
            remaining = []
            for name, kind, parent, child, origin, axis in pending:
                if parent not in result:
                    remaining.append((name, kind, parent, child, origin, axis))
                    continue
                motion = np.eye(4)
                if kind == "revolute":
                    motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis * values[name]
                elif kind != "fixed":
                    raise ValueError("Unsupported FR3 joint kind: " + str(kind))
                result[child] = result[parent] @ origin @ motion
            if len(remaining) == len(pending):
                raise ValueError("Disconnected or cyclic FR3 URDF")
            pending = remaining
        return result

    def fk(self, q, *, link="link8", gripper_width=0.):
        """Return base-from-link, with column-vector transform convention."""
        transforms = self.link_transforms(q, gripper_width=gripper_width)
        if link not in transforms:
            raise ValueError("Unknown FR3 link: " + link)
        return transforms[link]
