from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pressb.robots.franka.kinematics import Fr3Kinematics, closure_to_width, width_to_closure
from pressb.robots.franka.parallel import Fr3JointDriver
from pressb.robots.franka.isaac import fixed_link_groups


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/robots/fr3/fr3_hand.urdf"
HOME = np.array([0., -.4, 0., -2., 0., 2., .78, 0.])


def test_fixed_flange_assembly_filter_excludes_mobile_links_and_fingers():
    groups = fixed_link_groups(ASSET)
    flange = next(group for group in groups if "hand" in group)
    assert set(flange) == {"link7", "link8", "hand", "hand_tcp"}
    assert all("leftfinger" not in group and "rightfinger" not in group for group in groups)
    for group in groups:
        assert len(set(group) & {f"link{i}" for i in range(1, 8)}) <= 1


def test_controller_arm_geometry_is_preserved():
    source = ET.parse(ASSET.parent/"source/controller_fr3.urdf").getroot()
    combined = ET.parse(ASSET).getroot()
    for index in range(1, 9):
        a, b = [node.find(f"joint[@name='joint{index}']") for node in (source, combined)]
        for tag in ("origin", "axis", "parent", "child", "limit"):
            av, bv = a.find(tag), b.find(tag)
            assert (None if av is None else av.attrib) == (None if bv is None else bv.attrib)


def test_flange_tcp_and_symmetric_finger_kinematics():
    kin = Fr3Kinematics(ASSET)
    controller = Fr3Kinematics(ASSET.parent/"source/controller_fr3.urdf")
    q = HOME[:7]
    np.testing.assert_allclose(kin.fk(q), controller.fk(q), atol=1e-12)
    frames = kin.link_transforms(q, gripper_width=.08)
    local = np.linalg.inv(frames["hand"]) @ frames["hand_tcp"]
    np.testing.assert_allclose(local[:3, 3], [0, 0, .1034], atol=1e-12)
    for name, sign in (("leftfinger", 1), ("rightfinger", -1)):
        finger = np.linalg.inv(frames["hand"]) @ frames[name]
        np.testing.assert_allclose(finger[:3, 3], [0, sign*.04, .0584], atol=1e-12)


def test_closure_convention_and_round_trip():
    values = np.array([0., .5, 1.])
    np.testing.assert_allclose(closure_to_width(values, .0799894035), [.0799894035, .03999470175, 0.])
    np.testing.assert_allclose(width_to_closure(closure_to_width(values)), values)
    with pytest.raises(ValueError):
        closure_to_width(-.01)


class FakeArms:
    def __init__(self):
        # Reverse PhysX rows and DOF order to check both mappings.
        self.prim_paths = ["/World/env_1/FR3/base", "/World/env_0/FR3/base"]
        self.names = list(reversed([*[f"joint{i}" for i in range(1, 8)], "finger_joint1", "finger_joint2"]))
        self.q, self.v, self.targets = [np.zeros((2, 9)) for _ in range(3)]
        self.kp, self.kd = np.zeros((2, 9)), np.zeros((2, 9))
        self.view = self

    def get_dof_index(self, name): return self.names.index(name)
    def get_joint_positions(self): return self.q.copy()
    def get_joint_velocities(self): return self.v.copy()
    def set_joint_positions(self, values, indices): self.q[indices] = values
    def set_joint_velocities(self, values, indices): self.v[indices] = values
    def set_joint_position_targets(self, values, indices): self.targets[indices] = values
    def get_dof_stiffnesses(self): return self.kp.copy()
    def get_dof_dampings(self): return self.kd.copy()
    def set_dof_stiffnesses(self, values, indices): self.kp[indices] = values
    def set_dof_dampings(self, values, indices): self.kd[indices] = values


def driver(scene_overrides=None, **kwargs):
    engine = SimpleNamespace(num_envs=2, robot_paths=["/World/env_0/FR3", "/World/env_1/FR3"])
    return Fr3JointDriver(engine, {"robot_urdf": str(ASSET), "gripper_max_width_m": .0799894035,
                                  **(scene_overrides or {})},
                          HOME, arms=FakeArms(), **kwargs)


def test_force_drive_calibration_defaults_and_explicit_override():
    default = driver()
    np.testing.assert_allclose(default.arms.kp[:, default.dofs[:7]], 6000.)
    np.testing.assert_allclose(default.arms.kd[:, default.dofs[:7]], 100.)
    np.testing.assert_allclose(default.arms.kp[:, default.dofs[7:]], 1800.)
    np.testing.assert_allclose(default.arms.kd[:, default.dofs[7:]], 60.)
    custom = driver(scene_overrides={"arm_stiffness": 5000., "arm_damping": 90.})
    np.testing.assert_allclose(custom.arms.kp[:, custom.dofs[:7]], 5000.)
    np.testing.assert_allclose(custom.arms.kd[:, custom.dofs[:7]], 90.)


def test_actuator_width_and_tensor_permutations():
    robot = driver()
    np.testing.assert_allclose(robot.states(), [HOME, HOME])
    robot.begin_action([1], [np.r_[HOME[:7], .5]])
    robot.before_tick([1], 0, 4)
    q = robot.arms.targets[robot.arm_rows[1], robot.dofs]
    np.testing.assert_allclose(q[-2:], [.0799894035/4]*2)
    np.testing.assert_allclose(robot.arms.targets[robot.arm_rows[0], robot.dofs][-2:], [.0799894035/2]*2)


def test_ema_smooths_seventh_joint_and_never_closure():
    robot = driver(smoothing={"mode": "joint_ema", "alpha": .5})
    action = HOME.copy(); action[6] += .2; action[7] = 1.
    robot.begin_action([0], [action])
    assert robot.last_sent[0, 6] == pytest.approx(HOME[6]+.1)
    assert robot.last_sent[0, 7] == 1.
    np.testing.assert_allclose(robot.last_sent[1], HOME)


def test_indexed_reset_preserves_peer_and_resets_ema():
    robot = driver(smoothing={"mode": "joint_ema", "alpha": .5})
    robot.cache_initial_state()
    target = HOME.copy(); target[6] += .2; target[7] = 1.
    robot.begin_action([0, 1], [target, target])
    peer = robot.commands[1].copy()
    robot.reset([0], {0: {"initial_state": HOME.tolist()}})
    np.testing.assert_array_equal(robot.commands[1], peer)
    np.testing.assert_allclose(robot.last_sent[0], HOME)
    robot.begin_action([0], [target])
    assert robot.last_sent[0, 6] == pytest.approx(HOME[6]+.1)
