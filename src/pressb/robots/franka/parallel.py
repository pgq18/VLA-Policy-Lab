"""Seven-axis FR3 actuator adapter for PressB's shared parallel backend."""
from __future__ import annotations

import numpy as np

from pressb.runtime.parallel_contract import ParallelRobot
from pressb.runtime.physics import TensorArms
from .kinematics import Fr3Kinematics, closure_to_width


class Fr3JointDriver(ParallelRobot):
    """Absolute seven-joint radians plus normalized gripper closure (0=open)."""

    def __init__(self, engine, scene_config, initial_state, *, smoothing=None, arms=None):
        self.engine, self.scene_config = engine, dict(scene_config)
        self.num_envs, self.action_dim = engine.num_envs, 8
        self.kin = Fr3Kinematics(scene_config["robot_urdf"])
        self.maximum_width = float(scene_config.get("gripper_max_width_m", .08))
        if not 0 < self.maximum_width <= .08:
            raise ValueError("Standard Franka Hand travel must not exceed 80 mm")
        self.lower, self.upper = np.r_[self.kin.lower, 0.], np.r_[self.kin.upper, 1.]
        smoothing = dict(smoothing or {"mode": "none"})
        if set(smoothing) - {"mode", "alpha"}:
            raise ValueError("FR3 joint smoothing supports mode and alpha only")
        self.mode, self.alpha = smoothing.get("mode", "none"), float(smoothing.get("alpha", .8))
        if self.mode not in ("none", "joint_ema") or not np.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("Expected none/joint_ema with alpha in (0,1]")
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape == (8,):
            initial = np.repeat(initial[None], self.num_envs, axis=0)
        if initial.shape != (self.num_envs, 8):
            raise ValueError("FR3 initial state must be [8] or [num_envs,8]")
        self._validate_states(initial)
        self.initial_states = initial.copy()
        self.arms = arms if arms is not None else TensorArms(engine.simulation_view, engine.robot_paths)
        paths = [str(value) for value in self.arms.prim_paths]
        rows = []
        for root in engine.robot_paths:
            matches = [i for i, value in enumerate(paths) if value == root or value.startswith(str(root).rstrip('/')+'/')]
            if len(matches) != 1:
                raise RuntimeError("Expected one FR3 articulation under " + str(root))
            rows.append(matches[0])
        if len(set(rows)) != self.num_envs or len(paths) != self.num_envs:
            raise RuntimeError("FR3 tensor view contains unrelated or duplicate articulations")
        self.arm_rows = np.asarray(rows, dtype=np.int64)
        self.dofs = np.array([self.arms.get_dof_index(name) for name in
                             [*self.kin.joint_names, "finger_joint1", "finger_joint2"]])
        self.commands = self.arms.get_joint_positions()[self.arm_rows].copy()
        self.commands[:, self.dofs] = self._full(initial)
        self._set_gains()
        self.arms.set_joint_positions(self.commands, self.arm_rows)
        self.arms.set_joint_velocities(np.zeros_like(self.commands), self.arm_rows)
        self.arms.set_joint_position_targets(self.commands, self.arm_rows)
        self.last_sent = initial.copy()
        self.settled_joints = None

    def _full(self, states):
        width = closure_to_width(states[:, 7], self.maximum_width)
        return np.c_[states[:, :7], width/2, width/2]

    def _ids(self, env_ids):
        ids = np.asarray(env_ids)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids)
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= self.num_envs)):
            raise ValueError("Expected distinct existing FR3 environment IDs")
        return ids.astype(np.int64, copy=False)

    def _validate_states(self, states):
        if not np.isfinite(states).all() or np.any(states < self.lower-1e-6) or np.any(states > self.upper+1e-6):
            raise ValueError("FR3 joint/closure command exceeds bounds")
        if np.any(states[..., 7] < 0) or np.any(states[..., 7] > 1):
            raise ValueError("FR3 closure must be within [0,1]")

    def _set_gains(self):
        stiffness = np.zeros_like(self.arms.view.get_dof_stiffnesses())
        damping = np.zeros_like(self.arms.view.get_dof_dampings())
        stiffness[:, self.dofs[:7]] = self.scene_config.get("arm_stiffness", 6000.)
        damping[:, self.dofs[:7]] = self.scene_config.get("arm_damping", 100.)
        stiffness[:, self.dofs[7:]] = self.scene_config.get("gripper_stiffness", 1800.)
        damping[:, self.dofs[7:]] = self.scene_config.get("gripper_damping", 60.)
        if not np.isfinite(stiffness).all() or not np.isfinite(damping).all():
            raise ValueError("FR3 gains must be finite")
        rows = np.arange(self.num_envs, dtype=np.uint32)
        self.arms.view.set_dof_stiffnesses(stiffness, rows)
        self.arms.view.set_dof_dampings(damping, rows)

    def joint_arrays(self):
        q = self.arms.get_joint_positions()[self.arm_rows].copy()
        velocity = self.arms.get_joint_velocities()[self.arm_rows].copy()
        if not np.isfinite(q).all() or not np.isfinite(velocity).all():
            raise RuntimeError("Nonfinite measured FR3 articulation state")
        return q, velocity

    def states(self):
        q = self.joint_arrays()[0][:, self.dofs]
        # PhysX can overshoot the finger bounds slightly. The measured aperture
        # remains available in diagnostics; the policy observation has its
        # dataset-defined normalized closure range.
        return np.c_[q[:, :7], np.clip(1-(q[:, 7]+q[:, 8])/self.maximum_width, 0, 1)]

    def cache_initial_state(self):
        self.settled_joints = self.joint_arrays()[0]
        self.last_sent = self.states()

    def reset(self, env_ids, specs):
        ids = self._ids(env_ids)
        if self.settled_joints is None:
            raise RuntimeError("Cache settled FR3 state before reset")
        values = self.settled_joints[ids].copy()
        for row, i in enumerate(ids):
            initial = specs[int(i)].get("initial_state", specs[int(i)].get("initial_state_joint_rad_closure"))
            if initial is not None:
                initial = np.asarray(initial, dtype=float)
                if initial.shape != (8,):
                    raise ValueError("FR3 reset state must contain eight values")
                self._validate_states(initial)
                values[row, self.dofs] = self._full(initial[None])[0]
        self.arms.set_joint_positions(values, self.arm_rows[ids])
        self.arms.set_joint_velocities(np.zeros_like(values), self.arm_rows[ids])
        self.arms.set_joint_position_targets(values, self.arm_rows[ids])
        self.commands[ids] = values
        self.last_sent[ids] = self.states()[ids]

    def validate_actions(self, actions):
        values = np.asarray(actions, dtype=float)
        if values.ndim != 2 or values.shape[1] != 8 or not len(values):
            raise ValueError("FR3 actions must be nonempty [time,8]")
        self._validate_states(values)
        return values.copy()

    def begin_action(self, env_ids, actions):
        ids = self._ids(env_ids)
        values = np.asarray(actions, dtype=float)
        if values.shape != (len(ids), 8):
            raise ValueError("FR3 actions must be [selected_envs,8]")
        self._validate_states(values)
        sent = values.copy()
        if self.mode == "joint_ema":
            sent[:, :7] = self.alpha*sent[:, :7]+(1-self.alpha)*self.last_sent[ids, :7]
        self.last_sent[ids] = sent
        self.commands[np.ix_(ids, self.dofs)] = self._full(sent)
        return {int(i): sent[row].tolist() for row, i in enumerate(ids)}

    def before_tick(self, env_ids, substep, ticks_per_action):
        ids = self._ids(env_ids)
        if substep == 0:
            self.arms.set_joint_position_targets(self.commands[ids], self.arm_rows[ids])

    def observations(self, env_ids):
        ids, states = self._ids(env_ids), self.states()
        return {int(i): {"state": states[i].tolist(), "control_state": self.last_sent[i].tolist()} for i in ids}

    def diagnostic_state(self, env_ids):
        ids = self._ids(env_ids)
        q, velocity = self.joint_arrays()
        states = self.states()
        return {int(i): {"state": states[i].tolist(), "joints": q[i].tolist(),
                         "joint_velocity": velocity[i].tolist(), "command_joints": self.commands[i].tolist(),
                         "gripper_width_m": float(q[i, self.dofs[-2:]].sum()),
                         "control_state": self.last_sent[i].tolist()} for i in ids}
