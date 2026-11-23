"""FR3 task hooks; physics, rendering, reset scheduling and RPC stay shared."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from pressb.runtime.contract import finite_array
from pressb.runtime.parallel_contract import ParallelTask
from pressb.runtime.physics import ContactForceReducer, ContactPathClassifier
from .configuration import (EXECUTION_KEYS, STATE_NAMES, STATE_UNITS, load_config,
                            sample_objects, task_identity)
from .task import BowlOutcome, INSTRUCTION


class Fr3PapTask(ParallelTask):
    def __init__(self, config):
        self.config = cfg = load_config(config)
        self.execution = {key: deepcopy(cfg[key]) for key in EXECUTION_KEYS if key in cfg}
        self.execution.update(terminal_boundary='physics_tick', timeout_reason='timeout')
        from pressb.robots.franka.kinematics import Fr3Kinematics
        kin = Fr3Kinematics(cfg['scene']['robot_urdf'])
        self.lower = np.r_[kin.lower, 0.]
        self.upper = np.r_[kin.upper, 1.]
        initial = np.asarray(cfg['scene']['initial_state'])
        if np.any(initial < self.lower) or np.any(initial > self.upper):
            raise ValueError('FR3 initial state exceeds joint/closure bounds')
        self.identity = task_identity(cfg)

    def metadata(self):
        cfg = self.config
        return dict(schema_version=2, node_type='simulation', task_id='fr3_pap', num_envs=cfg['num_envs'],
            **deepcopy(self.identity),
            observation={'state': {'names': STATE_NAMES.copy(), 'units': STATE_UNITS.copy()},
                         'cameras': ['base', 'wrist']},
            action={'representation': 'joint_position', 'frame': 'robot_base', 'names': STATE_NAMES.copy(),
                'units': STATE_UNITS.copy(), 'lower': self.lower.tolist(), 'upper': self.upper.tolist(),
                'horizon': cfg['horizon'], 'control_hz': cfg['control_hz']},
            control_state_dim=8, task_features_dim=0, reset_mode='indexed',
            discount={'gamma': cfg['gamma'], 'timebase_hz': cfg['control_hz']},
            smoothing=deepcopy(cfg['smoothing']), task_instruction=INSTRUCTION,
            robot_model='Franka Research 3', gripper_coordinate='0=open, 1=closed',
            gripper_max_width_m=cfg['scene']['gripper_max_width_m'], reward=deepcopy(cfg['reward']),
            reward_schema={'kind': 'sparse_physical_event', 'success_reward': 'Once after released contact-supported bowl dwell',
                'within_chunk_discount': 'gamma ** ((event_tick - 1) / physics_ticks_per_action)', 'timeout_reward': 0.},
            success_rule='Cube is inside the tapered bowl, supported by the bowl rather than the table, released and low-speed for the configured dwell. Its top may protrude above the shallow rim.',
            outcome_thresholds=deepcopy(cfg['outcome']),
            outcome_measurements={'motion': 'Per-physics-tick integrated pose delta by default; solver velocity is logged separately',
                'bowl_support': 'Measured near-contact normal force inside cavity, including wall friction support; table contact excluded'},
            object_reset={'objects': ['cube', 'bowl'], 'options': ['center_xy', 'yaw_deg'],
                'frame': 'robot_base XY metres', 'randomization': deepcopy(cfg['randomization']),
                'semantics': 'Seeded cube and bowl placement with full tabletop support and separation; explicit XY may leave the default sampling rectangles.'})

    def build_template(self):
        from .scene import build_template
        template = build_template(deepcopy(self.config['scene']), deepcopy(self.execution))
        from pxr import PhysxSchema
        scene = template.data['scene']
        cube = template.stage.GetPrimAtPath(scene['cube_path'])
        PhysxSchema.PhysxContactReportAPI.Apply(cube).CreateThresholdAttr(0.)
        PhysxSchema.PhysxRigidBodyAPI.Apply(cube).CreateSleepThresholdAttr(0.)
        return template

    def create_robot(self, engine):
        from pressb.robots.franka.parallel import Fr3JointDriver
        return Fr3JointDriver(engine, self.config['scene'], self.config['scene']['initial_state'],
                              smoothing=self.config['smoothing'])

    def sample_reset(self, env_id, seed, options):
        allowed = {'task', 'model_inputs', 'objects', 'initial_state'}
        if not isinstance(options, dict) or set(options)-allowed:
            raise ValueError(f'FR3 reset supports {sorted(allowed)}')
        if options.get('task', INSTRUCTION) != INSTRUCTION:
            raise ValueError('FR3 PAP uses its fixed blue-cube-to-bowl instruction')
        objects = options.get('objects', {})
        if not isinstance(objects, dict) or set(objects)-{'cube', 'bowl'}:
            raise ValueError('Reset objects supports cube and bowl')
        initial = finite_array(options.get('initial_state', self.config['scene']['initial_state']), (8,),
                               'initial_state', dtype=np.float64)
        if np.any(initial < self.lower) or np.any(initial > self.upper):
            raise ValueError('FR3 reset state exceeds joint/closure bounds')
        model_inputs = deepcopy(self.config['model_inputs'])
        extra = options.get('model_inputs', {})
        if not isinstance(extra, dict):
            raise ValueError('model_inputs must be a mapping')
        model_inputs.update(deepcopy(extra))
        return dict(env_id=env_id, seed=seed, task=INSTRUCTION, model_inputs=model_inputs,
            initial_state=initial.tolist(), objects=sample_objects(self.config, env_id, seed, objects))

    def bind(self, engine):
        super().bind(engine)
        self.scene = engine.template.data['scene']
        self.bodies, self.rows = engine.rigid_bodies([self.scene['cube_path'], self.scene['bowl_path']])
        fingers = self.scene.get('finger_paths', [engine.template.robot_path+'/leftfinger', engine.template.robot_path+'/rightfinger'])
        self.contact_paths = ContactPathClassifier([self.scene['cube_path'], self.scene['bowl_path'],
            self.scene['table_path'], *fingers])
        cfg = self.config['outcome']
        self.contact_forces = ContactForceReducer(self.execution['physics_dt'],
            cfg['max_support_normal_tilt_deg'], cfg['contact_separation_tolerance_m'],
            'Invalid FR3 PAP contact')
        self.contacts = [self._empty_contacts() for _ in range(engine.num_envs)]
        self.checkers = [None] * engine.num_envs
        self.last_outcomes = [{} for _ in range(engine.num_envs)]

    @staticmethod
    def _empty_contacts():
        return dict(finger_forces_n=[0., 0.], bowl_support_force_n=0., bowl_contact_force_n=0., table_force_n=0.)

    def after_settle(self):
        positions, orientations = self.bodies.get_world_poses()
        self.initial_positions = positions[self.rows].copy()
        self.initial_orientations = orientations[self.rows].copy()

    def reset_task(self, env_ids, specs):
        positions, quaternions, indices = [], [], []
        info = {}
        scene = self.config['scene']
        for eid in env_ids:
            for j, name in enumerate(('cube', 'bowl')):
                requested = specs[eid]['objects'][name]
                p = self.initial_positions[eid, j].copy()
                p[:2] = np.asarray(requested['center_xy']) + self.engine.offsets[eid, :2]
                half = math.radians(requested['yaw_deg'])/2
                positions.append(p); quaternions.append([math.cos(half), 0., 0., math.sin(half)])
                indices.append(self.rows[eid, j])
            bowl = deepcopy(scene['bowl'])
            bowl['center_xy'] = (np.asarray(specs[eid]['objects']['bowl']['center_xy'])+self.engine.offsets[eid, :2]).tolist()
            self.checkers[eid] = BowlOutcome(scene['cube']['edge_m'],
                scene['table']['top_z']+float(self.engine.offsets[eid, 2]), bowl, **self.config['outcome'])
            self.contacts[eid] = self._empty_contacts(); self.last_outcomes[eid] = {}
            info[eid] = dict(**deepcopy(specs[eid]), task_outcome_available=True,
                            zero_tick_reset=True, object_teleport_after_reset=False)
        indices = np.asarray(indices, dtype=np.int64)
        self.bodies.set_world_poses(np.asarray(positions), np.asarray(quaternions), indices)
        # Kinematic bowl pose is set directly; PhysX forbids velocity writes
        # to kinematic bodies. Clear velocities only on the dynamic cubes.
        cube_indices = np.asarray([self.rows[eid, 0] for eid in env_ids], dtype=np.int64)
        self.bodies.set_velocities(np.zeros((len(cube_indices), 6)), cube_indices)
        return info

    def before_tick(self, env_ids):
        for eid in env_ids:
            self.contacts[eid] = self._empty_contacts()

    def on_contact(self, env_id, actor_paths, contacts):
        if not self.engine.backend.ready[env_id] or self.checkers[env_id] is None or not len(contacts):
            return
        cube, bowl, table, left, right = self.contact_paths(actor_paths)
        if not cube:
            return
        force, vertical = self.contact_forces(contacts)
        row = self.contacts[env_id]
        if bowl:
            row['bowl_support_force_n'] += vertical
            near = [c for c in contacts if abs(float(c.separation)) <= self.config['outcome']['contact_separation_tolerance_m']]
            row['bowl_contact_force_n'] += force if len(near) == len(contacts) else self.contact_forces(near)[0]
        if table:
            row['table_force_n'] += force
        for i, touched in enumerate((left, right)):
            if touched:
                row['finger_forces_n'][i] += force

    def after_tick(self, env_ids):
        positions, quats = self.bodies.get_world_poses(); velocities = self.bodies.get_velocities()
        if not all(np.isfinite(a).all() for a in (positions, quats, velocities)):
            raise RuntimeError('Nonfinite FR3 cube physics state')
        states = self.engine.robot.observations(env_ids)
        result = {}
        cfg = self.config
        for eid in env_ids:
            row = self.rows[eid, 0]
            closure = float(states[eid]['state'][7])
            outcome = self.checkers[eid].update(seconds=float(self.engine.backend.ticks[eid])*cfg['physics_dt'],
                position=positions[row], quaternion_wxyz=quats[row], velocity=velocities[row],
                gripper_width_m=(1.-np.clip(closure, 0., 1.))*cfg['scene']['gripper_max_width_m'],
                contacts=self.contacts[eid])
            self.last_outcomes[eid] = deepcopy(outcome)
            success, dropped = outcome['success'], outcome['dropped']
            terminal = bool(dropped or success and cfg['terminate_on_success'])
            result[eid] = dict(reward=cfg['reward']['failure'] if dropped else
                cfg['reward']['success'] if outcome['success_event'] else 0., success=success,
                terminated=terminal, truncated=False, reason='object_dropped' if dropped else 'success' if terminal else None,
                info=dict(physical_outcome=outcome, contacts=deepcopy(self.contacts[eid]),
                    task_outcome_available=True, object_teleport_after_reset=False))
        return result

    def diagnostic_state(self, env_ids):
        positions, quats = self.bodies.get_world_poses(); velocities = self.bodies.get_velocities()
        return {eid: {'objects': {name: {'position_world': positions[self.rows[eid, j]].tolist(),
            'orientation_wxyz': quats[self.rows[eid, j]].tolist(), 'velocity': velocities[self.rows[eid, j]].tolist()}
            for j, name in enumerate(('cube', 'bowl'))},
            'physical_outcome': deepcopy(self.last_outcomes[eid])} for eid in env_ids}



def create_backend(config, *, engine_factory=None):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(Fr3PapTask(config), engine_factory=engine_factory)
