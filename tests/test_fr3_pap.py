"""FR3 scene semantics without Kit: true bowl support and dataset coordinates."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.fr3_pap.configuration import ROOT, load_config, sample_objects
from pressb.tasks.fr3_pap.task import BowlOutcome

BOWL = dict(center_xy=[.45, -.1], top_diameter_m=.117, bottom_diameter_m=.064,
            height_m=.048, wall_thickness_m=.002, bottom_thickness_m=.003)


def frame(t=0., **changes):
    result = dict(seconds=t, position=[.45, -.1, .041], quaternion_wxyz=[1., 0., 0., 0.],
        velocity=np.zeros(6), gripper_width_m=.08,
        contacts=dict(finger_forces_n=[0., 0.], bowl_support_force_n=.3, table_force_n=0.))
    result.update(changes)
    return result


def test_shallow_bowl_support_can_succeed_with_cube_top_above_rim():
    checker = BowlOutcome(.05, 0., BOWL)
    outputs = [checker.update(**frame(t)) for t in [0., .1, .2, .3, .4, .5, .6]]
    assert outputs[-1]['success']
    assert outputs[-1]['top_world_m'] > BOWL['height_m']
    assert sum(row['success_event'] for row in outputs) == 1
    assert not outputs[0]['success']


@pytest.mark.parametrize('case', ['hover', 'held', 'closed', 'table_support', 'outside',
                                 'too_low', 'too_high', 'moving', 'spinning', 'rim'])
def test_bad_geometry_contacts_or_motion_cannot_be_success(case):
    c = BowlOutcome(.05, 0., BOWL, velocity_source="rigid_body")
    for t in [0., .3, .6, .9]:
        v = frame(t)
        if case == 'hover': v['contacts']['bowl_support_force_n'] = 0.
        if case == 'held': v['contacts']['finger_forces_n'] = [.1, .1]
        if case == 'closed': v['gripper_width_m'] = .049
        if case == 'table_support': v['contacts']['table_force_n'] = .2
        if case == 'outside': v['position'][0] += .06
        if case == 'too_low': v['position'][2] = .025
        if case == 'too_high': v['position'][2] = .075
        if case == 'moving': v['velocity'][0] = .1
        if case == 'spinning': v['velocity'][3] = 1.
        if case == 'rim': v['position'][0] += .03
        result = c.update(**v)
        assert not result['success'] and result['stable_seconds'] == 0.


def test_disturbance_restarts_continuous_dwell_and_only_one_reward():
    c = BowlOutcome(.05, 0., BOWL)
    c.update(**frame(0.));c.update(**frame(.3))
    bad = frame(.4);bad['contacts']['finger_forces_n'] = [.2, 0.]
    assert c.update(**bad)['stable_seconds'] == 0.
    assert not c.update(**frame(.5))['success']
    assert not c.update(**frame(.9))['success']
    assert c.update(**frame(1.01))['success_event']
    assert not c.update(**frame(1.1))['success_event']


def test_nonfinite_physics_and_nonmonotonic_time_fail():
    c = BowlOutcome(.05, 0., BOWL)
    with pytest.raises(ValueError): c.update(**frame(position=[np.nan, 0, 0]))
    c = BowlOutcome(.05, 0., BOWL);c.update(**frame(.1))
    with pytest.raises(ValueError): c.update(**frame(.1))


@pytest.fixture
def scene_file(tmp_path):
    scene = dict(robot_urdf=str(ROOT/'assets/robots/fr3/fr3_hand.urdf'),
        initial_state=[.218733, -.125317, .250366, -1.777210, .068360, 1.626233, -.471837, .0054596],
        gripper_max_width_m=.07998940348625183,
        table=dict(center_xy=[.45, 0.], size_m=[1.2, 1.2], top_z=0.),
        cube=dict(edge_m=.05, center_xy=[.45, .118], yaw_deg=0.), bowl=deepcopy(BOWL),
        cameras={name: dict(resolution=[640, 480]) for name in ['base', 'wrist']})
    path = tmp_path/'scene.json';path.write_text(json.dumps(scene))
    return path


def test_dataset_closure_and_control_frequency_are_preserved(scene_file):
    from pressb.tasks.fr3_pap.parallel import Fr3PapTask
    task = Fr3PapTask({'scene_config': str(scene_file)})
    meta = task.metadata()
    assert meta['action']['control_hz'] == 25.
    assert meta['action']['names'][-1] == 'gripper_closure'
    assert meta['action']['upper'][-1] == 1.
    assert meta['action']['units'] == ['rad']*7+['unitless']
    assert meta['observation']['cameras'] == ['base', 'wrist']
    initial = task.sample_reset(0, 123, {})['initial_state']
    assert initial[7] == pytest.approx(.0054596)
    assert task.execution['physics_dt'] == .01
    with pytest.raises(ValueError): task.sample_reset(0, 0, {'initial_state': [0.]*7+[80.]})


def test_seeded_cube_and_bowl_randomization_is_repeatable_and_separated(scene_file):
    cfg = load_config({'scene_config': str(scene_file), 'randomization': {'enabled': True, 'bowl_enabled': True,
        'x_range_m': [.30, .65], 'y_range_m': [-.25, .25], 'bowl_x_range_m': [.3, .65], 'bowl_y_range_m': [-.25, .25]}})
    a = [sample_objects(cfg, 0, seed) for seed in range(20)]
    assert a == [sample_objects(cfg, 0, seed) for seed in range(20)]
    assert len({tuple(row['cube']['center_xy']) for row in a}) > 1
    assert len({tuple(row['bowl']['center_xy']) for row in a}) > 1
    for row in a:
        delta = np.subtract(row['cube']['center_xy'], row['bowl']['center_xy'])
        assert np.linalg.norm(delta) >= .05/np.sqrt(2)+.117/2+.015


def test_explicit_layout_supports_moving_bowl_and_rejects_overlap(scene_file):
    cfg = load_config({'scene_config': str(scene_file)})
    desired = dict(cube=dict(center_xy=[.55, .15], yaw_deg=25.), bowl=dict(center_xy=[.4, -.2]))
    sampled = sample_objects(cfg, 0, 123, desired)
    assert sampled['bowl']['center_xy'] == [.4, -.2]
    assert sampled['cube']['yaw_deg'] == 25.
    with pytest.raises(ValueError): sample_objects(cfg, 0, 1, {'cube': {'center_xy': [.45, -.1]}})
    with pytest.raises(ValueError): sample_objects(cfg, 0, 1, {'bowl': {'center_xy': [2., 0.]}})


def test_center_ranges_allow_both_objects_at_the_requested_limits(scene_file):
    # Pin the centers at opposite edges. A second inset by the footprint
    # would incorrectly reject these valid positions.
    for cube_x, cube_y, bowl_x, bowl_y in [(.4,-.3,.7,.3), (.7,.3,.4,-.3)]:
        settings = dict(enabled=True, bowl_enabled=True, range_reference='center',
                        x_range_m=[cube_x]*2, y_range_m=[cube_y]*2,
                        bowl_x_range_m=[bowl_x]*2, bowl_y_range_m=[bowl_y]*2)
        cfg = load_config({'scene_config': str(scene_file), 'randomization': settings})
        layout = sample_objects(cfg, 0, 42)
        np.testing.assert_allclose(layout['cube']['center_xy'], [cube_x, cube_y])
        np.testing.assert_allclose(layout['bowl']['center_xy'], [bowl_x, bowl_y])
        # Older configurations without an explicit reference still fit the
        # entire footprint inside each range.
        settings.pop('range_reference')
        legacy = load_config({'scene_config': str(scene_file), 'randomization': settings})
        with pytest.raises(ValueError, match='cannot contain'):
            sample_objects(legacy, 0, 42)


def test_indexed_reset_updates_only_requested_slot_and_both_object_poses(scene_file):
    from pressb.tasks.fr3_pap.parallel import Fr3PapTask
    task = Fr3PapTask({'scene_config': str(scene_file), 'num_envs': 2})
    class Bodies:
        def __init__(self):
            self.positions = np.array([[.45, .118, .025], [.45, -.1, 0.], [3.45, .118, .025], [3.45, -.1, 0.]])
            self.quats = np.tile([1., 0., 0., 0.], (4, 1));self.vel = np.ones((4, 6));self.vel[[1, 3]] = 0.
        def get_world_poses(self): return self.positions.copy(), self.quats.copy()
        def set_world_poses(self, p, q, ids): self.positions[ids] = p;self.quats[ids] = q
        def set_velocities(self, v, ids):
            assert all(int(i) % 2 == 0 for i in ids), 'Cannot write a kinematic bowl velocity'
            self.vel[ids] = v
    task.bodies = Bodies();task.rows = np.array([[0, 1], [2, 3]])
    task.engine = SimpleNamespace(offsets=np.array([[0., 0., 0.], [3., 0., 0.]]))
    task.checkers = [None]*2;task.contacts = [None]*2;task.last_outcomes = [None]*2
    task.after_settle()
    peer = task.bodies.positions[:2].copy();peer_vel = task.bodies.vel[:2].copy()
    spec = task.sample_reset(1, 5, {'objects': {'cube': {'center_xy': [.6, .12]}, 'bowl': {'center_xy': [.45, -.15]}}})
    task.reset_task([1], {1: spec})
    np.testing.assert_array_equal(task.bodies.positions[:2], peer)
    np.testing.assert_array_equal(task.bodies.vel[:2], peer_vel)
    np.testing.assert_allclose(task.bodies.positions[2:, :2], [[3.6, .12], [3.45, -.15]])
    assert not np.any(task.bodies.vel[2:])
    assert task.checkers[0] is None


def test_distinct_fr3_camera_intrinsics_enable_grouped_rendering_by_default(scene_file):
    from pressb.tasks.fr3_pap.parallel import Fr3PapTask
    task = Fr3PapTask({'scene_config': str(scene_file)})
    assert task.config['camera_intrinsics_grouping'] is True
    assert task.execution['camera_intrinsics_grouping'] is True
    override = load_config({'scene_config': str(scene_file), 'camera_intrinsics_grouping': False})
    assert override['camera_intrinsics_grouping'] is False
    with pytest.raises(ValueError):
        load_config({'scene_config': str(scene_file), 'camera_intrinsics_grouping': 'true'})


def test_integrated_pose_stability_ignores_solver_bias_but_rejects_actual_motion():
    from scipy.spatial.transform import Rotation
    checker = BowlOutcome(.05, 0., BOWL)
    for t in np.arange(0, .62, .01):
        result = checker.update(**frame(float(t), velocity=[.01,0,0,0,0,.51]))
    assert result['success'] and result['angular_speed_rad_s'] == 0
    assert result['rigid_body_angular_speed_rad_s'] == .51
    for motion in ('linear', 'angular'):
        checker = BowlOutcome(.05, 0., BOWL)
        for i in range(80):
            value = frame(i*.01)
            if motion == 'linear':
                value['position'][0] += .001*(-1)**i
            else:
                q = Rotation.from_euler('z', i*.01).as_quat()
                value['quaternion_wxyz'] = q[[3,0,1,2]]
            result = checker.update(**value)
            assert not result['success']
        assert 'velocity' in result['stability_blockers']
    # q and -q represent the same orientation, including across resets.
    checker = BowlOutcome(.05, 0., BOWL)
    for i in range(60):
        result = checker.update(**frame(i*.01, quaternion_wxyz=[(-1)**i,0,0,0]))
    assert result['success']


def test_bowl_wall_contact_support_requires_close_measured_contact(scene_file):
    from pressb.tasks.fr3_pap.parallel import Fr3PapTask
    from pressb.runtime.physics import ContactForceReducer
    task = Fr3PapTask({'scene_config': str(scene_file)})
    task.engine = SimpleNamespace(backend=SimpleNamespace(ready=[True]))
    task.checkers = [object()]
    task.contact_paths = lambda paths: (True, True, False, False, False)
    task.contact_forces = ContactForceReducer(.01, 80., .003)
    for separation, expected in [(0., 1.), (.01, 0.)]:
        task.contacts = [task._empty_contacts()]
        task.on_contact(0, (), [SimpleNamespace(impulse=[.01,0,0], normal=[1.,0,0], separation=separation)])
        assert task.contacts[0]['bowl_contact_force_n'] == pytest.approx(expected)
        assert task.contacts[0]['bowl_support_force_n'] == 0
        checker = BowlOutcome(.05, 0., BOWL)
        for t in [0., .25, .5, .75]:
            result = checker.update(**frame(t, contacts=task.contacts[0]))
        assert result['success'] == bool(expected)
