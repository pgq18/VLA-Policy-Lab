"""PAP Plus changes target geometry/appearance, retaining shared task semantics."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.pap.parallel import PAPTask
from pressb.tasks.pap.task import PickPlaceOutcome
from pressb.tasks.pap_plus.configuration import load_config, sample_reset
from pressb.tasks.pap_plus.parallel import PAPPlusTask, create_backend

ROOT = Path(__file__).resolve().parents[1]


def scenes():
    return [json.loads((ROOT/'configs'/name).read_text()) for name in
            ('piper_pick_place.json', 'piper_pick_place_plus.json')]


def test_only_requested_geometry_and_visible_props_change():
    base, plus = scenes()
    for key in ('robot_usd', 'robot_urdf', 'robot_base_xy', 'table_height', 'table_depth',
                'table_width', 'table_x_min', 'up_camera', 'wrist_camera', 'overview_camera',
                'materials', 'material_colors', 'lighting', 'rendering', 'cube',
                'physics_dt', 'fps', 'gripper_max_width_m', 'task_instruction'):
        assert plus[key] == base[key], key
    np.testing.assert_array_equal(np.asarray(plus['box']['size']),
                                  np.asarray(base['box']['size'])*[.8, .8, 1.])
    np.testing.assert_allclose(plus['box']['center_xy'],
                               np.asarray(base['box']['center_xy'])+[.06, 0.], atol=1e-12, rtol=0)
    assert plus['difficulty_variant']['box_center_offset_xy_m'] == [.06, 0.]
    for key in ('yaw_deg', 'wall_thickness'):
        assert plus['box'][key] == base['box'][key]
    assert len(base['tape_markers']) == 4
    assert plus['tape_markers'] == []
    assert plus['sampling_region_edges'] == base['tape_markers']


@pytest.mark.parametrize('edge', [.02, .03, .04])
@pytest.mark.parametrize('yaw_range', [[0., 0.], [-45., 45.], [-180., 180.]])
def test_sampling_exactly_matches_pap_for_seeds_sizes_yaws_and_overrides(edge, yaw_range):
    kwargs = {'scene_overrides': {'cube': {'edge': edge, 'center_xy': [.32, .01],
                                          'mass': .025, 'yaw_deg': 0.}},
              'randomization': {'position': 'markers', 'colors': ['red', 'green', 'blue'],
                                'yaw_range_deg': yaw_range}}
    before = deepcopy(kwargs)
    base, plus = PAPTask(kwargs), PAPPlusTask(kwargs)
    for seed in (0, 1, 42, 999):
        for env_id in (0, 3):
            for options in ({}, {'yaw_deg': 37., 'color': 'yellow'}):
                assert plus.sample_reset(env_id, seed, options) == base.sample_reset(env_id, seed, options)
    assert kwargs == before
    assert plus.config['scene']['tape_markers'] == []


def test_metadata_and_final_sampling_identity_are_distinct_and_complete():
    base = PAPTask({'randomization': {'position': 'markers'}})
    plus = PAPPlusTask({})
    fixed = PAPPlusTask({'randomization': {'position': 'fixed'}})
    assert base.metadata()['task_id'] == 'pap'
    assert plus.metadata()['task_id'] == 'pap_plus'
    assert plus.metadata()['task_variant']['visible_tape_markers'] is False
    assert plus.config['randomization']['position'] == 'markers'
    assert plus.metadata()['observation'] == base.metadata()['observation']
    assert plus.metadata()['action'] == base.metadata()['action']
    assert plus.metadata()['reward_schema'] == base.metadata()['reward_schema']
    assert plus.identity['task_config_sha256'] != base.identity['task_config_sha256']
    assert plus.identity['task_config_sha256'] != fixed.identity['task_config_sha256']
    assert plus.identity['source_revision'] != base.identity['source_revision']
    assert 'tasks/pap_plus/configuration.py' in plus.identity['source_identity']
    assert plus.execution['reset_contract'] == base.execution['reset_contract']
    np.testing.assert_array_equal(plus.initial, base.initial)


def test_custom_scene_and_overrides_are_respected(tmp_path):
    _, plus = scenes()
    plus['box']['size'] = [.12, .095, .063]
    plus['wrist_camera']['fx'] += 12.
    path = tmp_path/'custom.json'
    path.write_text(json.dumps(plus))
    config = {'scene_config': str(path),
              'scene_overrides': {'box': {**plus['box'], 'center_xy': [.34, .21]}}}
    resolved = load_config(config)
    assert resolved['scene']['box']['size'] == [.12, .095, .063]
    assert resolved['scene']['box']['center_xy'] == [.34, .21]
    assert resolved['scene']['wrist_camera']['fx'] == plus['wrist_camera']['fx']
    assert resolved['scene']['tape_markers'] == []
    assert PAPPlusTask(config).config['randomization'] == resolved['randomization']


@pytest.mark.parametrize('overrides,match', [
    ({'tape_markers': [{'xy': [0., 0.], 'yaw_deg': 0}]}, 'tape_markers'),
    ({'box': {'size': [.008, .1, .05], 'center_xy': [.3, .2], 'wall_thickness': .005}}, 'cavity'),
    ({'box': {'size': [.1, .1, .05], 'center_xy': [.3, .2], 'wall_thickness': -1.}}, 'cavity'),
    ({'sampling_region_edges': []}, 'four finite tabletop marker'),
])
def test_invalid_geometry_is_rejected_before_simulation(overrides, match):
    with pytest.raises(ValueError, match=match):
        PAPPlusTask({'scene_overrides': overrides})


def test_bad_randomization_is_not_hidden_by_loader_adapter():
    with pytest.raises(ValueError, match='position'):
        load_config({'randomization': {'position': 'invalid'}})
    with pytest.raises(ValueError, match='Unsupported PAP randomization key'):
        load_config({'randomization': {'typo': True}})
    with pytest.raises(ValueError, match='yaw_range_deg'):
        load_config({'randomization': {'yaw_range_deg': [3, -3]}})


def test_fixed_and_explicit_polygon_do_not_require_sampling_edges():
    fixed = load_config({'randomization': {'position': 'fixed'},
                         'scene_overrides': {'sampling_region_edges': None}})
    assert sample_reset(fixed, 0, 0, {})['center_xy'] == fixed['objects'][0]['center_xy']
    polygon = [[.3, -.02], [.34, -.02], [.34, .02], [.3, .02]]
    config = load_config({'randomization': {'position': 'polygon', 'center_polygon': polygon},
                         'scene_overrides': {'sampling_region_edges': None}})
    sampled = sample_reset(config, 42, 0, {})
    assert .3 <= sampled['center_xy'][0] <= .34
    assert -.02 <= sampled['center_xy'][1] <= .02
    assert config['scene']['tape_markers'] == []


def test_factory_uses_common_parallel_runtime(monkeypatch):
    import pressb.runtime.parallel as shared
    monkeypatch.setattr(shared, 'SharedWorldBackend', lambda task: ('shared', task))
    kind, task = create_backend({})
    assert kind == 'shared' and isinstance(task, PAPPlusTask)
    assert task.metadata()['task_id'] == 'pap_plus'


def test_scene_authoring_receives_no_visible_or_sampling_tape(monkeypatch):
    import pressb.tasks.pap.worker as worker
    seen = {}
    def build(config, *, initialize):
        seen.update(deepcopy(config['scene']))
        assert initialize is False
        return SimpleNamespace(world='world', stage='stage', camera_paths=['up', 'wrist'],
            scene={'box_inside': {}}, base=[0, 0, .76], object_paths=['object'], material_paths=['material'])
    monkeypatch.setattr(worker, '_build', build)
    task = PAPPlusTask({})
    task.sample_reset(0, 42, {})
    task.build_template()
    assert seen['tape_markers'] == []
    assert len(seen['sampling_region_edges']) == 4


def _cavity(scene):
    b, h = scene['box'], scene['table_height']
    return {'center_world': [*np.add(scene['robot_base_xy'], b['center_xy']), h],
            'yaw_deg': b['yaw_deg'],
            'inner_size_xy': (np.asarray(b['size'][:2])-2*b['wall_thickness']).tolist(),
            'floor_z_world': h+b['wall_thickness'], 'rim_z_world': h+b['size'][2]}


def test_success_requires_containment_in_smaller_physical_cavity():
    base, plus = scenes()
    angle = np.deg2rad(base['box']['yaw_deg'])
    center = np.asarray(_cavity(base)['center_world'])
    rest = center + [-.04*np.cos(angle), -.04*np.sin(angle), .005+.015]
    quaternion = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
    checks = [PickPlaceOutcome(.03, scene['table_height'], _cavity(scene)) for scene in (base, plus)]
    outcomes = []
    for checker in checks:
        checker.update(0., [*rest[:2], .85], quaternion, np.zeros(6), .025, finger_contact=True)
        checker.update(1., rest, quaternion, np.zeros(6), .08)
        outcomes.append(checker.update(1.6, rest, quaternion, np.zeros(6), .08))
    assert outcomes[0]['inside_box'] and outcomes[0]['success']
    assert not outcomes[1]['inside_box'] and not outcomes[1]['success']
    # The shifted box centre remains a successful target.
    center = np.asarray(_cavity(plus)['center_world'])
    center[2] = rest[2]
    checker = PickPlaceOutcome(.03, plus['table_height'], _cavity(plus))
    checker.update(0., [*center[:2], .85], quaternion, np.zeros(6), .025, finger_contact=True)
    checker.update(1., center, quaternion, np.zeros(6), .08)
    assert checker.update(1.6, center, quaternion, np.zeros(6), .08)['success']
