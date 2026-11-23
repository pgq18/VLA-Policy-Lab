"""CPU-only FR3 bowl-task configuration and seeded layout sampling."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from pressb.control.joint_actions import JointActionPostprocessor
from pressb.runtime.contract import finite_array, fingerprint
from pressb.runtime.parallel import execution_config
from .task import INSTRUCTION, number, outcome_config

ROOT = Path(__file__).resolve().parents[4]
STATE_NAMES = [f'joint{i}' for i in range(1, 8)] + ['gripper_closure']
STATE_UNITS = ['rad'] * 7 + ['unitless']
EXECUTION_KEYS = ('num_envs', 'gpu', 'horizon', 'physics_dt', 'control_hz', 'max_episode_seconds',
    'settling_ticks', 'gamma', 'headless', 'spacing_m', 'render_resolution', 'renderer_profile',
    'render_subframes', 'observation_render_frames', 'reset_render_frames', 'cpu_threads',
    'png_workers', 'color_response_workers', 'log_dir', 'image_preprocessing', 'image_output_resolution',
    'physics_backend', 'gpu_dynamics', 'gpu_memory_capacities', 'renderer_settings', 'anti_aliasing',
    'camera_intrinsics_grouping')
DEFAULT_RANDOMIZATION = dict(enabled=False, x_range_m=[.30, .60], y_range_m=[-.25, .10],
    yaw_range_deg=[-180., 180.], bowl_enabled=False, bowl_x_range_m=[.35, .65],
    bowl_y_range_m=[-.25, .10], bowl_clearance_m=.015, max_attempts=1024,
    range_reference='footprint')
SOURCE_FILES = ('tasks/fr3_pap/__init__.py', 'tasks/fr3_pap/configuration.py', 'tasks/fr3_pap/task.py',
    'tasks/fr3_pap/parallel.py', 'tasks/fr3_pap/scene.py', 'robots/franka/parallel.py', 'robots/franka/kinematics.py',
    'robots/franka/isaac.py', 'robots/franka/appearance.py', 'sensors/tiled.py', 'sensors/camera.py',
    'sensors/color_response.py')


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def merge(base, updates):
    for key, value in updates.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            merge(base[key], value)
        else:
            base[key] = deepcopy(value)


def _range(value, name):
    a = finite_array(value, (2,), name, dtype=np.float64)
    if a[0] > a[1]:
        raise ValueError(f'{name} bounds must increase')
    return a.tolist()


def load_config(config):
    allowed = set(EXECUTION_KEYS) | {'scene_config', 'scene_overrides', 'smoothing', 'model_inputs',
        'randomization', 'outcome', 'reward', 'terminate_on_success'}
    if not isinstance(config, dict) or set(config)-allowed:
        raise ValueError(f'FR3 PAP config supports only {sorted(allowed)}')
    cfg = deepcopy(config)
    source = path(cfg.get('scene_config', 'configs/fr3_pap.json'))
    scene = json.loads(source.read_text())
    if not isinstance(scene, dict) or not isinstance(cfg.get('scene_overrides', {}), dict):
        raise ValueError('Scene and scene_overrides must be mappings')
    merge(scene, cfg.get('scene_overrides', {}))
    for key in ('robot_urdf', 'robot_usd'):
        if key in scene:
            scene[key] = str(path(scene[key]))
    scene['initial_state'] = finite_array(scene['initial_state'], (8,), 'initial_state', dtype=np.float64).tolist()
    if not 0 <= scene['initial_state'][7] <= 1:
        raise ValueError('Initial gripper_closure must be in [0,1]')
    scene['gripper_max_width_m'] = number(scene.get('gripper_max_width_m', .07998940348625183), 'gripper_max_width_m')
    for key in ('cube', 'bowl', 'table', 'cameras'):
        if not isinstance(scene.get(key), dict):
            raise ValueError(f'Scene requires {key} mapping')
    cube, bowl, table = scene['cube'], scene['bowl'], scene['table']
    cube['edge_m'] = number(cube.get('edge_m', .05), 'cube.edge_m')
    cube['center_xy'] = finite_array(cube['center_xy'], (2,), 'cube.center_xy', dtype=np.float64).tolist()
    cube['yaw_deg'] = float(finite_array([cube.get('yaw_deg', 0.)], (1,), 'cube.yaw_deg')[0])
    bowl['center_xy'] = finite_array(bowl['center_xy'], (2,), 'bowl.center_xy', dtype=np.float64).tolist()
    for key, default in [('top_diameter_m', .117), ('bottom_diameter_m', .064), ('height_m', .048),
                         ('wall_thickness_m', .002), ('bottom_thickness_m', .003)]:
        bowl[key] = number(bowl.get(key, default), 'bowl.'+key)
    if not (2*bowl['wall_thickness_m'] < bowl['bottom_diameter_m'] < bowl['top_diameter_m'] and
            bowl['bottom_thickness_m'] < bowl['height_m']):
        raise ValueError('Bowl must have an open widening inner cavity')
    table['center_xy'] = finite_array(table.get('center_xy', [.45, 0.]), (2,), 'table.center_xy', dtype=np.float64).tolist()
    table['size_m'] = finite_array(table.get('size_m', [1.2, 1.2]), (2,), 'table.size_m', dtype=np.float64).tolist()
    if min(table['size_m']) <= 0:
        raise ValueError('Table size must be positive')
    table['top_z'] = float(finite_array([table.get('top_z', 0.)], (1,), 'table.top_z')[0])
    for name in ('base', 'wrist'):
        cam = scene['cameras'].get(name)
        if not isinstance(cam, dict):
            raise ValueError('FR3 requires base and wrist camera configurations')
        size = cam.get('resolution', [640, 480])
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
            raise ValueError('Camera resolution must contain width, height')
        cam['resolution'] = size
    random = cfg.get('randomization', {})
    if not isinstance(random, dict) or set(random)-set(DEFAULT_RANDOMIZATION):
        raise ValueError('Unknown randomization settings')
    random = {**deepcopy(DEFAULT_RANDOMIZATION), **random}
    if random['range_reference'] not in ('center', 'footprint'):
        raise ValueError("randomization.range_reference must be center or footprint")
    if type(random['enabled']) is not bool or type(random['bowl_enabled']) is not bool:
        raise ValueError('randomization.enabled must be boolean')
    for key in ('x_range_m', 'y_range_m', 'yaw_range_deg', 'bowl_x_range_m', 'bowl_y_range_m'):
        random[key] = _range(random[key], key)
    random['bowl_clearance_m'] = number(random['bowl_clearance_m'], 'bowl_clearance_m', zero=True)
    if type(random['max_attempts']) is not int or random['max_attempts'] < 1:
        raise ValueError('max_attempts must be positive integer')
    smoothing = cfg.get('smoothing', {'mode': 'none', 'alpha': .8})
    if not isinstance(smoothing, dict) or set(smoothing)-{'mode', 'alpha'}:
        raise ValueError('smoothing supports mode and alpha')
    processor = JointActionPostprocessor(**smoothing)
    cfg['smoothing'] = dict(mode=processor.mode, alpha=processor.alpha)
    cfg.setdefault('model_inputs', {})
    if not isinstance(cfg['model_inputs'], dict):
        raise ValueError('model_inputs must be a mapping')
    reward = cfg.get('reward', {})
    if not isinstance(reward, dict) or set(reward)-{'success', 'failure'}:
        raise ValueError('reward supports success and failure')
    reward = {'success': 1., 'failure': 0., **reward}
    reward = {key: float(finite_array([value], (1,), 'reward.'+key)[0]) for key, value in reward.items()}
    cfg.setdefault('terminate_on_success', True)
    if type(cfg['terminate_on_success']) is not bool:
        raise ValueError('terminate_on_success must be boolean')
    execution = dict(num_envs=1, gpu=0, horizon=16, control_hz=25., physics_dt=1/100,
        settling_ticks=100, max_episode_seconds=40., gamma=.99, spacing_m=3.,
        render_resolution=scene['cameras']['base']['resolution'], renderer_profile='calibrated',
        camera_intrinsics_grouping=True, renderer_settings=deepcopy(scene.get('renderer_settings', {})),
        observation_render_frames=2, reset_render_frames=2, log_dir=str(ROOT/'outputs/fr3_pap/workers'))
    execution.update({key: cfg[key] for key in EXECUTION_KEYS if key in cfg})
    execution = execution_config(execution)
    for name in ('base', 'wrist'):
        camera_size = scene['cameras'][name]['resolution']
        render_size = execution['render_resolution']
        if camera_size[0]*render_size[1] != camera_size[1]*render_size[0]:
            raise ValueError('Render resolution must preserve each calibrated camera aspect ratio')
    cfg.update(execution, scene_config=str(source), scene=scene, randomization=random,
        outcome=outcome_config(cfg.get('outcome')), reward=reward)
    if cube['edge_m']+cfg['outcome']['release_margin_m'] > scene['gripper_max_width_m']:
        raise ValueError('Cube release width exceeds robot gripper maximum')
    validate_layout(cfg, {'center_xy': cube['center_xy'], 'yaw_deg': cube['yaw_deg']})
    return cfg


def validate_layout(config, cube, bowl=None):
    scene = config['scene']
    xy = finite_array(cube['center_xy'], (2,), 'cube center_xy', dtype=np.float64)
    yaw = float(finite_array([cube['yaw_deg']], (1,), 'cube yaw_deg')[0])
    edge = scene['cube']['edge_m']
    half = edge/2*(abs(math.sin(math.radians(yaw)))+abs(math.cos(math.radians(yaw))))
    table = scene['table']
    low = np.asarray(table['center_xy'])-np.asarray(table['size_m'])/2
    high = np.asarray(table['center_xy'])+np.asarray(table['size_m'])/2
    if np.any(xy-half < low) or np.any(xy+half > high):
        raise ValueError('Cube footprint must be fully on the table')
    bowl_xy = finite_array((bowl or scene['bowl'])['center_xy'], (2,), 'bowl center_xy', dtype=np.float64)
    radius = scene['bowl']['top_diameter_m']/2
    if np.any(bowl_xy-radius < low) or np.any(bowl_xy+radius > high):
        raise ValueError('Bowl footprint must be fully on the table')
    # Conservative square-vs-circle separation includes all yaw angles.
    if np.linalg.norm(xy-bowl_xy) < edge/math.sqrt(2)+radius+config['randomization']['bowl_clearance_m']:
        raise ValueError('Initial cube and bowl footprints overlap or lack configured clearance')


def sample_objects(config, env_id, seed, overrides=None):
    if type(seed) is not int or not 0 <= seed < 2**63 or type(env_id) is not int or env_id < 0:
        raise ValueError('Reset requires nonnegative env ID and integer seed in [0,2**63)')
    overrides = {} if overrides is None else deepcopy(overrides)
    if not isinstance(overrides, dict) or set(overrides)-{'cube', 'bowl'}:
        raise ValueError('Object overrides support cube and bowl')
    for name, values in overrides.items():
        if not isinstance(values, dict) or set(values)-{'center_xy', 'yaw_deg'}:
            raise ValueError('Object overrides support center_xy and yaw_deg only')
    random = config['randomization']; cube = config['scene']['cube']; bowl = config['scene']['bowl']
    rng = np.random.default_rng(np.random.SeedSequence([seed, env_id]))
    error = None
    for _ in range(random['max_attempts']):
        objects = {}
        for name, spec, enabled, keys in [('cube', cube, random['enabled'], ('x_range_m', 'y_range_m')),
                ('bowl', bowl, random['bowl_enabled'], ('bowl_x_range_m', 'bowl_y_range_m'))]:
            change = overrides.get(name, {})
            yaw = change.get('yaw_deg', float(rng.uniform(*random['yaw_range_deg']))
                if enabled and name == 'cube' and 'center_xy' not in change else spec.get('yaw_deg', 0.))
            yaw = float(finite_array([yaw], (1,), name+' yaw_deg')[0])
            half = (cube['edge_m']/2*(abs(math.sin(math.radians(yaw)))+abs(math.cos(math.radians(yaw))))
                    if name == 'cube' else bowl['top_diameter_m']/2)
            if 'center_xy' in change:
                xy = change['center_xy']
            elif enabled:
                inset = half if random['range_reference'] == 'footprint' else 0.
                ranges = [(random[key][0]+inset, random[key][1]-inset) for key in keys]
                if any(high < low for low, high in ranges):
                    raise ValueError('Sampling range cannot contain '+name)
                xy = [float(rng.uniform(low, high)) for low, high in ranges]
            else:
                xy = spec['center_xy']
            objects[name] = dict(center_xy=finite_array(xy, (2,), name+' center_xy', dtype=np.float64).tolist(), yaw_deg=yaw)
        try:
            validate_layout(config, objects['cube'], objects['bowl'])
            return objects
        except ValueError as exc:
            error = exc
            can_resample = any(enabled and 'center_xy' not in overrides.get(name, {}) for name, enabled in
                               [('cube', random['enabled']), ('bowl', random['bowl_enabled'])])
            if not can_resample:
                raise
    raise ValueError(f'Cannot sample cube/bowl layout: {error}')


def sample_cube(config, env_id, seed, overrides=None):
    return sample_objects(config, env_id, seed, {'cube': overrides or {}})['cube']


def task_identity(config):
    scene = deepcopy(config['scene'])
    assets = {}
    for key in ('robot_urdf', 'robot_usd'):
        if key in scene and Path(scene[key]).is_file():
            assets[key] = hashlib.sha256(Path(scene[key]).read_bytes()).hexdigest()
            scene[key] = {'sha256': assets[key]}
    sources = {}
    for name in SOURCE_FILES:
        p = ROOT/'src/pressb'/name
        if p.is_file():
            data = p.read_bytes(); sources[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    semantic = {key: deepcopy(config[key]) for key in ('scene', 'randomization', 'outcome', 'reward',
        'terminate_on_success', 'smoothing', 'model_inputs')}
    semantic['scene'] = scene
    semantic['instruction'] = INSTRUCTION
    semantic['outcome_version'] = 'fr3_cube_supported_bowl_v1'
    return dict(task_config_sha256=fingerprint(semantic), source_identity=sources,
        source_revision='sha256:'+fingerprint(sources), asset_sha256=assets)
