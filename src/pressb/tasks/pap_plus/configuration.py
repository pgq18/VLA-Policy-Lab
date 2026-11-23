"""PAP Plus configuration and sampling, without a second simulation backend."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from pressb.tasks.pap.backend import sample_reset as pap_sample_reset
from pressb.tasks.pap.parallel import load_config as load_pap_config

DEFAULT_SCENE = 'configs/piper_pick_place_plus.json'


def prepare_config(config):
    """Prepare PAP validation while keeping sampling boundaries off the stage.

    PAP's loader eagerly validates marker sampling using visible tape geometry.
    This variant stores its identical boundary lines under a sampling-only key.
    Use fixed-position validation temporarily, then validate the requested
    sampling against those non-rendered lines in ``finalize_config``.
    """
    result = deepcopy(config)
    result.setdefault('scene_config', DEFAULT_SCENE)
    result.setdefault('log_dir', 'outputs/pap_plus_workers')
    requested = deepcopy(result.get('randomization', {}))
    if not isinstance(requested, dict):
        raise ValueError('PAP Plus randomization must be a mapping')
    requested.setdefault('position', 'markers')
    if requested['position'] not in ('fixed', 'markers', 'polygon'):
        raise ValueError('PAP Plus randomization.position must be fixed, markers, or polygon')
    result['randomization'] = {**requested, 'position': 'fixed'}
    return result, requested


def sampling_config(config):
    """A private sampling view; these lines are never passed to scene authoring."""
    result = dict(config)
    result['scene'] = dict(config['scene'])
    if config['randomization']['position'] == 'markers':
        edges = config['scene'].get('sampling_region_edges')
        if not isinstance(edges, list):
            raise ValueError('PAP Plus marker-region sampling requires scene.sampling_region_edges')
        result['scene']['tape_markers'] = deepcopy(edges)
    return result


def _validate_geometry(config):
    scene = config['scene']
    if scene.get('tape_markers') != []:
        raise ValueError('PAP Plus scene.tape_markers must be [] to keep the workspace unmarked')
    box = scene.get('box')
    if not isinstance(box, dict):
        raise ValueError('PAP Plus requires an open box')
    try:
        size = np.asarray(box['size'], dtype=float)
        center = np.asarray(box['center_xy'], dtype=float)
        thickness = box['wall_thickness']
        yaw = box.get('yaw_deg', 0.)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('PAP Plus box needs finite size, center, wall_thickness and yaw') from error
    if (size.shape != (3,) or center.shape != (2,) or not np.isfinite(size).all()
            or not np.isfinite(center).all() or np.any(size <= 0)
            or isinstance(thickness, bool) or not isinstance(thickness, (int, float))
            or not math.isfinite(thickness) or thickness <= 0
            or isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw)
            or np.any(size <= 2*thickness)):
        raise ValueError('PAP Plus box must have positive finite size and a nonempty cavity')


def finalize_config(config, requested):
    """Restore actual runtime sampling and validate it using PAP's sampler."""
    config['randomization'] = deepcopy(requested)
    _validate_geometry(config)
    sampling = sampling_config(config)
    for index in range(len(config['objects'])):
        pap_sample_reset(sampling, 0, index, {'object_index': index})
    return config


def load_config(config):
    prepared, requested = prepare_config(config)
    return finalize_config(load_pap_config(prepared), requested)


def sample_reset(config, seed, env_id, options):
    return pap_sample_reset(sampling_config(config), seed, env_id, options)
