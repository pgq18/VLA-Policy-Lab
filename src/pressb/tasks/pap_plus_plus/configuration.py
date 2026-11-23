"""PAP++ scene defaults; validation and sampling are shared with PAP+."""
from __future__ import annotations

from copy import deepcopy

from pressb.tasks.pap_plus.configuration import load_config as load_plus_config
from pressb.tasks.pap_plus.configuration import sample_reset

DEFAULT_SCENE = 'configs/piper_pick_place_plus_plus.json'


def with_defaults(config):
    result = deepcopy(config)
    result.setdefault('scene_config', DEFAULT_SCENE)
    result.setdefault('log_dir', 'outputs/pap_plus_plus_workers')
    return result


def load_config(config):
    return load_plus_config(with_defaults(config))
