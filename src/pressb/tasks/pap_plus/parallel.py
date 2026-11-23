"""PAP Plus task hooks on the unchanged PAP/common parallel execution path."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from pressb.tasks.pap.backend import _source_identity, _source_revision, task_identity
from pressb.tasks.pap.parallel import PAPTask, RESET_CONTRACT, renderer_profile
from .configuration import finalize_config, prepare_config, sample_reset

_SOURCE_FILES = ('tasks/pap_plus/__init__.py', 'tasks/pap_plus/configuration.py',
                 'tasks/pap_plus/parallel.py')


class PAPPlusTask(PAPTask):
    """Same measured pick/place task, with a smaller cavity and no visible tape."""

    def __init__(self, config):
        prepared, requested = prepare_config(config)
        super().__init__(prepared)
        self.config = cfg = finalize_config(self.config, requested)
        # Recompute from the final marker-free scene and actual sampling mode,
        # rather than retaining the parent's temporary fixed-reset identity.
        final_identity = task_identity(cfg)
        semantics = {'task_id': 'pap_plus', 'task': final_identity['task_config_sha256'],
            'reset': RESET_CONTRACT, 'render_resolution': cfg['render_resolution'],
            'spacing_m': cfg['spacing_m'], 'renderer_profile': cfg['renderer_profile'],
            'image_preprocessing': cfg['image_preprocessing'],
            'renderer_settings': renderer_profile(cfg['renderer_profile']),
            'observation_render_frames': cfg['observation_render_frames'],
            'render_subframes': cfg['render_subframes'], 'reset_render_frames': cfg['reset_render_frames'],
            'sampling_boundary_source': 'scene.sampling_region_edges'}
        final_identity['task_config_sha256'] = hashlib.sha256(
            json.dumps(semantics, sort_keys=True, allow_nan=False).encode()).hexdigest()
        final_identity['source_identity'].update(self.identity['source_identity'])
        final_identity['source_identity'].update(_source_identity(_SOURCE_FILES))
        final_identity['source_revision'] = _source_revision(final_identity['source_identity'])
        self.identity = final_identity

    def metadata(self):
        result = super().metadata()
        result['task_id'] = 'pap_plus'
        result['task_variant'] = {'base_task': 'pap', 'visible_tape_markers': False,
            'box_outer_size_m': deepcopy(self.config['scene']['box']['size']),
            'sampling_boundary_source': 'non-rendered scene.sampling_region_edges',
            'placement_sampling': deepcopy(self.config['randomization'])}
        return result

    def sample_reset(self, env_id, seed, options):
        return sample_reset(self.config, seed, env_id, options)


def create_backend(config):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(PAPPlusTask(config))
