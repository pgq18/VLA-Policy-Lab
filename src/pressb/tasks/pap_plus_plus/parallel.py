"""PAP++ task identity on PAP's shared scene, outcomes and parallel backend."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from pressb.tasks.pap.backend import _source_identity, _source_revision
from pressb.tasks.pap_plus.parallel import PAPPlusTask
from .configuration import with_defaults

_SOURCE_FILES = ('tasks/pap_plus_plus/__init__.py',
                 'tasks/pap_plus_plus/configuration.py',
                 'tasks/pap_plus_plus/parallel.py')


class PAPPlusPlusTask(PAPPlusTask):
    """Smaller placement opening and less displacement than the PAP+ default."""

    def __init__(self, config):
        super().__init__(with_defaults(config))
        # The parent already includes resolved scene, sampling, renderer and
        # execution semantics. Add a distinct task identity without altering
        # PAP/PAP+ behavior or accepting their checkpoints as native resumes.
        semantics = {'task_id': 'pap_plus_plus',
                     'parent_config_sha256': self.identity['task_config_sha256']}
        self.identity['task_config_sha256'] = hashlib.sha256(
            json.dumps(semantics, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.identity['source_identity'].update(_source_identity(_SOURCE_FILES))
        self.identity['source_revision'] = _source_revision(self.identity['source_identity'])

    def metadata(self):
        result = super().metadata()
        result['task_id'] = 'pap_plus_plus'
        result['task_variant']['box_center_xy_m'] = deepcopy(self.config['scene']['box']['center_xy'])
        return result


def create_backend(config):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(PAPPlusPlusTask(config))
