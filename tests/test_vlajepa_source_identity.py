"""Checkpoint source revisions across the pose9 training-contract validator update."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from pressb.model_sources import builtin_source_root
from pressb.models import vlajepa, vlajepa_joint
from pressb.models.vlajepa_source_identity import inference_source_revision
from pressb.runtime.contract import fingerprint


BUNDLE = builtin_source_root("vlajepa")
# Source snapshots immediately after and before the validator update.
# Subsequent edits must retain their own live fingerprint.
LEGACY_REVISION = {
    "73e217e92045fdbb23dc73f7c93d4960a35bf71654e32afab98e5bb8c7a31151":
        "419c92b98ab7a11c416288ff0c23c9498e8eff8b89d07a2d6ffb79fadfb2f6d8",
    "7efaac5cda0b9f8f995e7e2c35bb08caa9bb59229f7bbd495cff9ef0550885f5":
        "a859d25118bca67bbd84d16cb27ab1619a49da002ab5012eddd624fdda74fdfa",
}
LEGACY_ADAPTER = {
    "d6b0a1108612722cb4980b2e5216bc234a5daff4fefc11a5e78ea4e7b894fc5a":
        "0c64fbb1d71134310f3cc13d69dc9e66c832b6b1a3f0d5cabeae886ed983e68d",
    "1c896490d454be83b1b6eec0ef9e297f9eca17dc00ac021a84c3b4c77ec58b25":
        "49cb3b8156eb73e2e87be0e9a26101cd3991aa174688dd36d3363ca68fbc6238",
}


def _source_record(kind, root):
    if kind == "joint":
        return vlajepa_joint._inference_sources(root)
    from pressb.models.vlajepa_backend import _source_identity

    identity = _source_identity(root)
    identity.pop("vla_repo")
    return {"model_source": identity, "adapter": hashlib.sha256(Path(vlajepa.__file__).read_bytes()).hexdigest()}


@pytest.mark.parametrize("kind", ["pose9", "joint"])
def test_legacy_checkpoint_with_exact_bundled_sources_keeps_revision(kind):
    sources = _source_record(kind, BUNDLE)
    live = fingerprint(sources)
    legacy = LEGACY_REVISION.get(live, live)
    assert inference_source_revision(sources, {}) == legacy
    for config in (None, {"image_transforms": {}}):
        assert inference_source_revision(sources, config) == (legacy if kind == "joint" else live)


@pytest.mark.parametrize("kind", ["pose9", "joint"])
def test_change_to_any_hashed_input_invalidates_bundled_revision(kind):
    def change_one_field(record):
        for key, value in record.items():
            if isinstance(value, dict):
                for changed in change_one_field(value):
                    yield {**record, key: changed}
            else:
                yield {**record, key: "a" * 64}

    for altered in change_one_field(_source_record(kind, BUNDLE)):
        revision = inference_source_revision(altered, {})
        assert revision == fingerprint(altered)
        assert revision not in LEGACY_REVISION.values()


@pytest.mark.parametrize("kind", ["pose9", "joint"])
def test_external_checkout_preserves_its_own_source_identity(tmp_path, kind):
    external = tmp_path / "external"
    shutil.copytree(BUNDLE / "starVLA", external / "starVLA", ignore=shutil.ignore_patterns("__pycache__"))
    sources = _source_record(kind, external)
    adapter = LEGACY_ADAPTER.get(sources["adapter"], sources["adapter"])
    expected = fingerprint({**sources, "adapter": adapter})
    assert inference_source_revision(sources, {}) == expected
    assert expected not in LEGACY_REVISION.values()
    loader = external / "starVLA/inference/piper_policy.py"
    with loader.open("a") as stream:
        stream.write("\n# Locally modified checkout.\n")
    altered = _source_record(kind, external)
    assert inference_source_revision(altered, {}) == fingerprint({**altered, "adapter": adapter})
    assert inference_source_revision(altered, {}) != expected


def test_pose_adapter_reports_actual_source_files_and_preserves_model_identity():
    from pressb.models.vlajepa_backend import _source_identity

    identity = _source_identity(BUNDLE)
    health = dict(checkpoint_sha256="1" * 64, checkpoint_step=1, frozen=True,
                  controller_gripper_width_m=.008, resolution=224, source_identity=identity)
    frozen = SimpleNamespace(action_dim=9, action_horizon=7, feature_dim=3,
                             policy=SimpleNamespace(config={}), health=lambda: health)
    first = vlajepa.VLAJEPABackend(frozen).describe()
    live = fingerprint(_source_record("pose9", BUNDLE))
    assert first["source_revision"] == LEGACY_REVISION.get(live, live)
    assert first["source_identity"] == identity
    health.update(checkpoint_sha256="2" * 64, resolution=448)
    second = vlajepa.VLAJEPABackend(frozen).describe()
    assert second["source_revision"] == first["source_revision"]
    assert second["model_id"] != first["model_id"]
    assert second["checkpoint_sha256"] != first["checkpoint_sha256"]
    assert second["model_config_sha256"] != first["model_config_sha256"]
    frozen.policy.config["image_transforms"] = {}
    assert vlajepa.VLAJEPABackend(frozen).describe()["source_revision"] == live
