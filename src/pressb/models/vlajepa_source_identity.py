"""Checkpoint compatibility for the pose9 training-contract validator update."""

from pressb.runtime.contract import fingerprint


# Keys include the live adapter, sampler and model sources. The pose9 validator
# change affects only saved configs with augmentation. The joint loader uses
# other functions from that same file. Keep these one-time pins fixed.
_BUNDLED_REVISIONS = {
    # Pose9
    "73e217e92045fdbb23dc73f7c93d4960a35bf71654e32afab98e5bb8c7a31151":
        "419c92b98ab7a11c416288ff0c23c9498e8eff8b89d07a2d6ffb79fadfb2f6d8",
    # Joint6 and joint7
    "7efaac5cda0b9f8f995e7e2c35bb08caa9bb59229f7bbd495cff9ef0550885f5":
        "a859d25118bca67bbd84d16cb27ab1619a49da002ab5012eddd624fdda74fdfa",
}

# External checkouts never execute the updated bundled validator. Preserve the
# previous adapter digest for the exact edits that only select this revision.
_EXTERNAL_ADAPTERS = {
    "d6b0a1108612722cb4980b2e5216bc234a5daff4fefc11a5e78ea4e7b894fc5a":
        "0c64fbb1d71134310f3cc13d69dc9e66c832b6b1a3f0d5cabeae886ed983e68d",
    "1c896490d454be83b1b6eec0ef9e297f9eca17dc00ac021a84c3b4c77ec58b25":
        "49cb3b8156eb73e2e87be0e9a26101cd3991aa174688dd36d3363ca68fbc6238",
}


def inference_source_revision(sources, config):
    joint_bundle = "bundled_model_source" in sources
    pose_bundle = "bundled_model_source_sha256" in sources.get("model_source", {})
    if joint_bundle or pose_bundle:
        revision = fingerprint(sources)
        if joint_bundle or (config is not None and "image_transforms" not in config):
            return _BUNDLED_REVISIONS.get(revision, revision)
        return revision
    adapter = _EXTERNAL_ADAPTERS.get(sources["adapter"], sources["adapter"])
    return fingerprint({**sources, "adapter": adapter})
