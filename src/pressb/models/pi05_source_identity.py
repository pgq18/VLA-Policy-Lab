"""Preserve the previous RL identity for the verified, byte-equivalent π0.5 relocation."""

from pressb.model_sources import source_tree_sha256
from pressb.runtime.contract import fingerprint


# This is a one-time mapping for the upstream 0.4.5 policy and the adapter moved
# from the package-installed implementation to the bundled tree. Do not update
# these hashes on later model changes: those need their own source revision.
_LEGACY_ADAPTER = "fb781e5b0f5ac915d7d9f0b53a460e3fd640fb197de386ed58ebe651138d1352"
_BUNDLED_ADAPTER = "e7fd70658bc7513b805f1e736106988516fcb9f80754393971f50fa5216de8e9"
_BUNDLED_SOURCE = "1992425b114d2db5c737fb30de55752449b905df319c0596f840044ba13c53e1"


def inference_source_revision(lerobot_version, adapter_sha256):
    source = source_tree_sha256("pi05")
    if (lerobot_version, adapter_sha256, source) == ("0.4.5", _BUNDLED_ADAPTER, _BUNDLED_SOURCE):
        return fingerprint(dict(lerobot=lerobot_version, adapter=_LEGACY_ADAPTER))
    return fingerprint(dict(lerobot=lerobot_version, adapter=adapter_sha256, bundled_source=source))
