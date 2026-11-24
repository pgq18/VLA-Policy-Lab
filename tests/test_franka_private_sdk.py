# Adapted from the station LeRobot build and transport checks for PressB.
# SPDX-License-Identifier: Apache-2.0

"""Private SDK build guards, requiring neither a compiler nor a robot."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "franka_private_sdk", Path(__file__).resolve().parents[1] / "scripts/franka/private_sdk.py"
)
assert _SPEC is not None and _SPEC.loader is not None
sdk = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sdk)


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a tiny source tree with a separately reviewed fixture hash."""
    source = tmp_path / "upstream"
    (source / "src").mkdir(parents=True)
    (source / "build").mkdir()
    (source / ".git").mkdir()
    (source / "CMakeLists.txt").write_text("set(libfranka_VERSION 0.20.5)\n")
    network = (
        "// Minimal fixture; tests substitute its reviewed hash.\n"
        + sdk._FUNCTION_START
        + "uint32_t command_id) { return {}; }\n\n"
        "template <>\nvoid preserve_specialization();\n"
    )
    (source / "src/network.h").write_text(network)
    (source / "LICENSE").write_text("license retained\n")
    (source / "build/never_copy").write_text("existing shared build\n")
    (source / ".git/never_copy").write_text("git metadata\n")
    (source / "build/CMakeCache.txt").write_text(
        f"# SDK cache\nCMAKE_HOME_DIRECTORY:INTERNAL={source}\npinocchio_DIR:PATH=/test/existing/pinocchio\n"
    )
    monkeypatch.setattr(sdk, "UPSTREAM_NETWORK_SHA256", hashlib.sha256(network.encode()).hexdigest())
    return source


def test_unknown_upstream_is_rejected() -> None:
    """Unknown headers must fail before any build or patch is attempted."""
    with pytest.raises(ValueError, match="Unsupported libfranka network.h"):
        sdk.patch_network("An SDK version not reviewed for this patch")


def test_private_build_cannot_overlap_original(upstream: Path, tmp_path: Path) -> None:
    destination = upstream / "build/private"
    for invalid in (destination, tmp_path):
        with pytest.raises(ValueError, match="separate from the libfranka"):
            sdk.prepare_private_sdk(upstream / "build", invalid)
    assert not destination.exists()


def test_private_copy_preserves_original_and_reuses_unchanged_source(upstream: Path, tmp_path: Path) -> None:
    """Isolation and repeatability keep the existing SDK usable unchanged."""
    original = {
        path.relative_to(upstream): path.read_bytes() for path in upstream.rglob("*") if path.is_file()
    }
    destination = tmp_path / "private"
    result = sdk.prepare_private_sdk(upstream / "build", destination)
    assert result["config_dir"] == destination / "build/sdk"
    assert "-Dpinocchio_DIR=/test/existing/pinocchio" in result["dependency_options"]
    assert (destination / "source/LICENSE").read_text() == "license retained\n"
    assert not (destination / "source/build").exists()
    assert not (destination / "source/.git").exists()
    patched_file = destination / "source/src/network.h"
    patched = patched_file.read_text()
    assert patched != original[Path("src/network.h")].decode()
    assert patched.endswith("template <>\nvoid preserve_specialization();\n")
    before = patched_file.stat().st_mtime_ns
    assert sdk.prepare_private_sdk(upstream / "build", destination) == result
    assert patched_file.stat().st_mtime_ns == before
    assert original == {
        path.relative_to(upstream): path.read_bytes() for path in upstream.rglob("*") if path.is_file()
    }
    marker = json.loads((destination / "pressb_private_sdk.json").read_text())
    assert marker["network_patched_sha256"] == hashlib.sha256(patched.encode()).hexdigest()


def test_interrupted_first_copy_can_be_retried(upstream, tmp_path, monkeypatch) -> None:
    destination = tmp_path / "private"
    with monkeypatch.context() as patch:
        def interrupted(source, target):
            raise OSError("copy interrupted")

        patch.setattr(sdk.shutil, "copy", interrupted)
        with pytest.raises(OSError, match="copy interrupted"):
            sdk.prepare_private_sdk(upstream / "build", destination)
    assert not (destination / "source").exists()
    sdk.prepare_private_sdk(upstream / "build", destination)
    assert (destination / "source/src/network.h").is_file()


def test_unsupported_version_is_rejected_before_copy(upstream: Path, tmp_path: Path) -> None:
    """An unreviewed version must not create a private SDK directory."""
    (upstream / "CMakeLists.txt").write_text("set(libfranka_VERSION 0.21.0)\n")
    destination = tmp_path / "private"
    with pytest.raises(ValueError, match="only for libfranka 0.20.5"):
        sdk.prepare_private_sdk(upstream / "build", destination)
    assert not destination.exists()


def test_unmanaged_source_and_tampered_patch_are_not_overwritten(upstream: Path, tmp_path: Path) -> None:
    """The builder must not silently replace user files or local modifications."""
    destination = tmp_path / "private"
    unmanaged = destination / "source"
    unmanaged.mkdir(parents=True)
    user_file = unmanaged / "preserve.txt"
    user_file.write_text("User file")
    with pytest.raises(ValueError, match="unmanaged SDK"):
        sdk.prepare_private_sdk(upstream / "build", destination)
    assert user_file.read_text() == "User file"
    managed = tmp_path / "managed"
    sdk.prepare_private_sdk(upstream / "build", managed)
    patched = managed / "source/src/network.h"
    patched.write_text("changed outside builder")
    with pytest.raises(ValueError, match="changed outside"):
        sdk.prepare_private_sdk(upstream / "build", managed)
    assert patched.read_text() == "changed outside builder"
