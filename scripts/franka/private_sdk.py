# Adapted from the station LeRobot build and transport checks for PressB.
# SPDX-License-Identifier: Apache-2.0

"""Build an isolated libfranka with safe concurrent move/stop response handling.

Only the reviewed 0.20.5 network implementation is accepted. The original SDK
is never patched or rebuilt. The private library has a distinct SONAME so an
already-loaded system libfranka cannot silently replace this dependency.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

UPSTREAM_NETWORK_SHA256 = "5f38e76b0bf61d7dfde23270235ca314a282f75fbb31ff348fffdcfa37a9fa7e"
PATCH_VERSION = 2
_FUNCTION_START = "template <typename T>\ntypename T::Response Network::tcpBlockingReceiveResponse("

_PATCHED_FUNCTION = """template <typename T>
typename T::Response Network::tcpBlockingReceiveResponse(uint32_t command_id,
                                                         std::vector<uint8_t>* vl_buffer) {
  using namespace std::literals::chrono_literals;
  while (true) {
    std::unique_lock<std::mutex> lock(tcp_mutex_);
    tcpReadFromBuffer<T>(kTimeout);
    const auto received_response = received_responses_.find(command_id);
    if (received_response != received_responses_.end()) {
      // LeRobot local fix: a concurrent move/stop receiver can insert or erase
      // map entries. Keep lookup, validation, copying and erase under one lock.
      using Message = typename T::template Message<typename T::Response>;
      const auto& bytes = received_response->second;
      if (bytes.size() < sizeof(Message)) {
        throw ProtocolException("libfranka: Incorrect TCP message size.");
      }
      const auto message = *reinterpret_cast<const Message*>(bytes.data());
      if (message.header.size < sizeof(Message) || message.header.size != bytes.size()) {
        throw ProtocolException("libfranka: Incorrect TCP message size.");
      }
      if (vl_buffer != nullptr && message.header.size != sizeof(Message)) {
        *vl_buffer = std::vector<uint8_t>(bytes.begin() + sizeof(Message), bytes.end());
      }
      const auto response = message.getInstance();
      received_responses_.erase(received_response);
      return response;
    }
    lock.unlock();
    // yield() alone can immediately retake this mutex and starve the thread
    // sending stop while a move awaits its response. Sleep only off the lock.
    std::this_thread::sleep_for(1ms);
  }
}

"""


def patch_network(source: str) -> str:
    """Fail closed instead of applying an old textual patch to another SDK."""
    if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_NETWORK_SHA256:
        raise ValueError("Unsupported libfranka network.h; review the concurrent stop patch for this SDK")
    start = source.index(_FUNCTION_START)
    end = source.index("template <>\n", start)
    return source[:start] + _PATCHED_FUNCTION + source[end:]


def _cache_values(directory: Path) -> dict[str, str]:
    cache = directory / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    return dict(re.findall(r"^([A-Za-z_][A-Za-z_0-9]*):[^=\n]+=(.*)$", cache.read_text(), re.MULTILINE))


def _source_files(source: Path, upstream_build: Path) -> Iterator[tuple[Path, Path]]:
    for root, directories, names in os.walk(source):
        directory = Path(root)
        directories[:] = sorted(
            name for name in directories
            if name not in {".git", "build", "__pycache__", ".venv"}
            and directory / name != upstream_build
        )
        for name in sorted(names):
            # Submodule worktrees can represent .git as a file.
            item = directory / name
            if name != ".git" and item.is_file():
                yield item, item.relative_to(source)


def prepare_private_sdk(upstream_build: Path, destination: Path) -> dict:
    """Copy and patch source, returning configuration without starting a build."""
    cache = _cache_values(upstream_build)
    if not cache.get("CMAKE_HOME_DIRECTORY"):
        raise ValueError(f"--libfranka-dir must contain the SDK's CMakeCache.txt: {upstream_build}")
    source = Path(cache["CMAKE_HOME_DIRECTORY"]).resolve()
    destination = destination.resolve()
    for original in (source, upstream_build.resolve()):
        if destination.is_relative_to(original) or original.is_relative_to(destination):
            raise ValueError("Keep the private build separate from the libfranka source and build trees")
    cmake = (source / "CMakeLists.txt").read_text()
    if re.search(r"set\(libfranka_VERSION\s+0\.20\.5\)", cmake) is None:
        raise ValueError("Private Franka SDK patch is reviewed only for libfranka 0.20.5")
    network = (source / "src/network.h").read_text()
    patched = patch_network(network)
    files = list(_source_files(source, upstream_build.resolve()))
    digest = hashlib.sha256()
    for item, relative in files:
        digest.update(str(relative).encode() + b"\0" + item.read_bytes())
    identity = {
        "source": str(source),
        "source_sha256": digest.hexdigest(),
        "network_upstream_sha256": UPSTREAM_NETWORK_SHA256,
        "network_patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "patch_version": PATCH_VERSION,
        "libfranka_version": "0.20.5",
    }
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "pressb_private_sdk.json"
    copied_source = destination / "source"
    if copied_source.exists() and not marker.is_file():
        raise ValueError(f"Refusing to replace unmanaged SDK source directory: {copied_source}")
    previous = json.loads(marker.read_text()) if marker.is_file() else None
    if previous != identity or not copied_source.exists():
        # Collect the entire source before replacing a previously working copy.
        # Fresh mtimes make CMake rebuild modified code even after an upstream
        # checkout with timestamps older than objects in our private build.
        with tempfile.TemporaryDirectory(dir=destination) as directory:
            staged = Path(directory) / "source"
            for item, relative in files:
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(item, target)
            (staged / "src/network.h").write_text(patched)
            staged_marker = Path(directory) / marker.name
            staged_marker.write_text(json.dumps(identity, indent=2) + "\n")
            if copied_source.exists():
                shutil.rmtree(copied_source)
            # Publishing the identity first also permits a retry if the
            # process is interrupted before the source directory is renamed.
            staged_marker.replace(marker)
            staged.rename(copied_source)
    elif (copied_source / "src/network.h").read_text() != patched:
        raise ValueError("Private SDK network.h was changed outside the managed build")
    # Build identity is set by a wrapper, leaving the SDK CMake source untouched.
    (destination / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(pressb_private_franka LANGUAGES CXX)\n"
        "add_subdirectory(source sdk)\n"
        "set_target_properties(franka PROPERTIES OUTPUT_NAME pressb_franka)\n"
    )
    dependency_keys = (
        "CMAKE_PREFIX_PATH",
        "Eigen3_DIR",
        "Poco_DIR",
        "PCRE_INCLUDE_DIR",
        "PCRE_LIBRARY",
        "pinocchio_DIR",
        "console_bridge_DIR",
        "fmt_DIR",
        "tinyxml2_DIR",
        "urdfdom_DIR",
        "urdfdom_headers_DIR",
    )
    options = [f"-D{key}={cache[key]}" for key in dependency_keys if cache.get(key)]
    return {
        "identity": identity,
        "dependency_options": options,
        "config_dir": destination / "build/sdk",
    }


def build_private_sdk(upstream_build: Path, destination: Path) -> tuple[Path, list[str]]:
    """Build the private source and carry its public dependencies over to the extension."""
    prepared = prepare_private_sdk(upstream_build.resolve(), destination.resolve())
    subprocess.run(
        [
            "cmake",
            "-S",
            str(destination),
            "-B",
            str(destination / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_DOCUMENTATION=OFF",
            "-DGENERATE_PYLIBFRANKA=OFF",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            *prepared["dependency_options"],
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(destination / "build"),
            "--target",
            "franka",
            "--parallel",
            "2",
        ],
        check=True,
    )
    public_options = [
        option for option in prepared["dependency_options"]
        if option.startswith(("-DCMAKE_PREFIX_PATH=", "-DEigen3_DIR=", "-Dfmt_DIR="))
    ]
    return prepared["config_dir"], public_options
