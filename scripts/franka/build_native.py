# Adapted from the station LeRobot build and transport checks for PressB.
# SPDX-License-Identifier: Apache-2.0

"""Build the direct libfranka extension for this Python interpreter without opening a robot connection."""

import argparse
import fcntl
import importlib.util
import subprocess
import sys
import sysconfig
from pathlib import Path

from private_sdk import build_private_sdk


def verify_extension(package_dir: Path) -> Path:
    path = package_dir / ("_franka_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    spec = importlib.util.spec_from_file_location("_franka_native", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the native extension at {path}")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if native.libfranka_version != "0.20.5" or native.gripper_stop_concurrency_fixed is not True:
        raise RuntimeError(f"Expected the reviewed libfranka 0.20.5 build at {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libfranka-dir", type=Path, default=Path.home() / "libfranka/build",
        help="Existing libfranka 0.20.5 CMake build tree with its sources and common submodule",
    )
    parser.add_argument(
        "--pybind11-dir", type=Path,
        help="Directory containing pybind11Config.cmake (defaults to this Python's pybind11 or CMake discovery)",
    )
    args = parser.parse_args()
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        parser.error("Use the PressB device host's Python 3.10 or 3.11 interpreter")
    root = Path(__file__).resolve().parents[2]
    build = root / "build"
    build.mkdir(exist_ok=True)
    source = root / "src/pressb/robots/franka/native"
    extension_build = build / "franka_native" / sys.implementation.cache_tag
    pybind_dir = args.pybind11_dir
    if pybind_dir is None:
        try:
            import pybind11
        except ModuleNotFoundError:
            pass  # A system pybind11 CMake package also works with the selected Python.
        else:
            pybind_dir = Path(pybind11.get_cmake_dir())
    if pybind_dir is not None and not (pybind_dir / "pybind11Config.cmake").is_file():
        parser.error(f"Missing pybind11Config.cmake in {pybind_dir}")

    with (build / "franka_native.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        private_sdk, dependency_options = build_private_sdk(args.libfranka_dir.expanduser(), build / "franka_libfranka")
        subprocess.run(
            [
                "cmake", "-S", str(source), "-B", str(extension_build),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DFranka_DIR={private_sdk}",
                f"-DCMAKE_BUILD_RPATH={private_sdk}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={source.parent}",
                *dependency_options,
                f"-Dpybind11_DIR={pybind_dir.resolve()}" if pybind_dir is not None else "-Upybind11_DIR",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(extension_build), "--parallel", "2"], check=True)
        extension = verify_extension(source.parent)
    print(f"Built and imported {extension} with {sys.executable}")


if __name__ == "__main__":
    main()
