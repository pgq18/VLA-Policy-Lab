# Install and verify libfranka for the FR3

Run these commands **on the physical Franka host**, from the root of the PressB checkout. Simulation, inference, and learner hosts do not need this SDK. Use the Python **3.10 or 3.11** environment that will launch `pressb real_robot`. All checks below work with the arm disconnected.

## Check an existing installation

**Is PressB ready to use the SDK?** Run this first. It imports the native extension, checks its compiled SDK version and gripper fix, and reads the shared library actually loaded by this Linux process.

```bash
python - <<'PY'
from pathlib import Path
import sys
from pressb.robots.franka import _franka_native as native

print("Python:", sys.executable)
print("Extension:", native.__file__)
print("SDK version:", native.libfranka_version)
print("Concurrent gripper stop fix:", native.gripper_stop_concurrency_fixed)
expected = Path("build/franka_libfranka/build/sdk/libpressb_franka.so").resolve(strict=True)
loaded = {Path(line.split(maxsplit=5)[5]).resolve()
          for line in Path("/proc/self/maps").read_text().splitlines()
          if "/libpressb_franka.so." in line}
print("Loaded SDK:", *sorted(loaded))
if (native.libfranka_version != "0.20.5" or
        native.gripper_stop_concurrency_fixed is not True or loaded != {expected}):
    sys.exit("FR3 SDK check failed: verify the Python environment, build tree and library paths")
print("PressB FR3 SDK ready")
PY
```

If it prints `PressB FR3 SDK ready`, installation is complete. A missing `_franka_native` usually means the extension has not been built for this interpreter. If the import names a missing shared library, inspect the extension for this Python ABI. Restore any missing dependency paths or rebuild the private library in this checkout.

```bash
franka_suffix=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
ldd "src/pressb/robots/franka/_franka_native$franka_suffix"
```

**Can the existing upstream source/build tree be reused?** On the `franka` station the usual build tree is `$HOME/libfranka/build`. Set `FRANKA_BUILD_DIR` to the actual location. The builder reads the **upstream CMake cache**, its original source tree, and the `common` submodule. A system package listing, an installed `FrankaConfig.cmake`, or `pip show pylibfranka` alone cannot verify these inputs.

```bash
export FRANKA_BUILD_DIR="${FRANKA_BUILD_DIR:-$HOME/libfranka/build}"
if test -r "$FRANKA_BUILD_DIR/CMakeCache.txt"; then
  franka_src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$FRANKA_BUILD_DIR/CMakeCache.txt")
  git -C "$franka_src" rev-parse HEAD
  git -C "$franka_src" submodule status -- common
  sha256sum "$franka_src/src/network.h"
else
  echo "Upstream CMake cache missing: $FRANKA_BUILD_DIR"
fi
```

The reviewed SDK commit is `ba86f977b469cc2cf9c66f68b874612559049698`, the initialized `common` commit is `2e090a65e51c94a98f9fc82b6ec3d1aa54d0f85f`, and the upstream header SHA-256 is `5f38e76b0bf61d7dfde23270235ca314a282f75fbb31ff348fffdcfa37a9fa7e`. If the tree matches, skip to [Build the PressB extension](#build-the-pressb-extension). For a missing or different tree, prepare the reviewed revision in a fresh directory.

## Prepare the upstream SDK on a new host

The following dependency example is for **Ubuntu 24.04 amd64** with the official ROS 2 Jazzy APT repository enabled. It obtains the Pinocchio C++ library from that repository. PressB talks directly to libfranka over FCI. If `ros-jazzy-pinocchio` has no APT candidate, configure the official package source first. For a non-ROS installation, the pinned libfranka build guide gives a source-build route for Pinocchio and its dependencies. [ROS 2 Jazzy APT setup](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) · [Pinned libfranka build guide](https://github.com/frankarobotics/libfranka/blob/ba86f977b469cc2cf9c66f68b874612559049698/README.rst)

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git pybind11-dev libeigen3-dev \
  libpoco-dev libfmt-dev libconsole-bridge-dev libtinyxml2-dev ros-jazzy-pinocchio
export FRANKA_DEPS_PREFIX=/opt/ros/jazzy
```

Use CMake 3.22 or later and an active Python environment with its matching development headers. On a new Conda host, for example, use `conda create -n pressb-fr3 python=3.11 pip` and `conda activate pressb-fr3`. Clone into a **new** SDK directory outside the PressB checkout. Checking out the precise commit matters because multiple revisions identify themselves as 0.20.5. For an alternative Pinocchio installation, set `FRANKA_DEPS_PREFIX` to its CMake prefix.

```bash
export FRANKA_SRC="$HOME/libfranka-pressb-0.20.5"
export FRANKA_BUILD_DIR="$FRANKA_SRC/build"
git clone https://github.com/frankarobotics/libfranka.git "$FRANKA_SRC" &&
  git -C "$FRANKA_SRC" checkout --detach ba86f977b469cc2cf9c66f68b874612559049698 &&
  git -C "$FRANKA_SRC" submodule update --init --recursive &&
cmake -S "$FRANKA_SRC" -B "$FRANKA_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$FRANKA_DEPS_PREFIX" \
  -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCUMENTATION=OFF \
  -DGENERATE_PYLIBFRANKA=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON
```

Successful configuration produces `$FRANKA_BUILD_DIR/CMakeCache.txt` and checks the C++ dependencies. This is sufficient input for the next step, which performs the actual SDK compilation. Keep the source and build tree available for subsequent rebuilds.

## Build the PressB extension

```bash
python -m pip install -e '.[franka-real]'
python scripts/franka/build_native.py --libfranka-dir "$FRANKA_BUILD_DIR"
```

The Python extra supplies device-node dependencies including RealSense. The builder checks the SDK version and header, applies the station's gripper concurrency fix to a private copy, builds `libpressb_franka.so`, and imports the extension for the active Python ABI. It prints `Built and imported …` on success. Keep `build/franka_libfranka/` in this checkout for runtime linking. If CMake cannot find pybind11, use `--pybind11-dir` with the directory containing `pybind11Config.cmake`. After installing or changing interpreters, repeat the [installation check](#check-an-existing-installation).

Optionally exercise the compiled SDK's gripper transport against a local test server. It uses `127.0.0.1:1338` and reports `cycles: 80`, `following_moves: 80`, and `physical_hardware_connected: false` when all cycles pass.

```bash
python scripts/franka/check_gripper_transport.py --sdk-build build/franka_libfranka/build/sdk
```

SDK verification ends here. RealSense serial discovery, robot FCI availability and camera capture are separate commissioning steps in the [physical node guide](real_robot.md#configure-the-franka-host).
