# Adapted from the station LeRobot build to require the PressB private SDK.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(pressb_franka LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Franka 0.20.5 EXACT CONFIG REQUIRED)
get_target_property(FRANKA_PRIVATE_LOCATION Franka::Franka IMPORTED_LOCATION_RELEASE)
if(NOT EXISTS "${Franka_DIR}/../../pressb_private_sdk.json" OR
   NOT FRANKA_PRIVATE_LOCATION MATCHES "/libpressb_franka\\.so\\.")
  message(FATAL_ERROR "Build the reviewed private SDK with scripts/franka/build_native.py")
endif()
pybind11_add_module(_franka_native bindings.cpp)
target_link_libraries(_franka_native PRIVATE Franka::Franka Eigen3::Eigen)
target_compile_definitions(_franka_native PRIVATE PRESSB_LIBFRANKA_VERSION="${Franka_VERSION}"
                                                  PRESSB_FRANKA_SAFE_CONCURRENT_STOP=1)
target_compile_options(_franka_native PRIVATE -Wall -Wextra -Werror=return-type)
