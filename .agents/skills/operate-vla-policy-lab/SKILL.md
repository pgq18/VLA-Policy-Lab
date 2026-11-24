---
name: operate-vla-policy-lab
description: Configure and run VLA-Policy-Lab (PressB) for supervised training, online RL, inference, collection, or evaluation. Use for task and model selection, node placement and connectivity, PiPER or FR3 hardware setup, and installation or verification of the Franka libfranka SDK.
---

# Operate VLA-Policy-Lab

Run commands from the repository root. The Python package and CLI are both named `pressb`. Use the same code revision across hosts. Install each role's dependencies on the host that runs it, and make its weights, datasets, Isaac assets, or robot devices available there. [Setup guide](../../../README.md)

## Choose what to run

| User goal | Command | Processes and inputs |
|---|---|---|
| Serve a model | `pressb inference --config ...` | Inference host with model weights |
| Measure a base or trained policy's success rate | `pressb eval` | Environment, inference service, and evaluation runner with learning disabled |
| Record policy rollouts | `pressb collect` | Same services as evaluation. This records action chunks. For frame-level Stack or Cleanup demonstrations, use the [collection](../../../scripts/collect_manipulation.py) and [LeRobot export](../../../scripts/export_manipulation_dataset.py) scripts. |
| Train online | `pressb train` | Environment, inference service, and learner. Choose the algorithm in `run.method`. |
| Fine-tune on an existing LeRobot dataset | `pressb-model-train` | Run the model-specific SFT YAML on the training host with the dataset and model dependencies. |

The four roles are simulation, physical robot, inference, and training. Each online run uses one environment service, supplied by the simulator or the physical robot. The rollout client requests actions from inference, sends them to the environment, and reads the next observation, reward, and termination flags.

## Select the task, model, and method

1. Run `pressb list` and choose an [experiment JSON](../../../configs/experiments/) that matches the task, model family, and action representation. Create a child recipe with `extends`. The parent path is relative to the child JSON. Nested objects merge by key and lists replace inherited lists. When changing model families, start from a recipe for the target family so its adapter receives the right fields.
2. Set `task.plugin` to the environment, such as `pap`, `pap_plus_plus`, `cube_stack`, or `elevator`. Set scene assets, episode limits, and parallel environment count in `task.config`. Select `piper_real` for a physical PiPER or `franka_real` for a Franka Research 3. Both use one environment and the `real_robot` command. The hardware template starts with actuation disabled. Before physical motion, verify operator authorization, device status, and the reset procedure. [Physical robot setup](../../../docs/real_robot.md)
3. Set `model.plugin` to the inference adapter, such as `pi05`, `walloss`, `vlajepa`, `vlajepa_joint`, or `diffusion_policy`. Set weights, normalization, and device in the adapter's `model.config`. Match camera names, joint or end-effector actions, units, gripper dimensions, and control frequency. Check that both the model and the environment support `run.rollout.execution_horizon`. [Model configuration](../../../docs/bundled_models.md)
4. Set the method, budget, seed, and output path in `run`. The `train`, `eval`, and `collect` commands select the runtime mode. To evaluate a base policy, set `run.method` to `base` and use `--episodes` for the episode count. For online training, `--max-transitions` counts action chunks across all environments. Each chunk can contain several control steps. For DSRL, action residuals on a frozen DSRL policy, DPPO, resume, or SFT, follow the [training and checkpoint guide](references/training.md).

## Assign hosts, devices, and addresses

A node runs on the host where its process starts. Activate that role's environment on that host. Interpret GPU indices in the device view of the corresponding process.

| Role | Command | Device and network settings |
|---|---|---|
| Simulation | `pressb simulation` | Set the GPU field supported by the task in `task.config`. PAP uses `gpu_ids`. The service listens on `nodes.simulation.server`. |
| Physical robot | `pressb real_robot` | Run on the station with CAN (PiPER) or FCI Ethernet (FR3) and cameras. The physical service uses `nodes.simulation.server` and the client connects to `run.endpoints.simulation`. |
| Inference | `pressb inference` | Set the model device in `model.config.device`, the listener in `nodes.inference.server`, and the batch limit in `nodes.inference.service.batch_size`. |
| Training or evaluation | `pressb train` or `pressb eval` | Set the runner device with `run.device` or `--device` and the two service URLs in `run.endpoints`. A DPPO learner also loads the model on its own host. |

For a three-host example, single-host GPU placement, or connectivity checks, use the [deployment guide](references/deployment.md). It includes an executable recipe and commands for each host.

## Prepare the PiPER CAN interface

For `piper_real`, work on the robot host. Use `python scripts/real_robot_probe.py --probe-can` or `ip -brief link show type can` to enumerate interfaces. Match the follower's physical adapter using `ethtool -i` when multiple buses exist. Set `CAN_IF` and `task.config.robot.config.can_name` to the same verified interface.

Use `ip -details -statistics link show dev "$CAN_IF"` to check its state and bitrate. For a setup request with the verified interface down, run `sudo ip link set dev "$CAN_IF" type can bitrate 1000000 && sudo ip link set dev "$CAN_IF" up`. Reuse a healthy interface already up at 1 Mbit/s. For a status request, use the read-only checks. Confirm the `UP` flag, `ERROR-ACTIVE` controller state and receipt of joint and gripper frames before reporting that feedback works. [Activation, passive feedback check and troubleshooting](../../../docs/real_robot.md#activate-and-verify-the-piper-can-bus)

## Install or check the Franka SDK

For `franka_real`, use the Python 3.10/3.11 environment that will start the node on the physical FR3 host. A quick probe is `python -c 'from pressb.robots.franka import _franka_native as n; print(n.__file__, n.libfranka_version, n.gripper_stop_concurrency_fixed)'`. For an installation-status request, run the [hardware-free SDK check](../../../docs/franka_sdk.md#check-an-existing-installation) and report the interpreter, SDK version (`0.20.5`), fix flag (`True`) and the private library path loaded by that process. The node's `/health` route only reads configuration.

If installation is requested and the check fails, inspect the upstream CMake cache and source pins in that guide. Reuse a matching build tree. On a new host, follow [Prepare the upstream SDK](../../../docs/franka_sdk.md#prepare-the-upstream-sdk-on-a-new-host) to install the C++ dependencies, initialize the pinned submodule and configure the SDK. From the PressB root, build in the selected environment.

```bash
python -m pip install -e '.[franka-real]'
python scripts/franka/build_native.py --libfranka-dir "${FRANKA_BUILD_DIR:-$HOME/libfranka/build}"
```

Repeat the mapped-library check after building. For a transport regression, run `python scripts/franka/check_gripper_transport.py --sdk-build build/franka_libfranka/build/sdk` against loopback. SDK verification completes without opening FCI, streaming cameras or enabling actuation.

## Validate and run

- In the runner environment, use `pressb validate --config "$CFG"` to inspect the resolved `factories` and `nodes`. To write the node configs, run `pressb prepare --config "$CFG" --resolved-dir "$OUT/resolved"`. Both commands operate offline. For a configuration-only request, deliver the validated recipe.
- For a rollout, start inference and the chosen environment service. From the runner host, query both `/health` endpoints and check the model, task, environment count, and action contract before starting the client. For an inference-only deployment, check the inference endpoint.
- Write each run to a fresh `outputs/<run-name>/run` directory. After evaluation, use `status.json` and `episodes.jsonl` to verify the stop reason, completed episode count, and success count. Compare policies under the same task distribution, seed, time limit, and action execution settings.
