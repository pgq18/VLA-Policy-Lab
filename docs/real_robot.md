# PiPER physical environment and human feedback

The `piper_real` backend connects one physical PiPER arm and two RealSense cameras to the rollout service. The robot host reads joint feedback, captures observations and executes paced action prefixes. The same trainer can evaluate a policy or train a residual policy with episode outcomes supplied by an operator.

## Configure the robot host

In a Python 3.10 or 3.11 environment, install the hardware dependencies and discover the local devices.

```bash
python -m pip install -e '.[real-robot]'
python scripts/real_robot_probe.py --discover --probe-can
```

The probe lists camera serials and Linux CAN interfaces. Configure the dedicated follower CAN interface and check the two camera views. The logical names `up` and `wrist` identify the global and wrist cameras respectively. Each camera is sampled independently.

The evaluation configuration inherits the hardware, model and operator settings from `configs/experiments/pap_real_walloss_dsrl.json`. Set the following values for your installation.

| Setting | Value to supply |
| --- | --- |
| `task.config.robot.config` | CAN interface, command limits, speed and feedback timeout |
| `task.config.cameras.config.cameras` | Actual serial, resolution and frame rate for each logical camera |
| `task.config.task.config.directory` | Operator status and reset directory on the robot host |
| `run.human_feedback.directory` | Episode label directory on the trainer host |
| `model.config` | Model weights, normalization statistics and inference device |

The template defaults to `actuation_enabled: false`. Commission the arm and workspace, verify feedback and limits, and keep a physical emergency stop available before enabling actuation. The PiPER action is six absolute joint angles in radians followed by full gripper width in metres. The adapter converts these values to the SDK units and enforces the configured per-joint and gripper rates.

## Start an evaluation

Validate the configuration on the trainer host. Start inference in its model environment and the physical service on the robot host. Run evaluation after both services are ready. Commands use the repository root as their working directory.

```bash
python scripts/run_experiment.py validate --config configs/experiments/pap_real_walloss_eval.json

python scripts/run_experiment.py inference --config configs/experiments/pap_real_walloss_eval.json

python scripts/run_experiment.py real_robot --config configs/experiments/pap_real_walloss_eval.json

python scripts/run_experiment.py eval --config configs/experiments/pap_real_walloss_eval.json \
  --episodes 1 --output outputs/example_real_eval/run
```

The template binds to loopback. For separate hosts, configure SSH forwarding or reachable service addresses and set the trainer's `--inference` and `--environment` addresses. The physical service uses `nodes.simulation` and `run.endpoints.simulation` in the experiment file. [Node configuration](configurable_rl_nodes.md)

## Reset and label an episode

On the robot host, read the reset status, restore the physical scene, and copy the current reset token into the `ready` command. The command acknowledges that the scene is ready for the backend to read the measured state.

```bash
python -m pressb.runtime.real_robot_operator --directory outputs/pap_real_operator status
python -m pressb.runtime.real_robot_operator --directory outputs/pap_real_operator \
  ready --episode RESET_TOKEN_FROM_STATUS
```

When `run.human_feedback.mode` is `human`, obtain the current environment episode ID from the trainer host. Use `success` or `failure` as the observed outcome.

```bash
python -m pressb.runtime.human_feedback --directory outputs/pap_real_feedback status
python -m pressb.runtime.human_feedback --directory outputs/pap_real_feedback \
  success --episode EPISODE_ID_FROM_STATUS
```

The trainer applies the label at an action-chunk boundary. A reset token is consumed by the robot-host operator, while the trainer uses the episode ID issued after reset. To supply rewards from a sensor-based detector, configure a task factory on the robot host and use the trainer's `environment` feedback mode.
