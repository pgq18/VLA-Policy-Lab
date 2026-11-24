# Physical robot environments and human feedback

Choose the hardware with `task.plugin`. Both adapters run a single physical environment with `pressb real_robot`. The runner uses the same reset, step, reward, and termination interface as simulation. The robot host owns its hardware connection and cameras.

| Plugin | Hardware | Policy state and action | PAP camera keys |
| --- | --- | --- | --- |
| `piper_real` | PiPER follower over CAN | Six joint angles in radians and full gripper width in metres | `up`, `wrist` |
| `franka_real` | Franka Research 3 and Franka Hand over FCI Ethernet | Seven joint angles in radians and normalized `gripper_closure` (0 open, 1 closed) | `base` (D435i), `wrist` (D405) |

## Configure the Franka host

Only the FR3 station needs libfranka. First check the active Python environment and its loaded private SDK. If needed, reuse the station's existing `$HOME/libfranka/build` or build from the pinned source on a new host. The [FR3 SDK installation and verification guide](franka_sdk.md) includes both paths and a hardware-free acceptance check.

For physical execution, use a commissioned FCI connection, real-time scheduling permission for the controller, and the cameras attached. With the verified device Python environment active, discover the camera serials separately.

```bash
python scripts/real_robot_probe.py --discover
```

The C++ controller runs its 1 kHz loop on the robot host. Python sends joint targets and handles asynchronous gripper commands. PressB includes the control, camera and operator code.

Start with the [FR3 π₀.₅ evaluation recipe](../configs/experiments/fr3_real_pi05_eval.json). Set `task.config.robot.config.robot_ip` and replace both camera serial placeholders on the robot host. Set `model.config.checkpoint` to the native FR3 SFT `pretrained_model` directory on the inference host. The station recipe uses two 640 × 480 RGB streams at 30 Hz. The wrist camera uses manual exposure 16000 and gain 16. Camera pairing targets 20 ms and reports the measured skew when SDK timestamps share the global clock. The action chunk contains up to 16 absolute targets executed at 25 Hz. Joint EMA, if enabled, filters the seven arm joints and passes through gripper closure.

```bash
CFG=configs/experiments/fr3_real_pi05_eval.json
pressb real_robot --config "$CFG"  # robot station
```

By default `actuation_enabled` is false. `GET /health` reports the configuration without opening FCI. For a passive observation, `POST /observe` with an empty JSON object returns measured joints, closure, and both RGB frames. This request opens the FCI/Hand feedback connection and camera streams, applying the configured camera exposure. It issues zero arm or gripper targets. End the diagnostic session before another client takes ownership of the robot.

```bash
curl -fsS http://127.0.0.1:18741/health
python - <<'PY'
from pressb.runtime.client import NodeClient
client = NodeClient("http://127.0.0.1:18741")
try:
    sample = client.call("/observe", {})
    print(sample["observation"]["state"], list(sample["observation"]["images"]))
    print(sample["info"]["submitted_commands"])
finally:
    client.close()
PY
```

For an operator-authorized physical run, verify the scene, FCI state, Hand calibration and emergency stop, set `actuation_enabled` to true in a local recipe, and restart the robot node. Start inference on the model host, then the runner. The [DSRL recipe](../configs/experiments/fr3_real_pi05_dsrl.json) uses the same robot interface and accepts one operator label per episode. If using a local override for calibration or actuation, make sure every role resolves the intended configuration.

```bash
pressb inference --config "$CFG"                       # model host
pressb eval --config "$CFG" --episodes 1               # runner, evaluation recipe
# For online learning use the DSRL recipe on all nodes, then:
pressb train --config configs/experiments/fr3_real_pi05_dsrl.json
```

The robot service binds to `127.0.0.1:18741` and inference to `127.0.0.1:18751`. For example, on a runner colocated with inference, `ssh -N -L 18741:127.0.0.1:18741 franka` forwards the robot port. The FR3 operator mailbox defaults to `~/.local/state/pressb/fr3_pap/operator` on the robot host. Read the active token there and send `ready` after restoring the scene, then `success` or `failure` when the episode outcome is visible.

```bash
OP="$HOME/.local/state/pressb/fr3_pap/operator"
pressb-robot-operator --directory "$OP" status
pressb-robot-operator --directory "$OP" ready --episode RESET_TOKEN_FROM_STATUS
pressb-robot-operator --directory "$OP" success --episode ACTIVE_TOKEN_FROM_STATUS
```

The legacy standalone [SSH console recipe](../configs/evaluation/fr3_real_pi05.json) uses the same mailbox. Set its `operator_console.command[0]` to the absolute station path to `pressb-robot-operator` if the noninteractive SSH session cannot find that executable.

**Optional return to initial pose.** The FR3 template includes the recorded initial joints and closure with `task.config.return_to_initial.enabled` set to false. Enable return only for a verified, clear physical path. With both return and actuation enabled, the first reset can move the robot **before** the operator's `ready` acknowledgment. After a terminal step the runner receives the measured end pose and reward first. The next reset or final run close executes the pending return, moving the arm before submitting the optional gripper target. These motions are outside rollout step counts. The operator can send `abort` with the active token while returning.

## Configure the PiPER host

In a Python 3.10 or 3.11 environment, install the hardware dependencies and discover the local devices.

```bash
python -m pip install -e '.[real-robot]'
python scripts/real_robot_probe.py --discover --probe-can
```

The probe lists camera serials and Linux CAN interfaces. Its `--probe-can` option reads interface names and states from sysfs. The logical camera names `up` and `wrist` identify the global and wrist views respectively.

### Activate and verify the PiPER CAN bus

Run these commands on the **robot host** (for example, after `ssh piper`). On Ubuntu the tools are provided by `iproute2`, `ethtool`, `usbutils`, and `can-utils`.

1. Identify the USB-CAN adapter wired to the **follower**. If several adapters are present, compare each interface's `ethtool -i` output (`bus-info`) with its physical USB port. Set `CAN_IF` to the verified name and use the same name in `task.config.robot.config.can_name`. If the adapter is absent from `lsusb`, check its cable and USB port. If it is present but no CAN interface appears, load `gs_usb` with `sudo modprobe gs_usb` for adapters that use that driver, then repeat discovery.

   ```bash
   lsusb
   ip -brief link show type can
   export CAN_IF=can0  # replace with the dedicated follower interface
   ethtool -i "$CAN_IF"
   ip -details -statistics link show dev "$CAN_IF"
   ```

2. For a **down** interface, configure **1 Mbit/s** and bring it up. If it already shows the `UP` flag, `bitrate 1000000`, and controller state `ERROR-ACTIVE`, proceed directly to the feedback check. If it is up at another bitrate, stop the control session that owns it before intentionally taking it down with `sudo ip link set dev "$CAN_IF" down` and applying the commands below. Bringing the network interface up is separate from enabling the arm or gripper.

   ```bash
   sudo ip link set dev "$CAN_IF" type can bitrate 1000000 &&
     sudo ip link set dev "$CAN_IF" up
   ip -details -statistics link show dev "$CAN_IF"
   ```

3. With the arm powered, confirm that it publishes joint and gripper feedback. This bounded check listens for one frame from each joint pair (`2A5`–`2A7`) and the gripper (`2A8`). Expect one line per ID. A timeout, `BUS-OFF`, or rising error counters calls for checking follower selection, wiring, power and bitrate before launching the node.

   ```bash
   (
     for id in 2A5 2A6 2A7 2A8; do
       timeout 3s candump -n 1 "$CAN_IF,$id:7FF" || { echo "No frame read for $id" >&2; exit 1; }
     done
   )
   ```

### Configure the PiPER node

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
