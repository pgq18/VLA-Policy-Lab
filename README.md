# VLA Policy Lab

**English** · [简体中文](README.zh-CN.md)

VLA Policy Lab is a Python framework for training and evaluating robot policies. It supports supervised fine-tuning on LeRobot demonstrations and online reinforcement learning in parallel Isaac Sim environments and on physical PiPER and Franka Research 3 arms. Inference, simulation, and training can run on separate machines.

The framework includes VLA-JEPA, Wall-OSS, Diffusion Policy, and LeRobot π₀.₅. Tasks cover elevator and tabletop button pressing, pick-and-place, cube stacking, and tabletop cleanup. Online training can learn the initial noise (DSRL), action residuals, or both with a frozen policy. DPPO fine-tunes the action expert.

[Installation](#installation) · [Fine-tuning](#fine-tune-on-demonstrations) · [Online RL](#online-rl-training) · [Evaluation](#evaluation) · [Framework guide](docs/project_framework.md) · [Agent skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

## Getting started

### Installation

Use Python **3.10 or 3.11**. The package and CLI are named `pressb`. Install the model dependencies on the training and inference hosts, and Isaac Sim on the simulation host. In the model environment, install `python -m pip install -e '.[inference-server]'` for the inference service. In the runner environment, install `python -m pip install -e '.[online-rl]'`. [Model setup](docs/bundled_models.md) · [Simulator setup](docs/rtx5090_setup.md)

```bash
git clone https://github.com/pgq18/VLA-Policy-Lab.git
cd VLA-Policy-Lab
python -m pip install -e .
pressb list
```

### Fine-tune on demonstrations

Fine-tune VLA-JEPA to press the requested elevator button with a PiPER arm. On the SFT host, install the [VLA-JEPA training dependencies](src/pressb/model_sources/vlajepa/requirements-piper.txt) and supply a 30 Hz elevator **press-prefix LeRobot export** with global and wrist images and 100 successful episodes for each floor from 24 to 35 (1,200 total). The reader converts the demonstrated TCP poses to XYZ plus a 6D rotation. If starting from raw elevator demonstrations, use the [press-prefix exporter](scripts/export_press_dataset.py).

```bash
export PRESSB_DATASET_ROOT=/data/elevator/press_prefix_lerobot
export PRESSB_PRETRAINED_CHECKPOINT=/models/vla-jepa/model.pt
export PRESSB_QWEN_ROOT=/models/Qwen3-VL-2B-Instruct
export PRESSB_VJEPA_ROOT=/models/vjepa2-vitl-fpc64-256
export PRESSB_RUN_ROOT="$PWD/outputs/elevator/sft"

torchrun --standalone --nproc-per-node=1 scripts/train_model.py \
  --model vlajepa --profile pose9 --config configs/model_training/vlajepa_pose9.yaml \
  -- --run-id pose9
```

The [SFT configuration](configs/model_training/vlajepa_pose9.yaml) defaults to 2,000 optimizer steps. After completion, use the checkpoint selected on validation for inference. Make the **whole SFT run directory** available to the model host, including `processor/`, `config.yaml`, the schema and training metadata, and the checkpoint targeted by `best`.

```bash
export PRESSB_SFT_CHECKPOINT="$PRESSB_RUN_ROOT/pose9/checkpoints/best"
```

### Online RL training

The [example recipes](examples/elevator/) share the same SFT weights, one simulated environment and a training budget of 10,000 chunk transitions, with up to seven control steps per chunk. Choose a recipe for the parameters you want to update.

| Recipe | What learns | Starting point |
| --- | --- | --- |
| [`dsrl`](examples/elevator/dsrl.json) | A SAC actor selects the decoder's initial noise | Frozen SFT policy |
| [`action_residual`](examples/elevator/action_residual.json) | A SAC actor adds XYZ corrections of up to 0.02 m per axis | Frozen SFT policy |
| [`joint_residual`](examples/elevator/joint_residual.json) | One SAC actor adjusts both noise and XYZ actions | Frozen SFT policy |
| [`residual_after_dsrl`](examples/elevator/residual_after_dsrl.json) | A fresh XYZ residual actor and critics train on top of a fixed DSRL actor | Same SFT weights **plus** the trained DSRL checkpoint |
| [`dppo`](examples/elevator/dppo.json) | PPO updates the action expert and publishes each new version to inference | SFT weights on both inference and learner hosts, four denoising steps |

**Prepare the simulator.** The recipes use the frozen training scene under `outputs/rtx5090_eval_comparison/input/`: the scene's `config.json`, `scene.usda`, `dataset_metadata/meta/collection_metadata.json`, and `assets/` bundle. Supply these external inputs and the PiPER assets. The examples use GPU 0 for simulation, GPU 1 for inference and learning, and loopback service addresses. Adjust the devices and endpoints for your hosts. [Node configuration](docs/configurable_rl_nodes.md)

In each terminal, set `PRESSB_SFT_CHECKPOINT` to the fine-tuned checkpoint and set `PRESSB_QWEN_ROOT` and `PRESSB_VJEPA_ROOT` to the backbone directories. Use paths readable on the corresponding host. Select a recipe from the table and a fresh run name in all three terminals.

```bash
export METHOD=dsrl
export CFG="examples/elevator/$METHOD.json" ELEVATOR_RUN="$METHOD-train-01"
```

Start inference and simulation in separate terminals. Run the learner when both services are ready.

```bash
# Terminal 1: model environment
pressb inference --config "$CFG"
```

```bash
# Terminal 2: Isaac Sim environment
pressb simulation --config "$CFG"
```

```bash
# Terminal 3: learner environment, once both services are ready
pressb validate --config "$CFG"
pressb train --config "$CFG"
```

After training, read the **actual saved RL checkpoint path** from the runner status. SAC and DPPO use different checkpoint filenames.

```bash
export RL_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["last_checkpoint"])' \
  "outputs/elevator/$ELEVATOR_RUN/run/status.json")"
test -f "$RL_CHECKPOINT"
```

For two-stage training, take `RL_CHECKPOINT` from the completed DSRL run. Set it to the same saved path in all three terminals, then select the residual recipe and restart the services and learner.

```bash
export PRESSB_DSRL_CHECKPOINT="$RL_CHECKPOINT"
export METHOD=residual_after_dsrl
export CFG="examples/elevator/$METHOD.json" ELEVATOR_RUN="$METHOD-train-01"
```

DSRL now supplies deterministic initial noise while the residual learner starts from scratch. After this stage, read its new `RL_CHECKPOINT` and keep the same `PRESSB_DSRL_CHECKPOINT` for evaluation.

### Evaluation

Evaluate the SFT checkpoint as soon as fine-tuning finishes. In all three terminals select the base recipe and a fresh run name, then start the inference and simulation services with that configuration. Run the evaluation command in the third terminal.

```bash
# All terminals
export CFG=examples/elevator/base.json ELEVATOR_RUN=sft-eval-01

# Runner, once both services are ready
pressb eval --config "$CFG" --episodes 10 --seed 43
```

To evaluate the RL policy, select its training recipe and restart both services with the new configuration. In the runner terminal, pass the RL checkpoint saved by that method. For `residual_after_dsrl`, also keep the original `PRESSB_DSRL_CHECKPOINT` set in all three terminals.

```bash
# All terminals, with METHOD set to the trained recipe
export CFG="examples/elevator/$METHOD.json" ELEVATOR_RUN="$METHOD-eval-01"

# Runner
pressb eval --config "$CFG" --resume "$RL_CHECKPOINT" --episodes 10 --seed 43
```

For a paired DPPO comparison, also evaluate its starting expert using `dppo.json` with a freshly started SFT inference service, a fresh run name, and no `--resume`. This keeps the DPPO sampler identical before and after training. Keep the evaluation seed, floor sampler, episode timeout and action smoothing fixed.

Read `outputs/elevator/$ELEVATOR_RUN/run/status.json` for `episodes`, `successes` and `success_rate`. Check that all ten episodes completed. The neighboring `episodes.jsonl` contains the individual outcomes.

## Help and resources

- **Choose a workflow.** Use `pressb collect` to record policy rollouts or `pressb train` with an online RL recipe. For PiPER or Franka Research 3 rollouts, select the matching hardware recipe and start `pressb real_robot` on the robot host. [Online RL](docs/configurable_rl_nodes.md) · [DPPO](docs/dppo.md) · [Physical robots](docs/real_robot.md)
- **Inspect a configuration.** `pressb --help` lists commands. `pressb validate --config "$CFG"` shows the resolved settings. The deployment guide's `/health` probe checks the running services from the client.
- **Try the elevator scene without policy weights.** Run the scripted button-press sequence and check its contacts and camera streams. [Elevator scene walkthrough](docs/rtx5090_setup.md#run-the-elevator-scene)
- **Use an AI agent.** Invoke `$operate-vla-policy-lab` to select a recipe, assign GPUs, and configure node connections. [Repository skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

Reusable code, configs, and guides belong in Git. Put run-specific checkpoints, logs, plots, and recordings under the ignored `outputs/` directory. Primary documentation and agent skills use English. Translated guides use a language suffix such as `.zh-CN.md`.

## Acknowledgments

We thank the teams behind these projects.

- [Wall-OSS (Wall-X)](https://github.com/X-Square-Robot/wall-x) and [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) for the policies adapted for robot training and inference.
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) for the demonstration dataset format and the π₀.₅ policy implementation.
- [DPPO](https://github.com/irom-princeton/dppo) for diffusion-policy optimization, and [ZPRL](https://github.com/ManUtdMoon/ZPRL) for its SAC implementation and work on steering frozen policies.
