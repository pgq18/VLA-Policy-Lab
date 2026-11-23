# VLA Policy Lab

**English** · [简体中文](README.zh-CN.md)

VLA Policy Lab is a Python framework for training and evaluating robot policies. It supports supervised fine-tuning on LeRobot demonstrations and online reinforcement learning through parallel Isaac Sim environments or a physical PiPER arm. Inference, simulation, and training can run on separate machines.

The framework includes VLA-JEPA, Wall-OSS, Diffusion Policy, and LeRobot π₀.₅. Tasks cover elevator and tabletop button pressing, pick-and-place, cube stacking, and tabletop cleanup. Online training can learn the initial noise (DSRL), action residuals, or both with a frozen policy. DPPO fine-tunes the action expert.

[Getting started](#getting-started) · [Elevator walkthrough](#walkthrough-press-the-requested-elevator-button) · [Framework guide](docs/project_framework.md) · [Agent skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

## Getting started

Use Python **3.10 or 3.11**. The package and CLI are named `pressb`. Install the model dependencies in the training and inference environments and the Isaac Sim dependencies on the simulation host. Install `python -m pip install -e '.[online-rl]'` in the RL runner environment. [Model setup](docs/bundled_models.md) · [Simulator setup](docs/rtx5090_setup.md)

```bash
git clone https://github.com/pgq18/VLA-Policy-Lab.git
cd VLA-Policy-Lab
python -m pip install -e .
pressb list
```

## Walkthrough: press the requested elevator button

A PiPER arm must press the button for a requested floor between 24 and 35. The three stages below use the same VLA-JEPA pose9 policy. SFT learns from demonstrations. Online RL learns from simulated button presses. Evaluation measures completed episodes with learning disabled.

### 1. Fine-tune VLA-JEPA on demonstrations

On the SFT host, install the [VLA-JEPA training dependencies](src/pressb/model_sources/vlajepa/requirements-piper.txt) and supply a 30 Hz elevator **press-prefix LeRobot export** with global and wrist images and 100 successful episodes per floor (1,200 total). The reader converts the demonstrated TCP poses to XYZ plus a 6D rotation. If starting from raw elevator demonstrations, use the [press-prefix exporter](scripts/export_press_dataset.py).

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

### 2. Train online using one RL method

The [example recipes](examples/elevator/) share the same SFT weights, one simulated environment and a training budget of 10,000 chunk transitions, with up to seven control steps per chunk. Choose a recipe for the parameters you want to update.

| Recipe | What learns | Starting point |
| --- | --- | --- |
| [`dsrl`](examples/elevator/dsrl.json) | A SAC actor selects the decoder's initial noise | Frozen SFT policy |
| [`action_residual`](examples/elevator/action_residual.json) | A SAC actor adds XYZ corrections of up to 0.02 m per axis | Frozen SFT policy |
| [`joint_residual`](examples/elevator/joint_residual.json) | One SAC actor adjusts both noise and XYZ actions | Frozen SFT policy |
| [`residual_after_dsrl`](examples/elevator/residual_after_dsrl.json) | A fresh XYZ residual actor and critics train on top of a fixed DSRL actor | Same SFT weights **plus** the stage-one DSRL checkpoint |
| [`dppo`](examples/elevator/dppo.json) | PPO updates the action expert and publishes each new version to inference | SFT weights on both inference and learner hosts, four denoising steps |

**Prepare the simulator.** The recipes use the frozen training scene under `outputs/rtx5090_eval_comparison/input/`: the scene's `config.json`, `scene.usda`, `dataset_metadata/meta/collection_metadata.json`, and `assets/` bundle. Supply these external inputs and the PiPER assets. The examples use GPU 0 for simulation, GPU 1 for inference and learning, and loopback service addresses. Adjust the devices and endpoints for your hosts. [Node configuration](docs/configurable_rl_nodes.md)

**Launch the three processes.** In each terminal, set the same `PRESSB_SFT_CHECKPOINT`, `PRESSB_QWEN_ROOT`, and `PRESSB_VJEPA_ROOT` variables from stage 1 (use paths readable on the corresponding host). Choose one method below and give the run a fresh name. To train another method, change `METHOD` and start its services with the matching recipe.

```bash
export METHOD=dsrl
export CFG="examples/elevator/$METHOD.json" ELEVATOR_RUN="$METHOD-train-01"
```

| Terminal | Command |
| --- | --- |
| 1 · Inference | `pressb inference --config "$CFG"` |
| 2 · Simulation | `pressb simulation --config "$CFG"` |
| 3 · Learner | `pressb validate --config "$CFG"`, then `pressb train --config "$CFG"` once both services are ready |

After training, read the **actual saved RL checkpoint path** from the runner status. SAC and DPPO use different checkpoint filenames.

```bash
export RL_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["last_checkpoint"])' \
  "outputs/elevator/$ELEVATOR_RUN/run/status.json")"
test -f "$RL_CHECKPOINT"
```

For the two-stage method, complete `dsrl` first. Use its saved path for the following setup in all three terminals (copy the path from the learner terminal). Restart the three processes with this recipe.

```bash
export PRESSB_DSRL_CHECKPOINT="$RL_CHECKPOINT"
export METHOD=residual_after_dsrl
export CFG="examples/elevator/$METHOD.json" ELEVATOR_RUN="$METHOD-train-01"
```

DSRL now supplies deterministic initial noise while the residual learner starts from scratch. After this stage, read its new `RL_CHECKPOINT` and keep the same `PRESSB_DSRL_CHECKPOINT` for evaluation.

### 3. Evaluate SFT and RL on the same task

For each row, set `CFG` and `ELEVATOR_RUN` as shown **in all three terminals**. Stop the previous services and restart inference and simulation with the new configuration. In the runner terminal, execute the evaluation command. You can run the SFT baseline immediately after stage 1.

| Policy | Configuration and fresh run name | Runner command |
| --- | --- | --- |
| SFT baseline | `CFG=examples/elevator/base.json` · `export ELEVATOR_RUN=sft-eval-01` | `pressb eval --config "$CFG" --episodes 10 --seed 43` |
| Trained RL policy | `CFG="examples/elevator/$METHOD.json"` · `export ELEVATOR_RUN="$METHOD-eval-01"` | `pressb eval --config "$CFG" --resume "$RL_CHECKPOINT" --episodes 10 --seed 43` |

For a paired DPPO comparison, also evaluate its starting expert using `dppo.json` with a freshly started SFT inference service, a fresh run name, and no `--resume`. This keeps the DPPO sampler identical before and after training. Keep the evaluation seed, floor sampler, episode timeout and action smoothing fixed.

Read `outputs/elevator/$ELEVATOR_RUN/run/status.json` for `episodes`, `successes` and `success_rate`. Check that all ten episodes completed. The neighboring `episodes.jsonl` contains the individual outcomes.

## Help and resources

- **Choose a workflow.** Use `pressb collect` to record policy rollouts or `pressb train` with an online RL recipe. For physical PiPER rollouts, use a hardware recipe and start `pressb real_robot` on the robot host. [Online RL](docs/configurable_rl_nodes.md) · [DPPO](docs/dppo.md) · [Physical robots](docs/real_robot.md)
- **Inspect a configuration.** `pressb --help` lists commands. `pressb validate --config "$CFG"` shows the resolved settings. The deployment guide's `/health` probe checks the running services from the client.
- **Try the elevator scene without policy weights.** Run the scripted button-press sequence and check its contacts and camera streams. [Elevator scene walkthrough](docs/rtx5090_setup.md#run-the-elevator-scene)
- **Use an AI agent.** Invoke `$operate-vla-policy-lab` to select a recipe, assign GPUs, and configure node connections. [Repository skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

Reusable code, configs, and guides belong in Git. Put run-specific checkpoints, logs, plots, and recordings under the ignored `outputs/` directory. Primary documentation and agent skills use English. Translated guides use a language suffix such as `.zh-CN.md`.

## Project and license

[Contributors](https://github.com/pgq18/VLA-Policy-Lab/graphs/contributors) · [Version history](https://github.com/pgq18/VLA-Policy-Lab/commits/main/) · [Report an issue](https://github.com/pgq18/VLA-Policy-Lab/issues)

VLA Policy Lab's original code is licensed under the [MIT License](LICENSE). Third-party code, models, and robot assets retain their respective licenses and notices. [Third-party sources and licenses](THIRD_PARTY.md)

## Acknowledgments

We thank the teams behind these projects.

- [Wall-OSS (Wall-X)](https://github.com/X-Square-Robot/wall-x) and [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) for the policies adapted for robot training and inference.
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) for the demonstration dataset format and the π₀.₅ policy implementation.
- [DPPO](https://github.com/irom-princeton/dppo) for diffusion-policy optimization, and [ZPRL](https://github.com/ManUtdMoon/ZPRL) for its SAC implementation and work on steering frozen policies.

The README layout draws on [DomPizzie's template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc).
