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

The elevator example trains a PiPER arm to press the requested floor button. On the SFT host, install the [VLA-JEPA training dependencies](src/pressb/model_sources/vlajepa/requirements-piper.txt). The pose9 reader expects a 30 Hz LeRobot press-prefix dataset with global and wrist images and 100 successful demonstrations for each floor from 24 through 35. [Export from raw demonstrations](scripts/export_press_dataset.py)

Open [`sft.yaml`](examples/elevator/sft.yaml) and edit the `paths` block to match your dataset, pretrained VLA-JEPA checkpoint, and Qwen3-VL and V-JEPA2 backbone directories. The training settings and run ID are in the same file. Run from the repository root.

```bash
torchrun --standalone --nproc-per-node=1 scripts/train_model.py \
  --model vlajepa --profile pose9 --config examples/elevator/sft.yaml
```

The default is 2,000 optimizer steps. The validation-selected checkpoint appears at `outputs/elevator/sft/pose9/checkpoints/best`. Make the **whole SFT run directory** available on the inference host, including the processor, training metadata and the checkpoint targeted by `best`.

### Online RL training

The YAML recipes come with defaults for one simulated environment, 10,000 chunk transitions and up to seven control steps per chunk. Configure the deployment once in [`base.yaml`](examples/elevator/base.yaml). Its default SFT checkpoint matches the output above. If you change the SFT run ID or store assets elsewhere, update the checkpoint, backbone and frozen scene paths here. The example expects `config.json`, `scene.usda`, `dataset_metadata/meta/collection_metadata.json` and `assets/` under `outputs/rtx5090_eval_comparison/input/`.

The defaults use GPU 0 for Isaac Sim, GPU 1 for inference and online learning, and loopback service addresses. For separate hosts, set the bind addresses in `nodes` and the runner's connection URLs in `run.endpoints`. Select the learning method by editing **one line** in [`train.yaml`](examples/elevator/train.yaml), for example `extends: dsrl.yaml`. The selected recipe sets the run `name` and learner device. Use a fresh name when repeating a run.

| Training recipe | Parameters updated |
| --- | --- |
| [`dsrl.yaml`](examples/elevator/dsrl.yaml) (default) | A SAC actor selects the frozen decoder's initial noise. |
| [`action_residual.yaml`](examples/elevator/action_residual.yaml) | A residual actor adjusts the decoded XYZ actions by up to 0.02 m per axis. |
| [`joint_residual.yaml`](examples/elevator/joint_residual.yaml) | One actor adjusts both the initial noise and XYZ actions. |
| [`residual_after_dsrl.yaml`](examples/elevator/residual_after_dsrl.yaml) | A fresh residual learner corrects the actions of a fixed DSRL policy. |
| [`dppo.yaml`](examples/elevator/dppo.yaml) | PPO updates the four-step action expert and publishes it to inference. The learner host also needs the SFT model. |

Run the first two commands in separate terminals, each in its own service environment. Once both services are ready, use the third terminal to validate the configuration and start learning.

```bash
# Terminal 1: inference
pressb inference --config examples/elevator/train.yaml
```

```bash
# Terminal 2: simulation
pressb simulation --config examples/elevator/train.yaml
```

```bash
# Terminal 3: learner
pressb validate --config examples/elevator/train.yaml
pressb train --config examples/elevator/train.yaml
```

**To train a residual after DSRL**, finish the default DSRL run first. Change `extends` in `train.yaml` to `residual_after_dsrl.yaml`, stop the old services, then repeat the commands above. The residual recipe reads the completed DSRL run's `run/status.json` on the learner and freezes the checkpoint recorded in `last_checkpoint`. To use a different run or a particular step, change `run.frozen_dsrl.checkpoint` in that YAML.

### Evaluation

Choose one configuration for **all three terminals**. Use [`base.yaml`](examples/elevator/base.yaml) for the SFT baseline and [`eval.yaml`](examples/elevator/eval.yaml) for the trained policy. The latter inherits the method selected in `train.yaml` and defaults to the final checkpoint of `dsrl-train-01`. When evaluating a different training run, set `run.checkpoint.resume` to its completed `status.json` or an explicit `.pt` file. Give the evaluation its own fresh `name`. For a two-stage residual policy, keep the same frozen DSRL source and point `resume` at the **residual** run.

| Terminal | SFT baseline | After RL |
| --- | --- | --- |
| Inference | `pressb inference --config examples/elevator/base.yaml` | `pressb inference --config examples/elevator/eval.yaml` |
| Simulation | `pressb simulation --config examples/elevator/base.yaml` | `pressb simulation --config examples/elevator/eval.yaml` |
| Runner | `pressb eval --config examples/elevator/base.yaml` | `pressb eval --config examples/elevator/eval.yaml` |

Restart services between evaluations. Both recipes default to 10 episodes and seed 43. For a paired DPPO baseline, use the DPPO training recipe as the parent of `eval.yaml`, set `run.checkpoint.resume: null`, and start fresh inference from SFT. To evaluate the trained expert, restore its checkpoint reference and change the evaluation name.

Read `episodes`, `successes` and `success_rate` in `outputs/elevator/<name>/run/status.json`. Per-episode outcomes are in `episodes.jsonl`.

## Help and resources

- **Choose a workflow.** Use `pressb collect` to record policy rollouts or `pressb train` with an online RL recipe. For PiPER or Franka Research 3 rollouts, select the matching hardware recipe and start `pressb real_robot` on the robot host. [Online RL](docs/configurable_rl_nodes.md) · [DPPO](docs/dppo.md) · [Physical robots](docs/real_robot.md)
- **Inspect a configuration.** `pressb --help` lists commands. `pressb validate --config examples/elevator/train.yaml` shows the resolved settings. The deployment guide's `/health` probe checks the running services from the client.
- **Try the elevator scene without policy weights.** Run the scripted button-press sequence and check its contacts and camera streams. [Elevator scene walkthrough](docs/rtx5090_setup.md#run-the-elevator-scene)
- **Use an AI agent.** Invoke `$operate-vla-policy-lab` to select a recipe, assign GPUs, and configure node connections. [Repository skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

Reusable code, configs, and guides belong in Git. Put run-specific checkpoints, logs, plots, and recordings under the ignored `outputs/` directory. Primary documentation and agent skills use English. Translated guides use a language suffix such as `.zh-CN.md`.

## Acknowledgments

We thank the teams behind these projects.

- [Wall-OSS (Wall-X)](https://github.com/X-Square-Robot/wall-x) and [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) for the policies adapted for robot training and inference.
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) for the demonstration dataset format and the π₀.₅ policy implementation.
- [DPPO](https://github.com/irom-princeton/dppo) for diffusion-policy optimization, and [ZPRL](https://github.com/ManUtdMoon/ZPRL) for its SAC implementation and work on steering frozen policies.
