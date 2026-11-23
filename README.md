# VLA Policy Lab

**English** | [简体中文](README.zh-CN.md)

VLA Policy Lab is a Python framework for collecting robot trajectories, evaluating vision-language-action policies, and training them through environment interaction. Inference, simulation or PiPER hardware, and the experiment runner operate as separate processes. An experiment configuration selects the task and model and connects the processes.

The task plugins cover button pressing, pick-and-place with PiPER or Franka Research 3, cube stacking, and tabletop cleanup. Model adapters support Wall-OSS, VLA-JEPA, Diffusion Policy, and LeRobot π₀.₅. Online training options include DSRL, action residuals, and DPPO.

## Quick start

Activate a Python 3.10 or 3.11 environment, then install the core package. The Python package and experiment command are both named `pressb`.

```bash
git clone https://github.com/pgq18/VLA-Policy-Lab.git
cd VLA-Policy-Lab
python -m pip install -e .
pressb list
```

`pressb list` prints the registered task and model plugins. To inspect a supervised training entry point and its arguments on a CPU, run the following dry run.

```bash
pressb-model-train --model diffusion_policy --config configs/model_training/diffusion_policy_pap.yaml --dry-run
```

## Run a simulated evaluation

The [pick-and-place recipe](configs/experiments/pap_walloss_eval.json) selects Wall-OSS and inherits its scene, trajectory, and model settings from [the parent configuration](configs/experiments/pap_walloss_dsrl.json). Set the simulator interpreter, reference trajectory, model checkpoint, normalization statistics, and device settings for your machine. Keep the relative path in `extends` valid. The simulation service needs an NVIDIA GPU, Isaac Sim, and the task assets. Prepare the model environment and weights separately. [Simulator setup](docs/rtx5090_setup.md) · [Model setup](docs/bundled_models.md)

In the runner environment, install the runtime dependencies and validate the configuration.

```bash
python -m pip install -e '.[online-rl]'
pressb validate --config configs/experiments/pap_walloss_eval.json
```

From the repository root, run each command in its own terminal with the corresponding environment active. Start the evaluation once both services are ready.

```bash
pressb inference --config configs/experiments/pap_walloss_eval.json
pressb simulation --config configs/experiments/pap_walloss_eval.json
pressb eval --config configs/experiments/pap_walloss_eval.json --output outputs/pap_eval_01
```

Use `pressb train` with an online RL recipe, `pressb collect` to record trajectories, or `pressb real_robot` with a PiPER hardware configuration. [Framework guide](docs/project_framework.md) · [DPPO](docs/dppo.md) · [Real robots](docs/real_robot.md)

## Repository contents

Keep reusable code, experiment definitions, and setup instructions in Git. Write run-specific results, logs, plots, recordings, and checkpoints under the ignored `outputs/` directory. Supply datasets and pretrained weights separately. Third-party sources, versions, and license notices are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
