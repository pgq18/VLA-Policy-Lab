# RTX 5090 simulation setup

The RTX 5090 installer pins Python 3.11, Isaac Sim 5.0.0, Isaac Lab v2.2.0, and the CUDA 12.8 build of PyTorch 2.7.0. Run the commands below from the repository root.

## Install the environment and prepare the assets

Use a workstation with Conda, an NVIDIA driver that supports the RTX 5090, and network access to the packages and scene assets. The installer creates `.conda/envs/pressb` in the repository by default. If it cannot find Conda, set `CONDA_BIN` to the absolute path of the Conda executable.

```bash
bash scripts/setup_env_5090.sh

.conda/envs/pressb/bin/python scripts/fetch_assets.py
PYTHONPATH=.cache/usd-inspect \
  .conda/envs/pressb/bin/python scripts/prepare_wrist_asset.py
.conda/envs/pressb/bin/python scripts/fetch_assets.py --check-only
```

`fetch_assets.py` downloads the assets and verifies them against the source manifest. `prepare_wrist_asset.py` builds the wrist assembly using the standalone USD tools installed in `.cache/usd-inspect`. When launching Isaac, use the USD libraries in the simulation environment. Model inference can run in a separate Python environment. [Model setup](bundled_models.md)

## Run the elevator scene

Select an Isaac/Vulkan GPU index and an output directory. This example runs the button-press task on GPU 0 in headless mode using `configs/scene.json` and captures RGB-D frames from the wrist and global cameras.

```bash
bash scripts/run.sh --headless --gpu 0 --video --output outputs/elevator_example
.conda/envs/pressb/bin/python scripts/check_results.py outputs/elevator_example
```

`check_results.py` checks the button sequence, physical contact, and image streams against the run's report, trajectory, and camera records. The output directory contains the scene settings in `report.json`, the measured trajectory in `trajectory.npz`, and a reusable scene snapshot in `scene.usda`. The two camera streams are in `wrist_camera/` and `global_camera/`.

To inspect the scene on a graphical desktop, leave the window open with `--hold`.

```bash
bash scripts/run.sh --gpu 0 --hold --output outputs/elevator_gui
```

## Collect trajectories in parallel

Use the scene snapshot from the headless run to collect two trajectories. Choose a fresh output directory so the trajectories, camera metadata, and collection settings stay together.

```bash
bash scripts/collect_dataset.sh --gpu 0 --num-envs 2 --fps 30 \
  --episodes-per-task 1 --max-episodes 2 \
  --config configs/dataset_panel_randomized.json \
  --snapshot outputs/elevator_example/scene.usda \
  --output datasets/elevator_example
.conda/envs/pressb/bin/python scripts/audit_raw_dataset.py \
  datasets/elevator_example --episodes-per-task 1 --allow-partial
```

For training or evaluation, select the simulation and inference services in the experiment configuration. [Framework configuration](project_framework.md) · [Multi-node launch](configurable_rl_nodes.md)
