# Bundled model inference and supervised training

VLA-Policy-Lab provides four model families through the `pressb` Python package. Each family has a supervised-training entry point and an inference adapter. Supply your own weights, backbone assets and datasets. The bundled source manifests identify the imported revisions and local changes. [Source and license notices](../THIRD_PARTY.md)

| Model | Inference and training components | Training configuration |
|---|---|---|
| Wall-OSS | Wall-X model, supervised trainer, LeRobot reader, flow action expert and CUDA extension | `walloss_pap.yaml` |
| VLA-JEPA | PiPER pose and joint-space policies, single-task and multitask trainers | `vlajepa_<profile>.yaml` |
| Diffusion Policy | Visual conditioning, temporal UNet and DDPM/DDIM sampling | `diffusion_policy_pap.yaml` |
| π₀.₅ | LeRobot policy and processors, PiGemma action expert and flow-matching objective | `pi05_cube_stack.yaml` |

The model implementations live in `src/pressb/model_sources`. Node adapters live in `src/pressb/models`, and training launchers live in `src/pressb/training`. [Training templates](../configs/model_training)

## Run inference

Choose an experiment that pairs a model with your task. Configure the local checkpoint, any backbone or normalization assets, the device and the service endpoint in a copy of the JSON. Validate the template in the runner environment, then start inference in the model's environment. [Experiment templates](../configs/experiments)

```bash
python scripts/run_experiment.py validate --config /path/to/experiment.json
python scripts/run_experiment.py inference --config /path/to/experiment.json
```

A service uses its bundled source by default. Set `model.config.repo` for a Wall-OSS checkout or `model.config.vla_repo` for VLA-JEPA to select a different source tree. For training, these two families accept `--source-root`. Start a fresh process when switching a model family to a different source tree. Checkpoint compatibility depends on the identity of the loaded model.

Wall-OSS also needs its native extension built for the environment's CUDA toolkit and GPU architecture. For example, run the following in the Wall-OSS environment with a CUDA toolkit for an RTX 5090. Substitute the toolkit location and compute capability for another GPU. The builder uses the pinned CUTLASS revision and writes the extension to the user cache. `PRESSB_WALLX_EXTENSION_DIR` selects an explicit build directory.

```bash
CUDA_HOME=/usr/local/cuda-12.8 python src/pressb/model_sources/walloss/build_extension.py \
  --arch 12.0 --jobs 4
```

## Run supervised training

From the repository root, copy the matching YAML and set the model, dataset and output paths. The launcher forwards options after `--` to the model's native trainer. Use `--dry-run` to inspect the configuration path, source and entry point before loading model dependencies. The native trainer checks the dataset contract when it starts.

```bash
python scripts/train_model.py --model walloss --config /path/to/walloss_job.yaml --dry-run
python scripts/train_model.py --model walloss --config /path/to/walloss_job.yaml -- --seed 42
```

Install the Wall-OSS training requirements into the model's environment. Use a new training YAML for supervised learning and the checkpoint's matching model configuration for inference. [Wall-OSS training requirements](../src/pressb/model_sources/walloss/requirements-training.txt)

VLA-JEPA training templates use environment variables for the external assets. For a single task with seven joint/gripper action coordinates, use `joint7_generic`.

```bash
export PRESSB_QWEN_ROOT=/path/to/Qwen3-VL-2B-Instruct
export PRESSB_VJEPA_ROOT=/path/to/vjepa2-vitl-fpc64-256
export PRESSB_PRETRAINED_CHECKPOINT=/path/to/pretrained-model.pt
export PRESSB_DATASET_ROOT=/path/to/lerobot-dataset
export PRESSB_RUN_ROOT=/path/to/new-runs

torchrun --standalone --nproc-per-node=1 scripts/train_model.py \
  --model vlajepa --profile joint7_generic \
  --config configs/model_training/vlajepa_joint7_generic.yaml \
  -- --run-id my_joint_policy
```

The generic reader expects a single-task LeRobot v3 export with at least two episodes, `up` and `wrist` cameras at 30 Hz, and an action and unit declaration in `meta/export_manifest.json`. Split episodes before frame sampling. The units are degrees for the six joints and centimetres for full gripper width. The profiles `pose9`, `joint6`, `joint7`, `joint6_multitask` and `joint7_multitask` each use their own dataset contract and matching YAML.

For a Diffusion Policy or π₀.₅ model, the same launcher accepts `--model diffusion_policy` or `--model pi05`. The respective guides cover data preparation, environment setup and checkpoint loading. [Diffusion Policy guide](dppo.md) · [π₀.₅ guide](pi05.md)

## Image augmentation

Training uses the shared `image_transforms` implementation. Validation uses the original frames. Set `image_transforms: {enable: false}` in a training YAML to disable augmentation. The default preset samples up to three transformations without replacement.

| Transform | Default range |
|---|---|
| Brightness, contrast and saturation | Each factor from 0.8 to 1.2 |
| Hue | −0.05 to 0.05 |
| Sharpness | 0.8 to 1.2 |
| Affine | Rotation up to 3° in either direction and translation up to 5% of each image axis |

To customize the preset, set `max_num_transforms`, `random_order` and the entries under `tfs` in the training YAML. A sample's temporal frames share the transformation drawn for that camera. Augmentation is keyed by seed, epoch, sample and camera. The state and action labels retain their original units and values.
