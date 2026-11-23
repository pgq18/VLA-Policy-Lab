# Train π₀.₅ on Ascend 910C

The π₀.₅ trainer uses HCCL for distributed NPU training. Each worker selects its visible device by local rank. On Ascend the policy computes its timestep embedding in FP32. The training and checkpoint interfaces are shared with CUDA. [Model inputs and outputs](pi05.md)

## Set up the environment

The installer targets an aarch64 host with Python 3.11, an installed Ascend driver and an initialized CANN environment. Its Python constraints pin PyTorch 2.10.0, torchvision 0.25.0 and torch_npu 2.10.0. Check compatibility with the driver and CANN installed on your host. Start in the repository root with a shell that supports `conda activate`. [Installer](../scripts/ascend/install_pi05.sh) · [Python constraints](../scripts/ascend/constraints-910c.txt)

```bash
export PRESSB_REPO="$PWD"
export PRESSB_ASCEND_ROOT=/path/to/large-volume/policy-training
mkdir -p "$PRESSB_ASCEND_ROOT"/{tmp,cache,models,datasets,runs}
conda create -y -p "$PRESSB_ASCEND_ROOT/env" -c conda-forge python=3.11 pip
conda activate "$PRESSB_ASCEND_ROOT/env"
source /usr/local/Ascend/cann/set_env.sh
export TMPDIR="$PRESSB_ASCEND_ROOT/tmp"
export PIP_CACHE_DIR="$PRESSB_ASCEND_ROOT/cache/pip"
bash "$PRESSB_REPO/scripts/ascend/install_pi05.sh"

export PYTHONPATH="$PRESSB_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PRESSB_ASCEND_ROOT/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
cd "$PRESSB_ASCEND_ROOT"
```

Substitute your toolkit's actual `set_env.sh` path. The installer expects an activated, dedicated Conda prefix. It installs the model dependencies, the NPU extension and the pinned LeRobot runtime.

## Prepare the data and model

Supply a LeRobot v3 dataset with the observation and action schema declared in your training YAML. The example templates use `datasets/cube_stack` or `datasets/tabletop_cleanup`, a local base model and a local tokenizer. Relative paths resolve against the working directory. Copy a template and set its paths, dataset identity, camera mapping, units and output directory for your data. [NPU training templates](../configs/model_training/pi05_ascend)

```bash
hf download lerobot/pi05_base --local-dir models/pi05_base
python "$PRESSB_REPO/scripts/prepare_pi05_tokenizer.py" \
  --destination models/paligemma-tokenizer
cp "$PRESSB_REPO/configs/model_training/pi05_ascend/cube_stack.yaml" train.yaml
```

After editing `train.yaml`, check its episode split, normalization and action horizon.

```bash
python -m pressb.training.pi05 --config train.yaml --prepare-only
```

## Launch the workers

Use `npu-smi info` to choose available devices. The following example maps two physical NPUs to local ranks 0 and 1. Adjust both the visibility list and process count together. In `train.yaml`, use `policy.device: npu:0` and select a per-device batch size that fits the model. The effective batch is the product of the number of workers, per-device batch and accumulation steps.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc-per-node=2 "$PRESSB_REPO/scripts/ascend/hccl_probe.py"
torchrun --standalone --nproc-per-node=2 "$PRESSB_REPO/scripts/train_model.py" \
  --model pi05 --config train.yaml
```

The probe exercises HCCL all-reduce and BF16 backward computation. To check the entire training path on a new installation, copy the YAML to a fresh output directory and set `steps: 2`, `eval_every: 2` and `save_every: 2`. The resulting training log and exported native checkpoint exercise data loading, model training, validation and saving. Use a fresh output directory when launching the full job.

When a single host cannot resolve its own hostname, replace `torchrun --standalone` with `python "$PRESSB_REPO/scripts/ascend/torchrun_local.py" --nnodes=1 --master-addr=127.0.0.1 --master-port=29551` in the worker commands. The wrapper scopes the hostname override to the launcher process. [Local launcher](../scripts/ascend/torchrun_local.py)
