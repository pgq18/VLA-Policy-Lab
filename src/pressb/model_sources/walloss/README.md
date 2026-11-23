# Bundled Wall-OSS implementation

This directory contains the deployed Wall-X model, flow/FAST training model,
LeRobot data loader, supervised trainer, WebSocket server/client, normalization
tools and CUDA kernel sources. The original `wall_x` and `openpi_client` import
names and checkpoint parameter names are retained. No Wall-X clone is required
at runtime. Checkpoints and normalization files remain external experiment data.

Explicit `device: npu:N` registers `torch_npu` without replacing CUDA APIs.
Non-CUDA execution uses the differentiable ATen implementations in
`wall_x/fusions/portable.py`; activation tensors remain on their device, while
small integer vision/token indices are prepared on CPU. These implementations
preserve the bundled CUDA routing, padding and rotary conventions. The CUDA
extension is loaded lazily for CUDA tensors and retains its original kernels.
NPU and CUDA execution have distinct model-configuration identities; device
ordinals do not change that identity. Native NPU/model/PPO qualification must
still be performed on the selected hardware before long training.

NPU execution defaults to `npu_kv_projection: baddbmm`: existing K/V linear
modules preserve the explicit batch axis and fuse bias addition. This avoids
the measured 910C BF16 skinny-projection row-position variation, without
changing parameter names, weights, dtype, or the trainable parameter set.
`linear` retains the unadapted NPU operation for diagnosis; CUDA and CPU keep
their original linear implementation. Both full and cached attention use the
same adapted K/V modules. The selected mode is included in model identity.

`source_manifest.json` records the source repository revision and every imported
working-tree file hash, including the PressB native-batch, split-encode/decode,
prefix-cache and window-index race fixes that were not in that revision. It is
provenance, not a requirement to keep the source checkout available.

## Environments

Keep model dependencies in their own Python 3.10 environment, separate from
Isaac Sim. The deployed RTX 5090 inference profile uses Torch 2.7.0+cu128,
torchvision 0.22.0+cu128, CUDA toolkit 12.8, Transformers 4.49.0,
Accelerate 1.10.1, PEFT 0.17.1 and BF16. The dependency lists are
`inference/requirements-rtx5090.txt` and `inference/constraints-rtx5090.txt`.
`requirements.txt` is retained verbatim as the older Torch 2.6 upstream profile;
do not use it to overwrite the RTX 5090 environment.

Training additionally needs `requirements-training.txt`: W&B, plotting and the
LeRobot revision recorded by upstream. LeRobot and CUTLASS are third-party
dependencies, not copies of the Wall-X project. The dataset loader imports
`lerobot.datasets`, so an arbitrary older LeRobot version is not interchangeable.
An inference-only environment is intentionally insufficient for dataset training.

```bash
# In your dedicated Wall-OSS Python environment, after installing CUDA Torch:
python -m pip install \
  -c src/pressb/model_sources/walloss/inference/constraints-rtx5090.txt \
  -r src/pressb/model_sources/walloss/inference/requirements-rtx5090.txt
# Only when supervised training is needed:
python -m pip install \
  -c src/pressb/model_sources/walloss/inference/constraints-rtx5090.txt \
  -r src/pressb/model_sources/walloss/requirements-training.txt
```

## Build the CUDA extension

`build_extension.py` builds only the bundled `csrc` files, outside the source
tree. It fetches CUTLASS at the exact recorded commit into the user cache; an
existing clean checkout at that revision can be supplied with `--cutlass-dir`.
The extension is not committed and must match the Python/Torch/CUDA environment.
Neither importing the framework nor starting the source loader downloads or
compiles anything.

```bash
# RTX 5090; for H200 use its matching toolkit/Torch and --arch 9.0.
CUDA_HOME=/usr/local/cuda-12.8 python \
  src/pressb/model_sources/walloss/build_extension.py --arch 12.0 --jobs 4
```

The result goes to
`~/.cache/pressb/walloss/extensions/<Python-Torch-CUDA-architecture>/` by default.
A receipt records source and binary hashes. Successful builds atomically update
`~/.cache/pressb/walloss/extensions/current.json`. Use `--output-dir` to select a
different directory and `PRESSB_WALLX_EXTENSION_DIR` to explicitly select the
extension in an inference process. `--dry-run` prints the build plan without
fetching or compiling; provide `--arch` to avoid probing a GPU.

The retained upstream `setup.py` is historical build source. The supported
PressB build entry is `build_extension.py`, which does not require submodules or
emit a binary beside the model source.

## Entrypoints

The common PressB inference node selects model adapter `walloss`. Supervised
training uses the native `train_qact.py --config CONFIG.yml --seed 42`, exposed
through PressB's `scripts/train_model.py --model walloss`. Distributed execution
uses the same script through Accelerate or torchrun; the trainer owns the
accelerator, gradient accumulation, optimizer, scheduler and checkpoints.

`workspace/piper_pap_20260606/config_qact.yml` is the deployed checkpoint's
**unchanged historical metadata**, including its original paths and resume
settings. It is used to reconstruct that policy, not as a portable training job.
`workspace/piper_pap_template/config_qact.yml` is a fresh-training example with
explicit `/path/to/...` fields and no resume. Supply actual checkpoint, dataset,
statistics and output paths before training. Flow mode does not need the optional
FAST tokenizer assets. Keep statistics paired with the model/dataset; changing
them can change behavior even if model weights remain fixed.

The old Wall-X WebSocket interface remains at `inference/launch_server.py` with
its original arguments, raw observation, RTC and optional RPC capabilities.
Native policy coordinates remain degrees and centimetres; the PressB adapter
exposes radians and metres. `scripts/compute_norm_stats.py` and
`scripts/merge_sharded_weights.py` are retained for data/training workflows.

## Focused verification

Run these in the model environment from the PressB root. The CPU suite covers
real mask construction, batched flow routing, vision isolation, prefix caching,
one tiny flow-training backward/AdamW step and checkpoint round-trip. The tiny
training test uses synthetic tokens/state/actions; it does not read a LeRobot
dataset or claim a successful production training run.

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src/pressb/model_sources/walloss:src/pressb/model_sources/walloss/packages:src/pressb/model_sources/walloss/tests \
python -m unittest discover -s src/pressb/model_sources/walloss/tests -p 'test_*.py'
```

The CPU tests use native differentiable ATen operators and do not require a
CUDA extension. `tests/check_window_index_cuda.py --repeats 20` is a separate
real-GPU kernel race check. Action-head construction no longer allocates CUDA
tensors: CUDA SFT still samples Beta timesteps on CUDA, while NPU samples these
scalar timesteps on CPU before transferring them. DPPO does not use this SFT
Beta sampler. Imported-source adaptations and support files are individually
recorded in the source manifest.

## Source notices

The source checkout has no top-level LICENSE or COPYING file. Its `setup.py`
contains a BSD classifier but no corresponding license text. Existing source
headers are preserved; no replacement license is invented here. The original
CUTLASS license is copied in `licenses/CUTLASS-LICENSE.txt`; downloaded CUTLASS
retains its own notices. Model weight licenses remain separate from code.
