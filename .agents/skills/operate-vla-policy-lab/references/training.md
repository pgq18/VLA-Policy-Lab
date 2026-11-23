# Policy training and evaluation

Decide which parameters to update and which data to use to measure performance. Use supervised fine-tuning (SFT) to fit a policy to demonstrations with action labels. Use `eval` to measure the performance of existing weights. To optimize a policy with environment feedback, use `train` and set `run.method` in the experiment JSON according to the table. Run all commands from the repository root.

| Goal | Method and behavior | Example recipe |
|---|---|---|
| Evaluate base or SFT weights | `base` runs evaluation only. Set the weights path in `model.config`. | [PAP baseline](../../../../configs/experiments/pap_walloss_eval.json) |
| Learn the starting point for decoding | `dsrl` uses SAC to learn the initial noise while keeping the action decoder frozen. | [PAP DSRL](../../../../configs/experiments/pap_walloss_dsrl.json) |
| Correct decoded actions | `action_residual` uses SAC to learn an action offset. If you configure `run.frozen_dsrl`, the decoder uses noise from the specified frozen policy before the offset is added. | [Base model with a residual](../../../../configs/experiments/pap_walloss_residual.json) · [Frozen DSRL with a new residual](../../../../configs/experiments/pap_walloss_residual_stochastic_dsrl_coldstart_300k.json) |
| Learn noise and action offsets together | With `joint_residual`, a single SAC actor outputs both. In some filenames, “joint” refers to joint coordinates. Check the resolved `method` to identify the algorithm. | [Elevator VLA-JEPA joint training](../../../../configs/experiments/elevator_vlajepa_joint_residual.json) |
| Fine-tune the denoising action expert | `dppo` samples denoising trajectories from the current expert, uses PPO to update the expert and value network, and publishes the new expert version to the inference service. | [PAP Wall-OSS DPPO](../../../../configs/experiments/pap_walloss_dppo.json) |

## Demonstration data and SFT

Choose a model-native YAML configuration from `configs/model_training/`. The `--model` argument to `scripts/train_model.py` accepts `walloss`, `vlajepa`, `diffusion_policy`, and `pi05`. For VLA-JEPA, choose the `--profile` and corresponding YAML that match the action representation in the dataset, such as `pose9`, `joint6`, or `joint7_generic`. Check camera keys, joint order and units, gripper width, action horizon, and the episode split between training and validation. Verify paths for the model, normalization statistics, and data. [Training templates and model entry points](../../../../docs/bundled_models.md)

For the π₀.₅ cube-stacking example, use the dry run below to check the training entry point. Before launching training, edit the YAML to set local paths for the data, initial weights, tokenizer, and a new output directory. Then run the `torchrun` command in the model environment. For a single GPU, use `python scripts/train_model.py` with the same training arguments.

```bash
sft=configs/model_training/pi05_cube_stack.yaml
python scripts/train_model.py --model pi05 --config "$sft" --dry-run
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_model.py --model pi05 --config "$sft"
```

In multi-device π₀.₅ training, each process selects one visible device using `LOCAL_RANK`. CUDA uses NCCL and Ascend uses HCCL. The nominal global batch size is the per-device batch size × number of ranks × gradient accumulation steps. `--dry-run` resolves the training entry point and arguments only. Pass the native option `-- --prepare-only` to also check π₀.₅ data and normalization. Use the resulting `checkpoints/last/pretrained_model` for π₀.₅ inference or to initialize another SFT run. [π₀.₅ configuration and outputs](../../../../docs/pi05.md) · [Ascend multi-device configuration](../../../../docs/ascend_pi05.md)

## Online training and evaluation

In the experiment JSON, set `model.config.device` for inference, `task.config.gpu_ids` for simulation backends that support it, and `run.device` for the learner. The CLI option `--device` overrides only the learner device. Across machines, use device indices local to each host and service addresses reachable from the trainer. For the PAP DSRL example below, check the external paths, ports, and devices in the recipe. Open separate terminals for the model, Isaac, and RL environments and set the same `recipe` in each. Start the trainer once both services are ready. [Node and service configuration](../../../../docs/configurable_rl_nodes.md)

```bash
recipe=configs/experiments/pap_walloss_dsrl.json
python scripts/run_experiment.py validate --config "$recipe"
python scripts/run_experiment.py inference --config "$recipe"       # model terminal
python scripts/run_experiment.py simulation --config "$recipe"      # Isaac terminal
python scripts/run_experiment.py train --config "$recipe"           # learner terminal
```

For a standalone evaluation, use `eval --config "$recipe" --resume "$rl_checkpoint" --episodes 100 --seed 43`, where `rl_checkpoint` points to a complete RL checkpoint from that recipe. Use the `base` recipe in the table to evaluate the frozen base model. To compare performance before and after DPPO, use the same DPPO recipe for both evaluations. Evaluate the baseline with a freshly started SFT inference service and pass `--resume` for the post-training evaluation. Keep the evaluation seed, task distribution, sampler, smoothing, execution horizon, and episode count fixed. For DPPO, run at least as many episodes as there are parallel environments. Specify distinct `--output` and `--resolved-dir` paths for each evaluation and check the number of completed episodes in `status.json` and `episodes.jsonl`. [DPPO evaluation](../../../../docs/dppo.md)

The DPPO learner loads a second copy of the model on the training host and uses the device in `run.device`. It inherits the inference service's `model` configuration by default. If the hosts use different file paths, supply a complete model configuration with paths readable on the training host in `run.model`. Keep the base weights, normalization, and action expert settings consistent across both hosts.

For multi-GPU DPPO, generate the **trainer node JSON**, then check it and launch training in the learner environment. The example below uses two GPUs visible to the learner. Start inference and simulation separately with the same recipe. Only rank 0 handles RPC, logging, checkpoints, and publishing the expert. `run.dppo.minibatch_size` is the global minibatch size shared across ranks. [Distributed training entry point](../../../../scripts/run_distributed_dppo.py)

```bash
dppo=configs/experiments/pap_walloss_dppo.json
resolved=outputs/pap_walloss_dppo/resolved_2rank
python scripts/run_experiment.py prepare --config "$dppo" --resolved-dir "$resolved"
python scripts/run_distributed_dppo.py --source-root src --config "$resolved/trainer.json" --check-only
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/run_distributed_dppo.py --source-root src --config "$resolved/trainer.json"
```

## Resuming training and two-stage residual training

- **To resume the same SAC run**, use `train --resume`. The checkpoint must include the replay buffer (`run.checkpoint.include_replay=true`). On resume, the trainer checks the model, task, action, and observation fingerprints, the learner configuration, and the reward and replay semantics. It restores the counters, optimizers, and replay buffer, then starts a new episode.
- **To start the second stage with frozen DSRL**, set `run.frozen_dsrl.checkpoint` to the original DSRL checkpoint and verify its SHA-256 hash and `deterministic` setting. A cold start uses a new residual policy and an empty replay buffer. If stage one used `residual_warmup`, you can set `run.residual_initialize_from` to the same file containing the warmup state and clear `run.checkpoint.resume`. To resume or evaluate stage two later, pass the residual policy's own checkpoint to `--resume` and retain the frozen DSRL checkpoint reference. Clear `residual_initialize_from` in the configuration when resuming training. [DSRL warmup recipe](../../../../configs/experiments/pap_walloss_dsrl_residual_warmup_100k.json) · [Residual initialization recipe](../../../../configs/experiments/pap_walloss_residual_from_warmup_300k.json)
- **To resume DPPO**, use the same model and action expert configuration for the inference service and the learner. A complete DPPO checkpoint includes the expert, value network, optimizers, and random number generator state. The trainer checks the model, task, sampler, and algorithm fingerprints on restore and samples fresh trajectories for new episodes. Use a local file or an HTTP(S) URL from the DPPO checkpoint sink. [DPPO checkpoint storage](../../../../docs/checkpoint_sink.md)
