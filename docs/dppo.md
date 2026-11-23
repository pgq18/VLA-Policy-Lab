# Diffusion Policy and action-expert reinforcement learning

DPPO fine-tunes a policy's action expert from simulated rollouts. The learner optimizes the sampled denoising transitions with PPO and fits a value network to rewards from the task. The observation encoder stays frozen. The Diffusion Policy bundle includes the upstream temporal UNet, cosine schedule and reverse diffusion equations. [Bundled source and MIT license](../src/pressb/model_sources/diffusion_policy/)

| Model plugin | Trainable action parameters | Transition kernel |
|---|---|---|
| `diffusion_policy` | Conditional temporal UNet | DDPM or subsampled DDIM |
| `vlajepa`, `vlajepa_joint` | Flow action head and state/action projections | Stochastic Euler flow |
| `walloss` | Action-route MoE MLPs and action projections | Stochastic Euler flow |
| `pi05` | Action-expert transformer, timestep MLPs and action projections | Stochastic Euler flow with descending time |

## Train a Diffusion Policy from demonstrations

In the model environment, install the diffusion-policy dependencies, then copy the training YAML and set the dataset, state and action units, output path and training device. The PAP example uses six joint coordinates and one gripper coordinate. It encodes images with a ResNet18 and spatial softmax, and learns instruction embeddings for the declared task strings. [Supervised training configuration](../configs/model_training/diffusion_policy_pap.yaml)

```bash
python -m pip install -e '.[diffusion-policy]'
python scripts/train_model.py --model diffusion_policy --config /path/to/training.yaml --dry-run
python scripts/train_model.py --model diffusion_policy --config /path/to/training.yaml
```

The reader splits by episode before fitting normalization on the training set. Tail actions are padded within the same episode. Validation uses original images and fixed diffusion noise and timesteps. The trainer writes `best.pt` by validation denoising loss and `last.pt` at the configured saves. Task success is measured in a subsequent rollout.

## Configure a DPPO run

Choose a template for the task and policy. Set the SFT checkpoint, simulator interpreter, devices, endpoints and a fresh output directory. For example, the PAP Diffusion Policy recipe takes a reset trajectory from a compatible PiPER pick-and-place dataset. Generate that input in the dataset environment, and point `task.config.trajectory` to the resulting `episode_000000.npz`. [Experiment templates](../configs/experiments)

```bash
python scripts/analyze_pap_dataset.py --dataset /path/to/pap_dataset \
  --output outputs/pap_dataset_audit --episodes 0
```

The learner and inference node must agree on the pretrained model, action expert and sampler. The number of executed actions per chunk must fit the model horizon. The following fields control how each rollout is scored and optimized.

| Setting | Effect |
|---|---|
| `nodes.inference.service.action_expert` | Selects the denoising kernel, step count and trainable parameters. The runner uses the same action-expert contract. |
| `run.dppo.logprob_reduction` | `upstream_mean` clamps per-coordinate log probabilities and averages over the executed action prefix. `exact_sum` sums all latent channels and horizon positions. |
| `run.dppo.critic_warmup_iterations` | Number of initial full rollout/update rounds used to train only the value network. |
| `run.rollout.deterministic_eval` | Uses kernel means during evaluation when enabled. Keep the sampler, including DDIM `eta`, consistent across evaluated policies. |
| `run.dppo.bootstrap_truncated` | Determines whether a time limit uses the terminal observation for bootstrapping. True terminal transitions end the return. |

During training, sampling and scoring use the same standard deviations, including a positive final-step floor. The configured denoising-step count covers the full chain. Neural evaluation can use BF16, while likelihood calculations and master parameters use FP32. Wall-OSS can fix batch and context shapes with `transition_microbatch_size` and `transition_context_length`. For the optional `frozen_prefix_cache`, bound storage with `prefix_cache_max_entries` and check raw likelihoods, gradients and device memory for your model. [Wall-OSS prefix check](../scripts/qualify_wall_dppo_prefix.py)

## Launching RL and evaluation

From the repository root, validate your edited template. In separate terminals, run the simulator in its Isaac environment, inference in the model environment and the trainer in its RL environment.

```bash
python scripts/run_experiment.py validate --config /path/to/dppo.json
python scripts/run_experiment.py simulation --config /path/to/dppo.json
python scripts/run_experiment.py inference --config /path/to/dppo.json
python scripts/run_experiment.py train --config /path/to/dppo.json --seed 42
```

For each rollout the learner requests an action chain from one inference policy version. The simulator executes the configured prefix and returns rewards, durations and terminal flags. Before optimizing, the learner verifies the sampled log probabilities. It publishes the new expert weights by RPC, and inference verifies the tensor payload before acknowledging the next version.

For a baseline evaluation, start inference with the SFT weights. For a trained evaluation, use `--resume` with a full DPPO checkpoint. Compare the two under the same held-out seed, episode count, task distribution, sampler, execution horizon and smoothing. Use separate output directories and sufficient `max_transitions` for the requested episodes.

```bash
python scripts/run_experiment.py eval --config /path/to/dppo.json \
  --episodes 100 --seed 43 --output outputs/baseline_eval
python scripts/run_experiment.py eval --config /path/to/dppo.json \
  --episodes 100 --seed 43 --resume /path/to/checkpoint.pt --output outputs/trained_eval
```

`status.json` gives the final run state. `episodes.jsonl` records completed episodes. Inspect `metrics.jsonl`, `events.jsonl` and `rpc_timing.jsonl` for updates, policy publication and service timing. A full training checkpoint includes the expert, value network, optimizer and RNG state. To save it on a separate storage host, configure the checkpoint sink. [Checkpoint storage](checkpoint_sink.md)

For a multi-device learner, materialize the experiment and pass its resolved trainer JSON to the distributed entry point. Use a global minibatch divisible by the number of ranks. Rank zero owns simulation and inference RPC, logging, saving and expert publication.

```bash
python scripts/run_experiment.py prepare --config /path/to/dppo.json --resolved-dir outputs/resolved_dppo
torchrun --standalone --nproc-per-node=2 scripts/run_distributed_dppo.py \
  --source-root src --config outputs/resolved_dppo/trainer.json
```
