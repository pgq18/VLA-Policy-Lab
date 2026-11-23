# Running inference, environment and trainer nodes

The inference node keeps the model and encoded observation contexts in memory. The environment node owns reset, action execution, camera capture and task outcomes. The trainer requests features and actions, sends the execution prefix to the environment, and records transitions. Public node contracts include action units, camera names, state shape and control frequency.

## Launch a pick-and-place evaluation

Start from `configs/experiments/pap_walloss_eval.json`, which inherits the task and model settings in `pap_walloss_dsrl.json`. Set the external model, normalization and trajectory paths, GPU devices and environment count for your machine. When reducing parallelism, set both `task.config.num_envs` and `run.rollout.num_envs`. Inference batching has its own limit in `nodes.inference.service.batch_size`. See the model and simulation setup guides for host dependencies. [Model installation](bundled_models.md) · [RTX 5090 simulation setup](rtx5090_setup.md)

Run the first command in the model environment and the second in the Isaac Sim environment, in separate terminals. Run the last command on the trainer host after both services are ready. The examples use paths relative to the repository root.

```bash
python scripts/run_experiment.py inference --config configs/experiments/pap_walloss_eval.json

python scripts/run_experiment.py simulation --config configs/experiments/pap_walloss_eval.json

python scripts/run_experiment.py eval --config configs/experiments/pap_walloss_eval.json \
  --episodes 10 --output outputs/example_eval/run
```

For multiple hosts, set a reachable bind address under each service's `server.host`, or forward the template's loopback ports. Set client addresses in `run.endpoints` or pass `--inference` and `--environment` on the trainer command. The PAP example uses WebSocket port 18131 for inference and HTTP port 18101 for the environment.

## Train or collect

For DSRL, configure the services with `pap_walloss_dsrl.json`, then start the trainer.

```bash
python scripts/run_experiment.py train --config configs/experiments/pap_walloss_dsrl.json \
  --output outputs/example_dsrl/run
```

| Method | What the trainer changes |
| --- | --- |
| `base` | Evaluates the pretrained model with its configured sampling policy. |
| `dsrl` | Trains a policy that selects the initial noise for the frozen action decoder. |
| `action_residual` | Trains offsets on the decoded action chunk. `frozen_dsrl` can supply a separate noise policy. |
| `joint_residual` | Trains one actor to adjust both the decoder noise and the decoded action. |
| `dppo` | Updates a supported action expert from on-policy samples and synchronizes weights with inference. |

Set the execution horizon to a prefix that both the model and task support. The environment returns the duration and termination status of each executed chunk. For DPPO, use a method-specific model and trainer configuration. [DPPO setup](dppo.md)

To evaluate a trained policy, use its matching training configuration and pass `eval --resume /path/to/checkpoint.pt`. For a full DSRL or action-residual training resume, the saved checkpoint must include replay. The trainer checks the connected model and task identities against the checkpoint. [Checkpoint storage](checkpoint_sink.md)

To collect chunk trajectories, run the evaluation services and change the trainer command to `collect`.

```bash
python scripts/run_experiment.py collect --config configs/experiments/pap_walloss_eval.json \
  --episodes 10 --output outputs/example_collection/run
```

The collector records observations, requested action prefixes, rewards, outcomes and durations. By default, the trajectory directory is a sibling of the run directory with suffix `_trajectories`. Set `collection.directory` to select another destination and `collection.include_images` to control image storage.
