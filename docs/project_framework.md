# Framework and experiment configuration

An experiment selects a task, a model and a rollout method. Three processes run the experiment. The environment produces observations and applies actions, the inference service encodes observations and decodes actions, and the trainer coordinates rollouts, learning and checkpoints. Each process can use a different Python environment or host.

## Run a configuration

From the repository root, list the built-in plugins and validate the pick-and-place evaluation example in the trainer environment.

```bash
python scripts/run_experiment.py list
python scripts/run_experiment.py validate --config configs/experiments/pap_walloss_eval.json
```

Validation prints the resolved node settings and factory names. Before launching, set the paths to model weights, normalization statistics and task inputs for your installation. Select a GPU on each host and configure the service addresses. The model guide describes the required external inputs. [Bundled models](bundled_models.md)

```bash
python scripts/run_experiment.py prepare --config configs/experiments/pap_walloss_eval.json \
  --resolved-dir outputs/pap_example/resolved
```

The directory contains `inference.json`, `simulation.json`, `trainer.json` and `manifest.json`. Use a fresh resolved directory after changing settings. Start the `inference` and `simulation` commands in separate terminals, then run `eval` from a third. The online RL guide includes the launch commands. [Running the nodes](configurable_rl_nodes.md)

## Configuration fields

| Field | Purpose |
| --- | --- |
| `task` | Environment factory and settings for scene, robot, cameras, control timing and task outcomes |
| `model` | Policy factory and settings for weights, normalization, action dimensions and device |
| `nodes` | Bind addresses, transports, batch sizes and cache settings for the services |
| `run` | Rollout method, client endpoints, action execution horizon, learner, checkpoints and output |
| `collection` | Optional trajectory directory and whether to include images in a collection |

Experiment recipes accept JSON or YAML (`.yaml` and `.yml`). Use `extends` to inherit a recipe in either format. Parent paths are relative to the declaring file. Nested objects merge by key and lists replace the parent list. In string values, `${project_root}` selects the checkout, `${experiment_dir}` the directory containing that field, and `${experiment_name}` the final merged experiment name. Optional `${env:NAME}` references require the variable on each host that reads the recipe. The [elevator examples](../examples/elevator/) supply their defaults directly in YAML.

For SAC or DPPO evaluation, set `run.checkpoint.resume` to a checkpoint or a completed training run's local `status.json`. A second-stage residual can similarly use a DSRL status file in `run.frozen_dsrl.checkpoint`. Only the learner reads the status file at execution time, so config validation and service startup work before training has finished. Use a specific checkpoint file to select an intermediate step or resume an interrupted run. Existing model, task and learner identity checks still apply.

## Add a task or model

For each of `task` and `model`, select a built-in `plugin` or give a Python `factory` such as `my_tasks.pick_place:create_backend`. Install the factory's package on the host that runs its service. The factory receives the corresponding `config` object.

| Component | Implementation |
| --- | --- |
| Environment service | Owns run, episode and step IDs, request validation and retry handling. The backend implements `describe`, `reset`, `step` and `close`. |
| Shared simulation | `pressb.runtime.parallel.SharedWorldBackend` owns the physics clock, environment cloning and camera rendering. A task defines its reset sampler, scene objects, reward and success conditions. A robot driver maps actions to the articulation. |
| Model adapter | Implements `describe`, `encode_batch`, `decode_batch` and `close`. It owns model loading, normalization, feature extraction and unit conversion. |
| Trainer | Negotiates camera, state, action and timing contracts, then runs evaluation, collection or learning through the same services. |

The built-in plugin map lives in `src/pressb/registry.py`. Native task implementations live under `src/pressb/tasks/`, model adapters under `src/pressb/models/`, and task-independent service code under `src/pressb/runtime/`. A physical PiPER (`piper_real`) or Franka Research 3 (`franka_real`) can provide the environment through the same rollout interface. [Physical robot setup](real_robot.md)
