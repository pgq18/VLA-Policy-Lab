# Streaming DPPO checkpoints

Use a checkpoint sink to save a learner's full training state on a storage host. The learner streams the action expert, FP32 master parameters, optimizer state, value network and RNG state directly to the receiver. Local disk checkpoints are the default. [DPPO training guide](dppo.md)

## Start the storage receiver

On the storage host, choose a directory with enough capacity for each interval checkpoint. The receiver uses the Python standard library and binds to loopback by default. For bearer authentication, set `PRESSB_CHECKPOINT_TOKEN` to the same secret in the receiver and learner environments. [Receiver](../scripts/serve_checkpoint_sink.py)

```bash
python scripts/serve_checkpoint_sink.py \
  --directory /absolute/path/to/checkpoints --port 18391
```

To reach the receiver from a different learner host, open a second terminal on the storage host and forward the learner's loopback port over SSH. Replace `learner-host` with its SSH name.

```bash
ssh -N -R 18391:127.0.0.1:18391 learner-host
```

## Configure the learner

Merge `checkpoint_sink` into the existing `run` object in the experiment JSON. Give each run a unique namespace. Set `max_bytes` and `timeout` to values appropriate for your checkpoint and link. The receiver also accepts `--max-bytes` and `--timeout`.

```json
{
  "run": {
    "checkpoint_sink": {
      "endpoint": "http://127.0.0.1:18391",
      "namespace": "my_training_run",
      "timeout": 1800,
      "max_bytes": 34359738368,
      "token_env": "PRESSB_CHECKPOINT_TOKEN"
    }
  }
}
```

The receiver writes to a temporary file in the destination namespace, checks the byte count and SHA-256, and publishes the completed file atomically. It keeps each interval at `<directory>/<namespace>/step_<transitions>.pt` and rejects an upload to an existing filename. Interrupted or corrupt transfers are discarded. The learner writes the acknowledged URL and checksum to `checkpoints/latest.json`. If an acknowledgement is lost, check the destination before retrying.

## Resume or evaluate

Pass the acknowledged URL to `--resume` when starting a trainer or evaluator. The receiver must be reachable from the learner. The following names illustrate the URL layout. Use the exact namespace and filename from your checkpoint.

```bash
python scripts/run_experiment.py train --config /path/to/dppo.json \
  --resume http://127.0.0.1:18391/checkpoints/my_training_run/step_000100000.pt
```

The learner verifies the content length and SHA-256 before loading the bytes with `weights_only=True`. Allocate host memory for the full checkpoint. The runner also accepts a local checkpoint path. Both forms undergo the normal task, model and sampler compatibility checks. [Checkpoint transport implementation](../src/pressb/runtime/checkpoint_sink.py)
