# AutoDL node launchers

Set the project, model, checkpoint and interpreter paths in the launchers and their JSON configs for your host. Inspect the resolved recipe before starting any service.

```bash
scripts/autodl/pressb-node pap++-arm6 validate
scripts/autodl/pressb-node pap++-arm6 inference
scripts/autodl/pressb-node pap++-arm6 simulation
scripts/autodl/pressb-node pap++-arm6 train
```

Run each service in its own terminal. The simulation and model nodes must be ready before the trainer connects. Residual recipes require a separately supplied frozen policy checkpoint. The launcher writes local invocation configs and logs under `outputs/autodl_runs/`. See the [framework guide](../../docs/project_framework.md) for distributed configuration.
