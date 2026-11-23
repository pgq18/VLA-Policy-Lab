# Run an experiment across three machines

The [example configuration](../assets/pap-walloss-lan-eval.json) evaluates the base Wall-OSS policy in the PAP simulator with a CPU evaluation client. It defaults to one environment and 10 episodes. The inference host loads the weights, the simulation host loads the initial trajectory and scene, and the evaluation host runs rollouts over the network. Run each command from the repository root on its host.

### Set the deployment parameters

Set the following variables in every terminal that reads the experiment JSON. Replace the example values with those for your deployment. The runner must be able to resolve both hostnames, and each service must be able to bind to its hostname on its own machine. For a single-machine deployment, set both hostnames to `127.0.0.1`.

```bash
export CFG=.agents/skills/operate-vla-policy-lab/assets/pap-walloss-lan-eval.json
export VLA_INFER_HOST=model-host
export VLA_SIM_HOST=sim-host
export VLA_WALL_MODEL_DIR=/srv/models/piper_pap
export VLA_SIM_PYTHON=/srv/conda/envs/pressb-sim/bin/python
export VLA_PAP_TRAJECTORY=/srv/data/pap/episode_000000.npz
```

The parser expands `${env:NAME}` throughout the configuration **when each process starts**, so every terminal needs all of these variables. The inference host reads the model directory and its `norm_stats.json`. The simulation host needs the simulator's Python interpreter and the trajectory for the scene episode. If you copy the example to another directory, update the relative path in `extends` to point to the parent JSON. For setup instructions, see [model deployment](../../../../docs/bundled_models.md) and [Isaac installation](../../../../docs/rtx5090_setup.md).

### Validate the configuration, start the services, and connect the client

1. **Validate on the evaluation or training host.** Run the following commands in an environment with the runner's dependencies installed. Check the resolved `nodes.inference.server`, `nodes.simulation.server`, and `nodes.trainer.endpoints` settings. Static validation can run on CPU. Check the model files, camera contract, and network connectivity once the corresponding processes have started.

   ```bash
   pressb validate --config "$CFG"
   pressb prepare --config "$CFG" --resolved-dir outputs/pap-lan/preflight
   ```

   `prepare` writes `inference.json`, `simulation.json`, `trainer.json`, and `manifest.json`. You can reuse the directory if the contents match. Choose a new directory after changing the configuration. Pass the original experiment JSON to `--config` for regular `pressb` commands.

2. **Start each service on its host.** You can use SSH to log in to each host and run the corresponding command in its local Python environment. Use the same plugin versions and experiment settings across machines. Make external files available at the configured paths on the hosts that use them.

   ```bash
   # On the model host, in the Wall-OSS environment
   pressb inference --config "$CFG"
   # On the simulation host, in the Isaac environment
   pressb simulation --config "$CFG"
   ```

3. **Check both services from the evaluation or training host.** The following read-only probe works over WebSocket and HTTP. Check that the reported task and model match the experiment. During a run, the runner initiates requests. The inference and environment services each listen on a port. A standard trainer requires no listening port for these requests.

   ```bash
   python - <<'PY'
   import os
   from pressb.experiments.config import load_experiment
   from pressb.runtime.client import NodeClient

   nodes = load_experiment(os.environ["CFG"])["nodes"]
   for role in ("inference", "simulation"):
       url = nodes["trainer"]["endpoints"][role]
       client = NodeClient(url, timeout=10, token=os.getenv("PRESSB_RL_TOKEN"))
       try:
           info = client.call("/health")
           assert (info["schema_version"], info["node_type"]) == (2, role)
           print(role, url, info.get("task_id") or info.get("model_id"),
                 info.get("num_envs"), info["action"])
       finally:
           client.close()
   PY
   ```

4. **Start the rollout on the evaluation host.** Run the command below once both services are ready. Use a new run directory for each evaluation. To run DSRL or DPPO, inherit the model and service settings from the corresponding training recipe and follow the [training reference](training.md).

   ```bash
   pressb eval --config "$CFG" --episodes 10 --seed 43 \
     --output outputs/pap-lan/eval-01 --resolved-dir outputs/pap-lan/eval-01-config
   ```

### Addresses, devices, and parallelism

- `nodes.{inference,simulation}.server.{host,port,transport}` specifies where each service **listens**. The runner **connects** using `run.endpoints.inference` and `run.endpoints.simulation`. The example uses `ws://model-host:18131` and `http://sim-host:18101`, respectively. If a service binds to `0.0.0.0`, give the client a reachable hostname or IP address. With SSH local port forwarding, the services can bind to loopback. On the runner, override the endpoints with `--inference ws://127.0.0.1:LOCAL_PORT` and `--environment http://127.0.0.1:LOCAL_PORT`, substituting the local port for each forward. To authenticate across hosts, set `server.token_env` to `PRESSB_RL_TOKEN` on both services and supply the same token value to all three processes.
- On a single machine with two GPUs, you can set `task.config.gpu_ids: [0]` for simulation and `model.config.device: "cuda:1"` for inference. When each process sees only one CUDA GPU, the model and learner each use their own visible `cuda:0`. Isaac uses the rendering and physics device indices reported by the simulation backend. The evaluation example sets `run.device` to `cpu`. For training, choose a device available on the learner host.
- To change the number of parallel PAP environments, set both `task.config.num_envs` and `run.rollout.num_envs`. Shared-world PAP uses one simulation GPU per process. Tune inference throughput separately with `nodes.inference.service.batch_size`. To use multiple simulation GPUs, run multiple services with a simulation pool. [Node configuration](../../../../docs/configurable_rl_nodes.md)
