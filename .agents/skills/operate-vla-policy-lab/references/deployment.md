# 一个实验分到三台机器

[示例配置](../assets/pap-walloss-lan-eval.json)使用 PAP 仿真、Wall-OSS 原始策略和 CPU 评估客户端，默认一个并行环境、10 局。推理机加载权重，仿真机加载初始轨迹和场景，评估机通过网络运行 rollout。命令均从各自仓库根目录执行。

### 填写部署参数

在每个要读取实验 JSON 的终端设置以下变量。把示例值换成本次部署的值。主机名需由运行器解析，并能在对应服务机上用于监听。单机部署可将两个主机名均设为 `127.0.0.1`。

```bash
export CFG=.agents/skills/operate-vla-policy-lab/assets/pap-walloss-lan-eval.json
export VLA_INFER_HOST=model-host
export VLA_SIM_HOST=sim-host
export VLA_WALL_MODEL_DIR=/srv/models/piper_pap
export VLA_SIM_PYTHON=/srv/conda/envs/pressb-sim/bin/python
export VLA_PAP_TRAJECTORY=/srv/data/pap/episode_000000.npz
```

解析器在**各进程启动时**展开整个配置的 `${env:NAME}`。每个终端都需要这些变量的值；模型目录及其中的 `norm_stats.json` 由推理机读取，仿真解释器和与场景 episode 对应的轨迹由仿真机读取。安装模型环境和仿真资产分别参阅[模型部署](../../../../docs/bundled_models.md)与[Isaac 安装](../../../../docs/rtx5090_setup.md)。复制示例到其他目录时，同步调整 `extends` 到父 JSON 的相对路径。

### 验证，启动，再从客户端连接

1. **在评估／训练机验证**。在具备运行器依赖的环境中执行以下命令。核对解析出的 `nodes.inference.server`、`nodes.simulation.server` 和 `nodes.trainer.endpoints`。静态验证能在 CPU 上完成；模型文件、相机契约和网络可达性还要在相应进程启动后验证。

   ```bash
   pressb validate --config "$CFG"
   pressb prepare --config "$CFG" --resolved-dir outputs/pap-lan/preflight
   ```

   `prepare` 写出 `inference.json`、`simulation.json`、`trainer.json` 和 `manifest.json`。再次使用相同目录时内容必须一致；为变更后的配置选新目录。常规 `pressb` 命令的 `--config` 接收原实验 JSON。

2. **在服务机分别启动**。可通过 SSH 登录指定主机，在其本地 Python 环境执行对应的一行。多机使用相同版本的插件和相同实验设置；按各主机实际路径准备外部文件。

   ```bash
   # 在模型主机的 Wall-OSS 环境
   pressb inference --config "$CFG"
   # 在仿真主机的 Isaac 环境
   pressb simulation --config "$CFG"
   ```

3. **从评估／训练机检查服务**。下面的只读探测同时适用于 WS 和 HTTP；核对返回的任务和模型与本次实验一致。正常的在线请求由运行器发起，推理服务与环境服务各自监听一个端口，普通训练器无需监听业务端口。

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

4. **在评估机启动 rollout**。服务就绪后执行以下命令；重复评估时换新的运行目录。改跑 DSRL 或 DPPO 时，从相应训练配方继承模型与服务设置，按[训练参考](training.md)启动。

   ```bash
   pressb eval --config "$CFG" --episodes 10 --seed 43 \
     --output outputs/pap-lan/eval-01 --resolved-dir outputs/pap-lan/eval-01-config
   ```

### 地址、设备和并行度

- `nodes.{inference,simulation}.server.{host,port,transport}` 是服务**监听**的地址。运行器通过 `run.endpoints.inference` 和 `run.endpoints.simulation` **连接**。示例分别使用 `ws://model-host:18131` 和 `http://sim-host:18101`。若监听 `0.0.0.0`，客户端须填实际可达的主机名或 IP。使用 SSH 本地端口转发时可让服务只监听回环，在运行器侧用 `--inference ws://127.0.0.1:本地端口`、`--environment http://127.0.0.1:本地端口` 覆盖连接地址。跨机鉴权可在两个服务的 `server.token_env` 中写 `PRESSB_RL_TOKEN`，由部署环境向三个进程提供相同的值。
- 单机两卡可在仿真进程用 `task.config.gpu_ids: [0]`，推理进程用 `model.config.device: "cuda:1"`。进程各自只暴露一张 CUDA 卡时，模型和 learner 使用各进程可见的 `cuda:0`；Isaac 的渲染及物理设备以仿真后端的实际枚举为准。评估示例的 `run.device` 为 `cpu`，训练时按学习器主机的资源另选设备。
- 调整 PAP 采样并行度时，同时设置 `task.config.num_envs` 和 `run.rollout.num_envs`。共享世界 PAP 每个进程使用一张仿真 GPU。推理吞吐另调 `nodes.inference.service.batch_size`；需要多张仿真卡时使用多服务与仿真池。[节点配置](../../../../docs/configurable_rl_nodes.md)
