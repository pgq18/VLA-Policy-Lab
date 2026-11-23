# 策略训练与评估

先确认要改哪些参数、用什么数据衡量结果。有带动作标签的示范且要拟合策略时选 SFT。只需测现有权重时选 `eval`。需要用环境反馈优化策略时选 `train`，并按下表设置实验 JSON 的 `run.method`。命令均从仓库根目录执行。

| 目标 | 配置与机制 | 已有配方 |
|---|---|---|
| 测基础模型或 SFT 权重 | `base`，权重路径写在 `model.config`，只做评估 | [PAP 基线](../../../../configs/experiments/pap_walloss_eval.json) |
| 学解码起点 | `dsrl`，SAC 学初始噪声，动作解码器冻结 | [PAP DSRL](../../../../configs/experiments/pap_walloss_dsrl.json) |
| 修正解码后的动作 | `action_residual`，SAC 学动作偏移。配 `run.frozen_dsrl` 时，先用指定的冻结噪声策略解码，再加偏移 | [基础模型加残差](../../../../configs/experiments/pap_walloss_residual.json) · [冻结 DSRL 加新残差](../../../../configs/experiments/pap_walloss_residual_stochastic_dsrl_coldstart_300k.json) |
| 同时学噪声和动作偏移 | `joint_residual`，一个 SAC actor 联合输出两部分。文件名里的 joint 有时只指关节坐标，以解析出的 `method` 为准 | [电梯 VLA-JEPA 联合学习](../../../../configs/experiments/elevator_vlajepa_joint_residual.json) |
| 微调去噪动作专家 | `dppo`，从当前专家采样去噪轨迹，PPO 更新专家和价值网络，向推理服务发布新专家版本 | [PAP Wall-OSS DPPO](../../../../configs/experiments/pap_walloss_dppo.json) |

## 示范数据与 SFT

从 `configs/model_training/` 选原生 YAML。`scripts/train_model.py --model` 支持 `walloss`、`vlajepa`、`diffusion_policy`、`pi05`。VLA-JEPA 需按数据的实际动作表示选 `--profile` 和对应 YAML，如 `pose9`、`joint6`、`joint7_generic`。核对相机键、关节顺序与单位、夹爪宽度、动作时域、训练/验证 episode 划分，以及模型、归一化与数据路径。[训练模板与各模型入口](../../../../docs/bundled_models.md)

以 π₀.₅ 方块堆叠为例，先运行第一条核对调用入口。正式运行前，将 YAML 中的数据、初始权重、tokenizer 和新输出目录换为本机值，再在模型环境中执行第二条。单卡可直接用 `python scripts/train_model.py` 加相同训练参数。

```bash
sft=configs/model_training/pi05_cube_stack.yaml
python scripts/train_model.py --model pi05 --config "$sft" --dry-run
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_model.py --model pi05 --config "$sft"
```

π₀.₅ 多卡每个进程按 `LOCAL_RANK` 选择一张可见卡，CUDA 用 NCCL，Ascend 用 HCCL。名义全局 batch 为每卡 batch × rank 数 × 梯度累积步数。`--dry-run` 只解析分发入口和参数。原生 `-- --prepare-only` 可进一步检查 π₀.₅ 的数据与归一化。训练产出的 `checkpoints/last/pretrained_model` 可供 π₀.₅ 推理或初始化新一轮 SFT。[π₀.₅ 配置与产物](../../../../docs/pi05.md) · [Ascend 多卡配置](../../../../docs/ascend_pi05.md)

## 在线训练与评估

在实验 JSON 中分别指定推理的 `model.config.device`、仿真的 `task.config.gpu_ids`（适用的仿真后端）和 learner 的 `run.device`。CLI 的 `--device` 只覆盖 learner。跨机时用各主机的本地设备编号，设置 trainer 可达的服务地址。下面以 PAP DSRL 为例，先校对模板里的外部路径、端口与设备，然后在模型、Isaac、RL 环境各开一个终端。每个终端都设置同一个 `recipe`，待两个服务就绪后运行 trainer。[节点和服务配置](../../../../docs/configurable_rl_nodes.md)

```bash
recipe=configs/experiments/pap_walloss_dsrl.json
python scripts/run_experiment.py validate --config "$recipe"
python scripts/run_experiment.py inference --config "$recipe"       # 模型终端
python scripts/run_experiment.py simulation --config "$recipe"      # Isaac 终端
python scripts/run_experiment.py train --config "$recipe"           # learner 终端
```

独立评估用 `eval --config "$recipe" --resume "$rl_checkpoint" --episodes 100 --seed 43`，其中 `rl_checkpoint` 指向这份配方训练出的完整 RL 文件。测冻结基础模型用表中的 `base` 配方。DPPO 前后对照用同一 DPPO 配方，基线从新启动的 SFT 推理服务评估，训练后传 `--resume`。固定评估 seed、任务分布、采样器、平滑、执行时域和 episode 数。DPPO 的 episode 数至少覆盖所有并行环境。各次显式指定不同的 `--output`、`--resolved-dir`，检查 `status.json` 和 `episodes.jsonl` 中的完成数。[DPPO 评估](../../../../docs/dppo.md)

DPPO 的 learner 会在训练机另加载一份模型，运行设备取自 `run.device`。默认继承推理端的 `model`。若两机文件路径不同，可在 `run.model` 填入训练机可读取的完整模型配置，保持基础权重、归一化和动作专家配置一致。

多卡 DPPO 先生成 **trainer 节点 JSON**，再在 learner 环境检查和启动。下例用两张 learner 可见卡，推理与仿真仍按配方分别启动。Rank 0 独占 RPC、日志、检查点和专家发布。`run.dppo.minibatch_size` 是各 rank 分摊的全局数量。[分布式入口](../../../../scripts/run_distributed_dppo.py)

```bash
dppo=configs/experiments/pap_walloss_dppo.json
resolved=outputs/pap_walloss_dppo/resolved_2rank
python scripts/run_experiment.py prepare --config "$dppo" --resolved-dir "$resolved"
python scripts/run_distributed_dppo.py --source-root src --config "$resolved/trainer.json" --check-only
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/run_distributed_dppo.py --source-root src --config "$resolved/trainer.json"
```

## 续训和两阶段残差

- **继续同一 SAC 运行**用 `train --resume`。断点需保存 replay（`run.checkpoint.include_replay=true`）。恢复时校验模型、任务、动作与观测指纹、learner 配置、奖励和回放语义。恢复计数器、优化器和 replay 后从新 episode 开始。
- **开始冻结 DSRL 的第二阶段**时，将原始 DSRL checkpoint 填入 `run.frozen_dsrl.checkpoint`，核实其 SHA-256 和 `deterministic` 设置。冷启动用新残差与空 replay。若第一阶段启用了 `residual_warmup`，可设 `run.residual_initialize_from` 为同一份含预热状态的文件，并清空 `run.checkpoint.resume`。以后续训或评估第二阶段时，用残差自身的 checkpoint 作为 `--resume`，保留冻结 DSRL 来源。续训配置中清空 `residual_initialize_from`。[DSRL 预热配方](../../../../configs/experiments/pap_walloss_dsrl_residual_warmup_100k.json) · [残差初始化配方](../../../../configs/experiments/pap_walloss_residual_from_warmup_300k.json)
- **继续 DPPO**时，推理端与 learner 使用相同的模型和动作专家配置。完整 DPPO checkpoint 含专家、价值网络、优化器及 RNG，按模型、任务、采样器和算法指纹恢复，新 episode 重新采样。可用本地文件或 DPPO checkpoint sink 的 HTTP(S) URL。[DPPO 检查点存储](../../../../docs/checkpoint_sink.md)
