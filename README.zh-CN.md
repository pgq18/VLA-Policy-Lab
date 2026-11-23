# VLA Policy Lab

[English](README.md) · **简体中文**

VLA Policy Lab 是一个用于训练和评估机器人策略的 Python 框架。它支持用 LeRobot 示范数据做监督微调，也支持在并行 Isaac Sim 仿真环境或 PiPER 真机上做在线强化学习。推理、仿真和训练可以部署在不同机器上。

框架内置 VLA-JEPA、Wall-OSS、Diffusion Policy 和 LeRobot π₀.₅。任务包括电梯与桌面按键、抓取放置、方块堆叠和桌面收纳。在线训练可以冻结策略，学习初始噪声（DSRL）、动作残差或同时学习两者，也可以用 DPPO 微调动作专家。

[快速开始](#快速开始) · [电梯按键教程](#按指定楼层的电梯按钮) · [框架指南](docs/project_framework.md) · [Agent skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

## 快速开始

使用 Python **3.10 或 3.11**。Python 包名和命令名均为 `pressb`。在训练与推理环境中安装模型依赖，在仿真主机上安装 Isaac Sim 依赖，并在强化学习运行器环境中执行 `python -m pip install -e '.[online-rl]'`。[模型安装](docs/bundled_models.md) · [仿真环境安装](docs/rtx5090_setup.md)

```bash
git clone https://github.com/pgq18/VLA-Policy-Lab.git
cd VLA-Policy-Lab
python -m pip install -e .
pressb list
```

## 按指定楼层的电梯按钮

PiPER 机械臂接收 24 至 35 层之间的目标楼层，按下对应按钮。以下三个阶段使用同一套 VLA-JEPA pose9 策略。先用示范数据做监督微调（SFT），再在电梯仿真中做在线强化学习（RL），最后关闭学习，统计完整回合的按键成功率。

### 1. 用示范数据微调 VLA-JEPA

在 SFT 主机上安装 [VLA-JEPA 训练依赖](src/pressb/model_sources/vlajepa/requirements-piper.txt)，并准备 **30 Hz 的 LeRobot 格式电梯示范数据**。每段示范保留到目标按键首次亮起的那一帧（press-prefix），包含全局和腕部相机图像。每层需有 100 个成功回合，共 1,200 个。读取数据时，工具中心点（TCP）的示范位姿会转换为 XYZ 位置和六维旋转表示。已有原始电梯示范数据时，可用 [press-prefix 导出脚本](scripts/export_press_dataset.py)生成训练集。

```bash
export PRESSB_DATASET_ROOT=/data/elevator/press_prefix_lerobot
export PRESSB_PRETRAINED_CHECKPOINT=/models/vla-jepa/model.pt
export PRESSB_QWEN_ROOT=/models/Qwen3-VL-2B-Instruct
export PRESSB_VJEPA_ROOT=/models/vjepa2-vitl-fpc64-256
export PRESSB_RUN_ROOT="$PWD/outputs/elevator/sft"

torchrun --standalone --nproc-per-node=1 scripts/train_model.py \
  --model vlajepa --profile pose9 --config configs/model_training/vlajepa_pose9.yaml \
  -- --run-id pose9
```

[SFT 配置](configs/model_training/vlajepa_pose9.yaml)默认执行 2,000 次优化器更新。训练结束后，使用根据验证集表现选出的检查点做推理。模型主机需要能读取**完整的 SFT 运行目录**，包括 `processor/`、`config.yaml`、数据格式描述、训练元数据以及 `best` 指向的检查点。

```bash
export PRESSB_SFT_CHECKPOINT="$PRESSB_RUN_ROOT/pose9/checkpoints/best"
```

### 2. 选择一种方法做在线强化学习

[电梯任务示例](examples/elevator/)共用同一组 SFT 权重，各自使用一个仿真环境。训练预算为 10,000 次交互，每次执行一个动作块，最多包含 7 个控制步。可按需要更新的策略参数选择方法。

| 配置 | 学习内容 | 训练起点 |
| --- | --- | --- |
| [`dsrl`](examples/elevator/dsrl.json) | SAC 的 actor 选择解码器的初始噪声 | 冻结的 SFT 策略 |
| [`action_residual`](examples/elevator/action_residual.json) | SAC 的 actor 修正 XYZ 动作，每轴修正幅度最多 0.02 米 | 冻结的 SFT 策略 |
| [`joint_residual`](examples/elevator/joint_residual.json) | 同一个 SAC actor 同时调整初始噪声和 XYZ 动作 | 冻结的 SFT 策略 |
| [`residual_after_dsrl`](examples/elevator/residual_after_dsrl.json) | 固定 DSRL actor，从头训练 XYZ 残差 actor 和 critic 网络 | 同一组 SFT 权重**加上第一阶段的 DSRL 检查点** |
| [`dppo`](examples/elevator/dppo.json) | PPO 更新动作专家，并将每一版新权重同步到推理服务 | 推理和训练主机均加载 SFT 权重，采样使用 4 个去噪步 |

**准备仿真环境。** 示例使用 `outputs/rtx5090_eval_comparison/input/` 下固定的训练场景。请准备场景文件 `config.json`、`scene.usda`、`dataset_metadata/meta/collection_metadata.json` 和 `assets/` 资产包，以及 PiPER 资产。示例默认在 GPU 0 上运行仿真，在 GPU 1 上运行推理与学习，各服务通过本机回环地址通信。请按实际部署调整设备编号和服务地址。[节点配置](docs/configurable_rl_nodes.md)

**启动三个进程。** 在每个终端设置第一阶段的 `PRESSB_SFT_CHECKPOINT`、`PRESSB_QWEN_ROOT` 和 `PRESSB_VJEPA_ROOT`，路径应在各自主机上可读。选择一种方法，在三个终端设置相同的方法名和新的运行名称。切换方法时，修改 `METHOD` 并使用对应配置重新启动服务。

```bash
export METHOD=dsrl
export CFG="examples/elevator/$METHOD.json" ELEVATOR_RUN="$METHOD-train-01"
```

| 终端 | 命令 |
| --- | --- |
| 1 · 推理 | `pressb inference --config "$CFG"` |
| 2 · 仿真 | `pressb simulation --config "$CFG"` |
| 3 · 训练 | 先执行 `pressb validate --config "$CFG"`，待两个服务就绪后执行 `pressb train --config "$CFG"` |

训练结束后，从运行器的状态文件读取**实际保存的 RL 检查点路径**。SAC 和 DPPO 的检查点文件名不同。

```bash
export RL_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["last_checkpoint"])' \
  "outputs/elevator/$ELEVATOR_RUN/run/status.json")"
test -f "$RL_CHECKPOINT"
```

使用两阶段方法时，先完成 `dsrl` 训练。将学习器终端得到的检查点路径复制到三个终端，设置以下变量，然后按上表重新启动三个进程。

```bash
export PRESSB_DSRL_CHECKPOINT="$RL_CHECKPOINT"
export METHOD=residual_after_dsrl
export CFG="examples/elevator/$METHOD.json" ELEVATOR_RUN="$METHOD-train-01"
```

此时，固定的 DSRL 策略提供确定性的初始噪声，残差策略从头学习。训练完成后，重新读取本阶段的 `RL_CHECKPOINT`，并保留同一个 `PRESSB_DSRL_CHECKPOINT` 供评估使用。

### 3. 在同一任务上评估 SFT 和 RL 策略

每次评估时，在**三个终端**按下表设置该策略的 `CFG` 和 `ELEVATOR_RUN`。停止上一轮的服务，使用新配置重新启动推理和仿真，再在运行器终端执行评估命令。完成第一阶段后即可评估 SFT 基线。

| 策略 | 配置和新的运行名称 | 运行器命令 |
| --- | --- | --- |
| SFT 基线 | `CFG=examples/elevator/base.json` · `export ELEVATOR_RUN=sft-eval-01` | `pressb eval --config "$CFG" --episodes 10 --seed 43` |
| 训练后的 RL 策略 | `CFG="examples/elevator/$METHOD.json"` · `export ELEVATOR_RUN="$METHOD-eval-01"` | `pressb eval --config "$CFG" --resume "$RL_CHECKPOINT" --episodes 10 --seed 43` |

要配对比较 DPPO 训练前后的表现，还需评估初始动作专家。使用 `dppo.json` 和新的运行名称，重新启动加载 SFT 权重的推理服务，评估命令省略 `--resume`。这样前后两次评估使用相同的 DPPO 采样器。各次评估还应使用相同的随机种子、楼层采样方式、回合超时设置及动作平滑设置。

查看 `outputs/elevator/$ELEVATOR_RUN/run/status.json` 中的 `episodes`、`successes` 和 `success_rate`，确认 10 个回合均已完成。逐回合结果见同目录下的 `episodes.jsonl`。

## 帮助与资源

- **选择运行方式。** 使用 `pressb collect` 记录策略轨迹，或使用 `pressb train` 加载在线 RL 配置。PiPER 真机使用硬件配置，并在连接机械臂的主机上启动 `pressb real_robot`。[在线 RL](docs/configurable_rl_nodes.md) · [DPPO](docs/dppo.md) · [真机配置](docs/real_robot.md)
- **检查配置。** `pressb --help` 列出可用命令，`pressb validate --config "$CFG"` 显示解析后的配置。部署指南中的 `/health` 探测可从客户端核对运行中的服务。
- **先体验电梯场景。** 无需策略权重即可运行脚本规划的按键动作，并检查接触事件和相机图像。[电梯场景操作步骤](docs/rtx5090_setup.md#run-the-elevator-scene)
- **使用 AI agent。** 调用 `$operate-vla-policy-lab` 选择实验配置、分配 GPU 并设置节点通信地址。[仓库内的 skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

仓库收录可复用的代码、配置和指南。各次运行的检查点、日志、图表和录像保存在 Git 忽略的 `outputs/` 下。主要文档和 agent skill 默认使用英文，翻译版使用 `.zh-CN.md` 等语言后缀。

## 项目信息与许可

[贡献者](https://github.com/pgq18/VLA-Policy-Lab/graphs/contributors) · [版本历史](https://github.com/pgq18/VLA-Policy-Lab/commits/main/) · [反馈问题](https://github.com/pgq18/VLA-Policy-Lab/issues)

VLA Policy Lab 的原创代码采用 [MIT 许可证](LICENSE)。第三方代码、模型和机器人资产沿用各自的许可与声明。[第三方来源与许可](THIRD_PARTY.md)

## 致谢

感谢以下项目及其贡献者。

- [Wall-OSS（Wall-X）](https://github.com/X-Square-Robot/wall-x) 和 [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) 提供了本框架适配的机器人策略，用于训练与推理。
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 提供了示范数据集格式和 π₀.₅ 策略实现。
- [DPPO](https://github.com/irom-princeton/dppo) 提供了针对扩散策略的优化方法。[ZPRL](https://github.com/ManUtdMoon/ZPRL) 提供了 SAC 实现，以及在冻结策略上学习调制的方法参考。

README 的结构参考了 [DomPizzie 的模板](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)。
