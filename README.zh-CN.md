# VLA Policy Lab

[English](README.md) · **简体中文**

VLA Policy Lab 是一个用于训练和评估机器人策略的 Python 框架。它支持用 LeRobot 示范数据做监督微调，也支持在并行 Isaac Sim 仿真环境、PiPER 或 Franka Research 3 真机上做在线强化学习。推理、仿真和训练可以部署在不同机器上。

框架内置 VLA-JEPA、Wall-OSS、Diffusion Policy 和 LeRobot π₀.₅。任务包括电梯与桌面按键、抓取放置、方块堆叠和桌面收纳。在线训练可以冻结策略，学习初始噪声（DSRL）、动作残差或同时学习两者，也可以用 DPPO 微调动作专家。

[安装](#安装) · [示范数据微调](#用示范数据微调) · [在线强化学习](#在线强化学习训练) · [评估](#评估) · [框架指南](docs/project_framework.md) · [Agent skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

## 快速开始

### 安装

使用 Python **3.10 或 3.11**。Python 包名和命令名均为 `pressb`。在训练与推理主机上安装模型依赖，在仿真主机上安装 Isaac Sim。推理环境另需执行 `python -m pip install -e '.[inference-server]'`，运行器环境执行 `python -m pip install -e '.[online-rl]'`。[模型安装](docs/bundled_models.md) · [仿真环境安装](docs/rtx5090_setup.md)

```bash
git clone https://github.com/pgq18/VLA-Policy-Lab.git
cd VLA-Policy-Lab
python -m pip install -e .
pressb list
```

### 用示范数据微调

电梯示例用 VLA-JEPA 训练 PiPER 机械臂按下指令指定的楼层按钮。在 SFT 主机上安装 [VLA-JEPA 训练依赖](src/pressb/model_sources/vlajepa/requirements-piper.txt)。pose9 数据读取器使用 30 Hz 的 LeRobot 示范数据，每段保留到目标按键首次亮起的那一帧（press-prefix），并包含全局和腕部相机图像。24 至 35 层每层需要 100 段成功示范。[从原始示范导出训练集](scripts/export_press_dataset.py)

在 [`sft.yaml`](examples/elevator/sft.yaml) 的 `paths` 中填写数据集、VLA-JEPA 预训练检查点和 Qwen3-VL、V-JEPA2 骨干模型的路径。训练参数和运行 ID 也在该文件中。在仓库根目录运行以下命令。

```bash
torchrun --standalone --nproc-per-node=1 scripts/train_model.py \
  --model vlajepa --profile pose9 --config examples/elevator/sft.yaml
```

默认执行 2,000 次优化器更新。根据验证集表现选出的检查点位于 `outputs/elevator/sft/pose9/checkpoints/best`。推理主机需要能读取**完整的 SFT 运行目录**，包括 `processor/`、训练元数据以及 `best` 指向的检查点。

### 在线强化学习训练

电梯任务的 YAML 配置默认使用一个仿真环境，训练预算为 10,000 次动作块交互，每个动作块最多执行 7 个控制步。在 [`base.yaml`](examples/elevator/base.yaml) 中统一设置部署参数。默认 SFT 检查点指向上述微调输出。如果改了 SFT 运行 ID，或模型、场景资产存放在其他位置，请在这里更新检查点、骨干模型和场景路径。示例从 `outputs/rtx5090_eval_comparison/input/` 读取 `config.json`、`scene.usda`、`dataset_metadata/meta/collection_metadata.json` 和 `assets/`。

默认在 GPU 0 上运行 Isaac Sim，在 GPU 1 上运行推理与在线学习，服务使用本机回环地址。分多台主机部署时，在 `nodes` 中设置服务监听地址，在 `run.endpoints` 中填写运行器连接服务的 URL。在 [`train.yaml`](examples/elevator/train.yaml) 中修改 `extends` **一行**即可选择学习方法，例如 `extends: dsrl.yaml`。所选配置指定了运行名称 `name` 和学习器设备。重复训练时请使用新的运行名称。

| 训练配置 | 更新的策略参数 |
| --- | --- |
| [`dsrl.yaml`](examples/elevator/dsrl.yaml)（默认） | SAC 的 actor 为冻结的解码器选择初始噪声。 |
| [`action_residual.yaml`](examples/elevator/action_residual.yaml) | 残差 actor 修正解码后的 XYZ 动作，每轴修正幅度最多为 0.02 米。 |
| [`joint_residual.yaml`](examples/elevator/joint_residual.yaml) | 同一个 actor 同时调整初始噪声和 XYZ 动作。 |
| [`residual_after_dsrl.yaml`](examples/elevator/residual_after_dsrl.yaml) | 从头训练残差学习器，修正已冻结的 DSRL 策略输出的动作。 |
| [`dppo.yaml`](examples/elevator/dppo.yaml) | PPO 更新使用 4 个去噪步的动作专家，并将新权重同步到推理服务。学习器主机也需要能读取 SFT 模型。 |

在两个终端分别进入对应的服务环境，启动推理和仿真。等两个服务就绪后，在第三个终端验证配置并启动训练。

```bash
# 终端 1（推理）
pressb inference --config examples/elevator/train.yaml
```

```bash
# 终端 2（仿真）
pressb simulation --config examples/elevator/train.yaml
```

```bash
# 终端 3（学习器）
pressb validate --config examples/elevator/train.yaml
pressb train --config examples/elevator/train.yaml
```

**在 DSRL 之后训练残差策略**时，先完成默认的 DSRL 训练，然后将 `train.yaml` 中的 `extends` 改为 `residual_after_dsrl.yaml`，停止旧服务，再次执行上述命令。学习器会读取已完成的 DSRL 运行的 `run/status.json`，从 `last_checkpoint` 指向的检查点加载并冻结 DSRL 策略。要使用其他运行或特定训练步的检查点，请修改残差配置中的 `run.frozen_dsrl.checkpoint`。

### 评估

**三个终端使用同一份配置**。评估 SFT 基线时使用 [`base.yaml`](examples/elevator/base.yaml)，评估经过在线强化学习的策略时使用 [`eval.yaml`](examples/elevator/eval.yaml)。后者继承 `train.yaml` 中选择的方法，默认加载 `dsrl-train-01` 的最终检查点。要评估其他训练运行，将 `run.checkpoint.resume` 设为该运行完成后的 `status.json` 或指定的 `.pt` 文件。为每次评估设置独立的 `name`。评估两阶段残差策略时，沿用训练时冻结的 DSRL 检查点，并将 `resume` 指向**残差训练**的运行结果。

| 终端 | SFT 基线 | 在线强化学习后 |
| --- | --- | --- |
| 推理 | `pressb inference --config examples/elevator/base.yaml` | `pressb inference --config examples/elevator/eval.yaml` |
| 仿真 | `pressb simulation --config examples/elevator/base.yaml` | `pressb simulation --config examples/elevator/eval.yaml` |
| 运行器 | `pressb eval --config examples/elevator/base.yaml` | `pressb eval --config examples/elevator/eval.yaml` |

每次切换评估配置时重启服务。两份配置默认均运行 10 个回合，随机种子为 43。要配对比较 DPPO 训练前后的表现，先让 `eval.yaml` 继承 DPPO 训练配置，将 `run.checkpoint.resume` 设为 `null`，从 SFT 权重重新启动推理服务，评估基线。再将该字段改回训练所得的 DPPO 检查点，并使用新的运行名称评估训练后的动作专家。

在 `outputs/elevator/<name>/run/status.json` 中查看 `episodes`、`successes` 和 `success_rate`。同目录下的 `episodes.jsonl` 记录了每个回合的结果。

## 帮助与资源

- **选择运行方式。** 使用 `pressb collect` 记录策略轨迹，或使用 `pressb train` 加载在线 RL 配置。PiPER 和 Franka Research 3 真机分别使用对应的硬件配置，并在连接机械臂的主机上启动 `pressb real_robot`。[在线 RL](docs/configurable_rl_nodes.md) · [DPPO](docs/dppo.md) · [真机配置](docs/real_robot.md)
- **检查配置。** `pressb --help` 列出可用命令，`pressb validate --config examples/elevator/train.yaml` 显示解析后的配置。部署指南中的 `/health` 探测可从客户端核对运行中的服务。
- **先体验电梯场景。** 无需策略权重即可运行脚本规划的按键动作，并检查接触事件和相机图像。[电梯场景操作步骤](docs/rtx5090_setup.md#run-the-elevator-scene)
- **使用 AI agent。** 调用 `$operate-vla-policy-lab` 选择实验配置、分配 GPU 并设置节点通信地址。[仓库内的 skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

仓库收录可复用的代码、配置和指南。各次运行的检查点、日志、图表和录像保存在 Git 忽略的 `outputs/` 下。主要文档和 agent skill 默认使用英文，翻译版使用 `.zh-CN.md` 等语言后缀。

## 致谢

感谢以下项目及其贡献者。

- [Wall-OSS（Wall-X）](https://github.com/X-Square-Robot/wall-x) 和 [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) 提供了本框架适配的机器人策略，用于训练与推理。
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 提供了示范数据集格式和 π₀.₅ 策略实现。
- [DPPO](https://github.com/irom-princeton/dppo) 提供了针对扩散策略的优化方法。[ZPRL](https://github.com/ManUtdMoon/ZPRL) 提供了 SAC 实现，以及在冻结策略上学习调制的方法参考。
