# VLA Policy Lab

[English](README.md) | **简体中文**

VLA Policy Lab 是一个机器人策略框架，用于采集轨迹、评估视觉语言动作（VLA）策略，以及通过环境交互训练策略。推理服务、环境服务（仿真或 PiPER 真机）和实验运行器分别在独立进程中运行。实验配置指定任务、模型及各进程的连接方式。

任务插件涵盖按键、PiPER 或 Franka Research 3 的抓取放置、方块堆叠和桌面收纳。模型适配器支持 Wall-OSS、VLA-JEPA、Diffusion Policy 和 LeRobot π₀.₅。在线训练支持 DSRL、动作残差学习和 DPPO。

在 AI agent 中使用 `$operate-vla-policy-lab`，可以按任务选择配置、分配节点和验证跨机连接。[仓库内的 skill](.agents/skills/operate-vla-policy-lab/SKILL.md)

## 快速开始

激活 Python 3.10 或 3.11 环境后，安装核心包。Python 包名和实验命令均为 `pressb`。

```bash
git clone https://github.com/pgq18/VLA-Policy-Lab.git
cd VLA-Policy-Lab
python -m pip install -e .
pressb list
```

`pressb list` 会列出已注册的任务和模型插件。下面的命令可在 CPU 上查看监督训练入口及其参数，`--dry-run` 只打印调用信息。

```bash
pressb-model-train --model diffusion_policy --config configs/model_training/diffusion_policy_pap.yaml --dry-run
```

## 运行仿真评估

[抓取放置评估配置](configs/experiments/pap_walloss_eval.json) 选用 Wall-OSS，从[父配置](configs/experiments/pap_walloss_dsrl.json)继承场景、参考轨迹和模型设置。按本机环境设置仿真 Python 解释器、参考轨迹、模型权重、归一化统计文件和计算设备，并保持 `extends` 的相对路径有效。仿真服务需要 NVIDIA GPU、Isaac Sim 和任务资产，模型环境及权重需另行准备。[仿真环境安装](docs/rtx5090_setup.md) · [模型配置](docs/bundled_models.md)

在实验运行器所在的 Python 环境中安装运行依赖并检查配置。

```bash
python -m pip install -e '.[online-rl]'
pressb validate --config configs/experiments/pap_walloss_eval.json
```

打开三个终端，分别激活对应环境，在仓库根目录执行各自的命令。待推理和仿真服务就绪后，再执行第三条命令启动评估。

```bash
pressb inference --config configs/experiments/pap_walloss_eval.json
pressb simulation --config configs/experiments/pap_walloss_eval.json
pressb eval --config configs/experiments/pap_walloss_eval.json --output outputs/pap_eval_01
```

使用 `pressb train` 配合在线强化学习配置训练策略，使用 `pressb collect` 采集轨迹。PiPER 真机配置通过 `pressb real_robot` 启动硬件服务。[框架说明](docs/project_framework.md) · [DPPO](docs/dppo.md) · [真机配置](docs/real_robot.md)

## 仓库内容

Git 仓库收录可复用的代码、实验配置和使用文档。每次运行生成的结果、日志、图表、录像和检查点统一存放在 Git 忽略的 `outputs/` 目录。数据集及预训练权重需另行准备。第三方组件的来源、版本与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
