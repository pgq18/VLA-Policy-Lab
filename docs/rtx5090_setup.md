# RTX 5090 仿真环境

仓库提供面向 RTX 5090 的安装脚本，固定使用 Python 3.11、Isaac Sim 5.0.0、Isaac Lab v2.2.0 和 PyTorch 2.7.0 的 CUDA 12.8 构建。以下命令均在仓库根目录运行。

## 安装与准备资产

准备可执行的 Conda、支持 RTX 5090 的 NVIDIA 驱动，以及下载软件包和场景资产所需的网络连接。安装脚本默认创建项目内的 `.conda/envs/pressb` 环境。若脚本无法定位 Conda，可通过 `CONDA_BIN` 指定其可执行文件的绝对路径。

```bash
bash scripts/setup_env_5090.sh

.conda/envs/pressb/bin/python scripts/fetch_assets.py
PYTHONPATH=.cache/usd-inspect \
  .conda/envs/pressb/bin/python scripts/prepare_wrist_asset.py
.conda/envs/pressb/bin/python scripts/fetch_assets.py --check-only
```

`fetch_assets.py` 按来源清单下载并核对素材，`prepare_wrist_asset.py` 生成仿真使用的腕部组件。安装脚本将独立的 USD 工具放在 `.cache/usd-inspect`。启动 Isaac 时使用仿真环境自带的 USD。模型推理可在单独的 Python 环境中运行。[模型安装说明](bundled_models.md)

## 运行电梯场景

显式指定 Isaac/Vulkan 设备编号和本次运行的输出目录。下例在设备 0 上无窗口运行，按 `configs/scene.json` 的设置执行按键任务并采集腕部和全局 RGB-D 图像。

```bash
bash scripts/run.sh --headless --gpu 0 --video --output outputs/elevator_example
.conda/envs/pressb/bin/python scripts/check_results.py outputs/elevator_example
```

`check_results.py` 读取此次运行生成的报告、轨迹及相机记录，核对按键顺序、物理接触和图像流。目录中的 `report.json` 保存场景设置，`trajectory.npz` 保存测量轨迹，`scene.usda` 可用于后续采集。两路相机的文件分别在 `wrist_camera/` 和 `global_camera/` 中。

在图形桌面查看场景时，可保持窗口打开。

```bash
bash scripts/run.sh --gpu 0 --hold --output outputs/elevator_gui
```

## 并行采集示例

用上一步生成的场景快照采集两条轨迹。将输出写入新目录，使轨迹、相机元数据和采集设置保持对应。

```bash
bash scripts/collect_dataset.sh --gpu 0 --num-envs 2 --fps 30 \
  --episodes-per-task 1 --max-episodes 2 \
  --config configs/dataset_panel_randomized.json \
  --snapshot outputs/elevator_example/scene.usda \
  --output datasets/elevator_example
.conda/envs/pressb/bin/python scripts/audit_raw_dataset.py \
  datasets/elevator_example --episodes-per-task 1 --allow-partial
```

训练和评估通过配置文件选择仿真服务与模型服务。[框架与配置](project_framework.md) · [多节点启动](configurable_rl_nodes.md)
