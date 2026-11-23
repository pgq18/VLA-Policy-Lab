---
name: operate-vla-policy-lab
description: 在 VLA-Policy-Lab（PressB）中选择任务、模型与训练方法，配置并运行监督训练、策略推理、采集或评估。适用于单机或跨机部署仿真／真机、推理与训练节点，设置设备、服务地址并验证连接。
---

# 使用 VLA-Policy-Lab

从仓库根目录操作。包名和 CLI 是 `pressb`。在新机器上取得相同版本的代码，按角色安装依赖。模型权重、数据集、Isaac 资产和机器人设备由使用它们的机器提供。[安装入口](../../../README.zh-CN.md)

## 先选运行方式

| 用户目标 | 入口 | 资源 |
|---|---|---|
| 只提供模型推理 | `pressb inference --config ...` | 模型主机即可对外提供服务 |
| 测基础权重或已有 RL 策略的成功率 | `pressb eval` | 环境 + 推理 + 运行评估的客户端，权重冻结 |
| 记录策略执行的轨迹 | `pressb collect` | 同评估，输出按动作 chunk 记录。Stack/Cleanup 的逐帧 LeRobot 示范另有[采集](../../../scripts/collect_manipulation.py)和[导出](../../../scripts/export_manipulation_dataset.py)入口 |
| 在线训练 | `pressb train` | 环境 + 推理 + 学习器；在实验 JSON 中选 `run.method` |
| 用已有 LeRobot 数据集做 SFT | `pressb-model-train` | 在训练主机用模型专用 YAML 直接训练，只需数据和模型环境 |

四种角色是仿真、真机、推理、训练。一次在线任务从仿真和真机中选一个作为环境服务。负责 rollout 的客户端（训练或评估进程）向推理端请求动作，再把动作发给环境端，读取下一次观测、奖励和终止信息。

## 选任务、模型和配置

1. 运行 `pressb list`，在[实验配置](../../../configs/experiments/)中找任务、模型、动作接口最接近的 JSON。用 `extends` 写子配置；父路径相对当前 JSON，嵌套对象按键合并，列表整体替换。换模型家族时优先从对应家族的配置起步，避免继承上一个适配器的专用字段。
2. `task.plugin` 选择环境（例如 `pap`、`pap_plus_plus`、`cube_stack`、`elevator`）；`task.config` 设置场景、执行时限、并行数和任务所需资产。真机选 `piper_real`，使用 `real_robot` 命令并设一个环境。真机模板默认关闭驱动输出；涉及真实运动时核对现场授权、设备状态与复位流程。[真机配置](../../../docs/real_robot.md)
3. `model.plugin` 选择推理适配器，例如 `pi05`、`walloss`、`vlajepa`、`vlajepa_joint` 或 `diffusion_policy`。按该适配器设置 `model.config` 中的权重、归一化参数和设备。核对观测相机名、动作表示（关节或末端位姿）、单位、夹爪维数、控制频率，以及 `run.rollout.execution_horizon` 是否在模型输出和环境支持的步数内。[模型配置](../../../docs/bundled_models.md)
4. 在 JSON 的 `run` 中写方法、预算、随机种子和输出位置。CLI 的 `train`、`eval`、`collect` 会设置实际运行模式。评估基础策略选 `method: base`，用 `--episodes` 指定局数。在线训练的 `--max-transitions` 按各环境合计的 chunk transition 计数，一条可包含多个控制步。做 DSRL、固定 DSRL 后的动作残差、DPPO、训练恢复或 SFT 时，按[训练方法与权重](references/training.md)选择配置。

## 指定机器和地址

每个节点运行在**启动该进程的主机**。在该机激活对应环境并执行命令；JSON 中的 GPU 编号属于该进程的设备视图。下列字段决定执行位置和连接方式。

| 角色 | 入口 | 关键位置 |
|---|---|---|
| 仿真 | `pressb simulation` | `task.config` 中任务支持的 GPU 参数；PAP 示例使用 `gpu_ids`。监听 `nodes.simulation.server` |
| 真机 | `pressb real_robot` | 在连接 CAN 和相机的主机启动，沿用 `nodes.simulation.server` 与 `run.endpoints.simulation` |
| 推理 | `pressb inference` | `model.config.device` 选模型设备，`nodes.inference.server` 设置监听地址，`nodes.inference.service.batch_size` 设置服务批量上限 |
| 训练／评估 | `pressb train` 或 `pressb eval` | `run.device` 或 CLI `--device` 选学习器／运行器设备，`run.endpoints` 指向两个服务。DPPO 学习器还要能加载模型 |

**跨机器部署、同机多卡或排查连通性**时，使用[部署步骤和健康检查](references/deployment.md)，其中有可直接解析的三机 JSON 和按主机拆开的命令。

## 开始与验收

- 在运行器环境执行 `pressb validate --config "$CFG"`，检查最终的 `factories` 和 `nodes`。需要导出配置时用 `pressb prepare --config "$CFG" --resolved-dir "$OUT/resolved"`。两者都不启动服务。用户只要求配置或预检时，到此即可交付。
- 用户要求实际运行时，分别启动推理和所选的环境服务，从运行器主机读取两个服务的 `/health`，核对模型、任务、并行数和执行契约后再运行客户端。单纯启动推理服务时只检查推理端。
- 使用新的 `outputs/<运行名>/run` 保存本次产物。评估结束后从 `status.json` 和 `episodes.jsonl` 核对结束原因、实际完成局数和成功数；对比策略时使用相同的任务分布、种子、时限和动作执行设置。
