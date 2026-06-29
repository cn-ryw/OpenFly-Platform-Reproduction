# 无人机大模型 Paper Review 汇报大纲

建议时长：12-15 分钟

主题：从 aerial VLN 到 UAV VLA，自主无人机导航中的大模型方法与复现

## Slide 1 标题页

题目：无人机大模型导航论文阅读与 OpenFly 复现

内容：

- Topic：无人机大模型 / UAV Vision-Language-Action。
- 阅读论文：OpenFly 和 AutoFly。
- 复现论文：OpenFly。
- 设备：RTX 4060 Laptop 8GB。

讲述重点：

这次分享不是泛泛介绍多模态大模型，而是聚焦“如何让无人机根据视觉和语言输出飞行动作”。

## Slide 2 背景：为什么无人机需要大模型

内容：

- 无人机任务包括搜索救援、巡检、配送、环境监测。
- 传统方法通常拆成感知、定位、规划、控制多个模块。
- VLM/VLA 希望把图像、语言和动作统一起来。

讲述重点：

无人机比地面机器人更难，因为它在三维空间飞行，视角变化大，安全风险更高。

## Slide 3 Topic 定义：UAV VLN 和 UAV VLA

内容：

- VLN：根据语言和视觉导航。
- VLA：根据视觉和语言直接输出动作。
- UAV 场景中的难点：长距离、三维运动、历史观测、避障、目标识别、sim-to-real。

讲述重点：

OpenFly 更偏 UAV VLN benchmark，AutoFly 更偏 UAV VLA autonomous navigation。

## Slide 4 OpenFly 论文概览

内容：

- 平台型论文。
- 集成 UE、GTA V、Google Earth、3D GS。
- 构建 100K trajectories、18 scenes。
- 提出 OpenFly-Agent。

讲述重点：

OpenFly 的价值在于把数据生成、仿真、模型、评估做成一套平台。

## Slide 5 OpenFly-Agent 方法

内容：

- 基于 OpenVLA。
- 输入当前帧和历史关键帧。
- Keyframe selection：动作转折 + landmark grounding。
- Visual token merging：压缩历史帧 token。
- Memory bank K=2。

讲述重点：

OpenFly-Agent 的核心不是简单多塞几张图，而是选择和压缩关键历史观测。

## Slide 6 OpenFly 论文结果

内容：

| Split | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| test-seen | 93m | 34.3% | 64.3% | 24.9% |
| test-unseen | 154m | 22.6% | 56.2% | 19.1% |

Ablation：

- OpenVLA baseline SR 2.3%。
- History + VTM SR 16.6%。
- KS + VTM SR 34.3%。

讲述重点：

历史帧、关键帧选择和 token merging 对 aerial VLN 的提升很明显。

## Slide 7 本地 OpenFly 复现目标

内容：

- 不做训练。
- 不下载完整 100K 数据集。
- 不完整复现 18 场景 benchmark。
- 在 RTX 4060 Laptop 8GB 上做 4bit inference 和单场景 AirSim smoke eval。

讲述重点：

复现目标是受设备约束下验证核心链路，而不是宣称完整 benchmark 复现。

## Slide 8 本地复现链路

内容：

- 修复 `train/extern/hf` custom class import。
- 4bit 加载 `IPEC-COMMUNITY/openfly-agent-7b`。
- 三帧图像 + prompt 输入。
- `predict_action(..., unnorm_key="vlnv1", do_sample=False)`。
- AirSim `env_airsim_16` 闭环执行。
- 记录 NE/SR/OSR/SPL、日志、显存、hash。

讲述重点：

这部分说明我们不是只看论文，而是把模型实际跑起来，并做了可审计记录。

## Slide 9 本地复现结果

内容：

| Scope | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| local 4bit `env_airsim_16` 30 samples | 76.457 | 16.67% | 56.67% | 13.46% |

补充：

- 30 条 AirSim eval 无 OOM、无 Traceback、无图像错误。
- PyTorch peak allocated 约 4.5GB。
- 代表案例图：`demo_results/review_assets/airsim_env16_30_cases/case_gallery.png`。

讲述重点：

OSR 说明模型经常能接近目标区域，但 SR/SPL 偏低说明 stop 时机和路径效率仍不稳定。

## Slide 10 失败分析：action 9 偏多

内容：

- 本地预测 action 9 占 86.80%。
- `env_airsim_16` ground-truth 非负动作中 action 9 占 65.72%。
- forward-heavy 是数据先验，但本地偏置更强。

可能原因：

- 4bit 量化。
- max steps 20 小于官方 100。
- 未完整实现 keyframe/grounding。
- prompt 和训练包装可能存在差异。

讲述重点：

失败不是简单 bug，而是数据先验、推理设置和模型能力共同作用。

## Slide 11 AutoFly 论文概览

内容：

- 目标：粗粒度指令下自主导航。
- 方法：pseudo-depth encoder + VLA + two-stage training。
- 数据：13,476 trajectories，1K real-world episodes。
- 指标：SR、CR、PER。

结果：

| 方法 | SR | CR | PER |
|---|---:|---:|---:|
| OpenVLA | 44.0 | 24.5 | 75.1 |
| AutoFly | 47.9 | 21.9 | 77.3 |

讲述重点：

AutoFly 不是详细路线跟随，而是更偏真实自主避障和 sim-to-real。

## Slide 12 OpenFly vs AutoFly

内容：

| 维度 | OpenFly | AutoFly |
|---|---|---|
| 重点 | 平台和 benchmark | 自主控制模型 |
| 指令 | 长地标导航 | 短目标引导 |
| 数据 | 100K, 18 scenes | 13K, 1K real |
| 核心技术 | keyframe + token merging | pseudo-depth + rebalancing |
| 本次状态 | 已复现 inference + smoke eval | 已阅读分析 |

讲述重点：

两篇论文互补：OpenFly 解决平台化和大规模 VLN，AutoFly 解决真实自主控制和几何感知。

## Slide 13 个人评价

内容：

OpenFly：

- 优点：平台完整、数据规模大、可复现性较好。
- 局限：完整 benchmark 成本高，test-unseen SR 仍低。

AutoFly：

- 优点：更接近真实粗粒度任务，pseudo-depth 有实用价值。
- 局限：语言复杂度较低，真实成功率和碰撞率仍有改进空间，复现门槛高。

讲述重点：

无人机大模型目前还处在“可运行、可展示、但距离可靠自主飞行还有距离”的阶段。

## Slide 14 与既有无人机项目的联系

内容：

| 既有项目 | 作用 | 和 OpenFly/AutoFly 的联系 |
|---|---|---|
| Fast-Drone-250 | 实机平台 | 未来真实部署和数据采集载体 |
| VINS-Fusion | 视觉惯性定位 | 提供位姿和历史状态 |
| FAST-LIO2 | 激光惯性建图 | 提供几何地图和安全约束 |
| EGO-Planner | 局部规划避障 | 执行或修正 VLA 高层动作 |
| Diff-Planner | 学习式轨迹生成 | 生成候选轨迹并结合语言目标选择 |

讲述重点：

OpenFly/AutoFly 不替代传统无人机栈，而是放在上层负责语义理解和高层决策；传统 SLAM/规划器负责安全、定位和可执行轨迹。

## Slide 15 后续计划

内容：

短期：

- OpenFly `--max-steps 100` 小样本 ablation。
- prompt wrapping 对比。
- history/keyframe 近似对比。

中期：

- 下载第二个 AirSim 场景，做跨场景 smoke eval。
- 尝试 8bit/BF16 offload 小样本对照。

长期：

- 完整 benchmark。
- 更接近真实无人机部署。

讲述重点：

后续不应该先追求训练，而应该先把评估对齐和失败原因分析做扎实。

补充研究路线：

- VLA + FAST-LIO2/EGO-Planner 安全执行框架。
- SLAM keyframe 驱动的 OpenFly 历史帧输入。
- Fast-Drone-250 小规模 image-language-action 数据采集。
- Pseudo-depth 与 LiDAR/FAST-LIO2 几何信息融合。

## Slide 16 总结

内容：

- 两篇论文展示了 UAV large model navigation 的两条路线。
- OpenFly 提供了平台和 benchmark。
- AutoFly 强调自主避障和伪深度 VLA。
- 本次已完成 OpenFly 本地 4bit inference 和 AirSim 单场景 smoke eval。
- 既有无人机项目为后续真实部署提供 SLAM、规划和实机基础。
- 当前复现是阶段性成功，不是完整 benchmark 复现。

讲述重点：

这次工作可以作为论文阅读加工程复现的阶段性汇报：既理解了论文，也实际跑通了核心链路，并能说明限制和下一步。
