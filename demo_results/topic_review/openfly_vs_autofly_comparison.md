# OpenFly 与 AutoFly 对比分析

## 总体判断

OpenFly 和 AutoFly 都属于“无人机 + 视觉语言/动作大模型”方向，但它们的问题设定不同：

- OpenFly 更像 aerial VLN 平台和 benchmark：重点是多源仿真数据、长文本地标指令、历史关键帧和标准化评估。
- AutoFly 更像面向真实无人机自主导航的 VLA 控制模型：重点是粗粒度目标、避障、伪深度、sim-to-real 和真实飞行。

因此，两篇论文不适合只按 SR 数值直接比较。它们分别代表了无人机大模型导航的两个方向：平台化 benchmark 和实机自主控制。

## 对比表

| 维度 | OpenFly | AutoFly |
|---|---|---|
| 论文定位 | Aerial VLN comprehensive platform | UAV autonomous navigation VLA model |
| 核心目标 | 建立大规模空中 VLN 数据、平台和 benchmark | 粗粒度指令下自主避障、识别和规划 |
| 指令类型 | 较长地标导航指令，平均长度约 59 | 简短目标/方向指令，平均长度约 12 |
| 语言复杂度 | 词汇量约 15.6K | 词汇量约 147 |
| 数据规模 | 100K trajectories | 13,476 trajectories |
| 真实数据 | 主要强调多源仿真和 real-to-sim | 包含 1K 真实世界 episodes |
| 场景来源 | UE, GTA V, Google Earth, 3D GS | 12 个 AirSim 场景 + 真实飞行 |
| 任务形式 | 根据语言导航到目标并 stop | 根据粗粒度目标自主避障和到达目标 |
| 模型基础 | OpenVLA-based OpenFly-Agent | OpenVLA/Prismatic-VLM 风格 VLA |
| 关键创新 | keyframe selection + visual token merging | pseudo-depth encoder + two-stage training |
| 历史信息 | 强调历史 keyframe memory, K=2 | 更强调当前 RGB 的伪深度空间推理 |
| 动作空间 | 4 DoF aerial action templates | 3D velocity/action vector |
| 评估指标 | NE, SR, OSR, SPL | SR, CR, PER |
| 成功条件 | 最终停止点在目标 20m 内 | 目标 5m 内且朝向误差不超过 15 度 |
| 代表结果 | test-seen SR 34.3%, test-unseen SR 22.6% | overall SR 47.9%, CR 21.9%, PER 77.3% |
| 工程复现难度 | 模型推理可复现，完整 benchmark 资产较重 | 实机/加速/数据链路更难完整复现 |
| 本次任务状态 | 已做 4bit inference + AirSim smoke eval | 已做论文阅读分析，未做工程复现 |

## 方法差异

OpenFly 的模型困难主要来自长距离 VLN：

- 要理解较长的地标路线指令。
- 要记住历史观测。
- 要判断什么时候转向和停止。
- 要避免输入太多历史图像导致 token 冗余。

AutoFly 的模型困难主要来自自主控制：

- 指令短，但环境未知。
- 需要自己避障和规划。
- 需要空间深度感知。
- 需要从仿真迁移到真实无人机。

因此，OpenFly 解决的是“如何在大规模 aerial VLN benchmark 上做语言导航”，AutoFly 解决的是“如何让无人机在粗粒度目标下更自主地飞”。

## 数据差异

OpenFly 的优势是规模和多样性：

- 100K 轨迹。
- 18 场景。
- 多渲染引擎。
- 真实地理区域和 3D GS real-to-sim。

AutoFly 的优势是真实飞行和控制闭环：

- 包含 1K 真实飞行 episodes。
- 使用 SAC agent 和 expert demonstrations 生成控制数据。
- 更重视 obstacle avoidance 和 target seeking 的行为平衡。

## 评估差异

OpenFly 的指标更像 VLN：

- NE 衡量最终误差。
- SR 看最终是否停在目标附近。
- OSR 看是否曾经到过目标附近。
- SPL 看成功和路径效率。

AutoFly 的指标更像机器人控制：

- SR 看是否到达目标并满足朝向。
- CR 看碰撞风险。
- PER 看路径效率。

这也解释了为什么不能直接说 AutoFly SR 47.9% 一定强于 OpenFly SR 34.3%。两者任务、成功阈值和测试集都不同。

## 为什么本次复现选择 OpenFly

选择 OpenFly 做复现更合理，原因如下：

1. OpenFly 是本次 topic 中更完整的平台型项目，和“复现一篇论文”要求更匹配。
2. OpenFly 提供 HuggingFace 模型 `IPEC-COMMUNITY/openfly-agent-7b`，可以在本地做 4bit inference。
3. OpenFly 仓库中有 `train/eval.py` 和 AirSim 配置，能做轻量闭环 smoke eval。
4. AutoFly 更依赖自建无人机、真实飞行数据、远程 4090 推理和部署系统，本地完整复现成本更高。
5. 在 RTX 4060 Laptop 8GB 条件下，OpenFly 的推理链路更容易给出可验证结果。

## 共同趋势

两篇论文共同说明：

- UAV navigation 正在从传统模块化规划，走向 VLM/VLA 端到端动作预测。
- 单纯 RGB 观测不够，历史信息或几何信息都很关键。
- 数据和仿真平台是无人机大模型落地的核心瓶颈。
- 真实部署仍然困难，成功率和碰撞安全还远未达到可靠应用水平。

## 可以在汇报中的结论

OpenFly 和 AutoFly 分别代表无人机大模型导航的两个方向：OpenFly 强调大规模 aerial VLN benchmark 和历史关键帧建模，AutoFly 强调粗粒度目标下的自主避障和伪深度空间推理。本次复现选择 OpenFly，是因为它的模型和 AirSim eval 链路更适合在本地 8GB GPU 条件下做可审计复现；AutoFly 则作为对照论文，用来说明该 topic 正在向真实自主飞行和 sim-to-real 迁移发展。

