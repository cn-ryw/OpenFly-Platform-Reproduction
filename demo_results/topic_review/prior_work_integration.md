# 与既有无人机项目工作的关联和未来结合方向

## 一句话结论

之前做的 Fast-Drone-250、VINS-Fusion、FAST-LIO2、EGO-Planner、Diff-Planner 等项目，主要解决无人机“能稳定定位、建图、规划和飞行”的问题；OpenFly 和 AutoFly 进一步关注“无人机能否理解视觉和语言任务，并输出导航动作”。两者不是割裂关系，而是可以组成一个分层无人机自主导航系统。

## 关系总览

| 既有项目/模块 | 原本解决的问题 | 与 OpenFly/AutoFly 的连接点 |
|---|---|---|
| Fast-Drone-250 | 实机平台、飞控、传感器、飞行实验 | 可作为未来 VLA 模型真实部署和数据采集平台 |
| VINS-Fusion | 视觉惯性里程计，估计无人机位姿 | 为大模型导航提供连续位姿、轨迹和历史状态 |
| FAST-LIO2 | 激光惯性里程计，建图和定位 | 为 VLA 输出动作提供几何地图和碰撞安全约束 |
| EGO-Planner | 局部轨迹规划、避障和动态可行轨迹生成 | 可执行或修正大模型给出的高层动作/waypoint |
| Diff-Planner | 学习式/生成式轨迹规划 | 可生成候选轨迹，由语言目标和视觉语义选择 |
| OpenFly | 长地标指令 + 历史视觉帧 -> UAV action | 提供语义导航和历史关键帧建模思路 |
| AutoFly | 粗粒度目标 + RGB/伪深度 -> 速度动作 | 提供自主避障、伪深度和 sim-to-real 思路 |

## 技术层次关系

可以把已有工作和 OpenFly/AutoFly 放在同一套系统中理解：

```text
人类语言指令
   ↓
OpenFly / AutoFly / UAV VLA 模型
   ↓
语义子目标 / 高层动作 / waypoint / velocity command
   ↓
EGO-Planner / Diff-Planner
   ↓
安全、动态可行的局部轨迹
   ↓
Fast-Drone-250 实机飞行
   ↑
VINS-Fusion / FAST-LIO2 提供位姿、地图和障碍物信息
```

这里，大模型不一定直接控制电机或底层飞控。更稳妥的方式是：

- 大模型负责理解语言、视觉目标、地标和任务意图。
- SLAM/里程计负责提供可靠状态估计。
- 传统或学习式规划器负责生成安全可执行轨迹。
- 飞控和实机平台负责稳定执行。

## 为什么这种结合有意义

### 1. 解决 OpenFly/AutoFly 的安全问题

OpenFly 本地 smoke eval 中出现了 action 9 forward bias 和 stop 不稳定。如果直接把大模型动作给无人机执行，风险较高。结合 FAST-LIO2 和 EGO-Planner 后，可以对大模型动作进行安全检查：

- 如果前方有障碍，拒绝 forward action。
- 如果轨迹不可行，调用局部重规划。
- 如果接近目标但模型不 stop，可以加入距离和速度约束辅助停止。

这相当于给 VLA 模型加一层 safety shield。

### 2. 让 OpenFly keyframe 更接近真实机器人系统

OpenFly 论文强调 keyframe selection，但本地复现目前主要用最近三帧。已有 VINS-Fusion/FAST-LIO2 经验可以引入更合理的关键帧来源：

- SLAM keyframe。
- 转弯点。
- 速度或航向突变点。
- 回环或重定位相关关键帧。
- 目标物首次出现或显著变大的帧。

这样可以把 OpenFly 的 keyframe-aware 思路和真实 SLAM 系统连接起来。

### 3. 让 AutoFly 的伪深度与真实几何互补

AutoFly 用 Depth Anything V2 从 RGB 生成 pseudo-depth，优点是减少硬件依赖。但如果平台上已经有 LiDAR 和 FAST-LIO2，那么可以研究：

- 用 LiDAR/FAST-LIO2 稀疏深度监督 pseudo-depth。
- 用真实点云验证伪深度是否可靠。
- 在伪深度不确定时切换到传统避障规划。
- 融合 RGB 语义和 LiDAR 几何，提升安全性。

这比单纯 RGB VLA 更适合真实无人机。

### 4. 把实机飞行日志整理成 VLA 数据

Fast-Drone-250 实机实验可以产生图像、IMU、LiDAR、位姿、轨迹和控制指令。如果再配上语言描述，就可以构造小规模 image-language-action 数据：

| 数据 | 来源 |
|---|---|
| 图像 | 前视/机载相机 |
| 位姿 | VINS-Fusion 或 FAST-LIO2 |
| 点云/地图 | FAST-LIO2 |
| 轨迹 | EGO-Planner 或人工飞行 |
| 动作 | 飞控指令或规划器输出 |
| 语言 | 人工标注或 LLM 生成 |

这可以作为未来微调、评估或小型 benchmark 的基础。

## 可形成的研究方向

### 方向一：语言引导的安全无人机导航

问题：如何让无人机理解“飞到红色屋顶建筑附近”“绕过树林到黑色车辆旁边”这类语言目标，同时保证轨迹安全？

方案：

- VLA 模型负责把语言和图像转成高层 waypoint/action。
- FAST-LIO2 提供局部地图。
- EGO-Planner 生成安全轨迹。
- 飞行过程中持续用视觉反馈更新目标。

适合延续 OpenFly 的长文本地标导航能力。

### 方向二：VLA 动作的安全屏蔽与重规划

问题：大模型输出动作不稳定，可能过度 forward 或过早 stop。

方案：

- 建立 VLA action validator。
- 对 forward/turn/up/down/stop 做几何和动力学检查。
- 不安全动作由 EGO-Planner 或 Diff-Planner 替换。
- 记录被替换动作，形成 failure dataset 反向改进模型。

这个方向和本地 OpenFly action 9 偏多分析直接相关。

### 方向三：SLAM keyframe 驱动的历史视觉记忆

问题：OpenFly 的历史帧选择对性能关键，但简单最近三帧不一定包含真正有用的信息。

方案：

- 用 VINS-Fusion/FAST-LIO2 的 keyframe 作为候选历史帧。
- 结合 landmark detection 或 VLM grounding 选择语义关键帧。
- 输入 OpenFly-Agent 或类似模型，比较最近三帧、均匀采样、SLAM keyframe 的差异。

这可以成为一个很自然的小论文方向。

### 方向四：伪深度、LiDAR 和视觉语言模型融合

问题：AutoFly 只用 RGB pseudo-depth，但真实无人机可能有 LiDAR；如何融合语义和几何？

方案：

- 用 LiDAR/FAST-LIO2 点云作为几何安全层。
- 用 pseudo-depth 提供稠密视觉空间先验。
- 用 VLA 模型理解目标和语义。
- 比较 RGB-only、pseudo-depth、LiDAR-assisted 三种设置。

这个方向可以连接 AutoFly 和 FAST-LIO2。

### 方向五：真实无人机 VLA 数据采集与小规模 benchmark

问题：大规模 OpenFly/AutoFly 数据集很难完整复现，但可以做一个小规模真实数据闭环。

方案：

- 用 Fast-Drone-250 采集 20-50 条短轨迹。
- 每条轨迹包含图像、位姿、规划轨迹、动作和语言任务。
- 用现有 VLA 模型做零样本/少样本评估。
- 分析模型在真实场景中的动作偏置、目标识别和停止行为。

这个方向适合科研训练后续扩展。

## 可以放进汇报的表述

> 我之前做的 VINS-Fusion、FAST-LIO2、EGO-Planner 和 Fast-Drone-250 实机实验，主要解决无人机底层自主飞行能力，包括定位、建图、避障和轨迹执行。OpenFly 和 AutoFly 则把问题推进到语义层：无人机能否理解语言指令和视觉目标，并输出导航动作。未来比较合理的路线不是让大模型直接替代传统飞行栈，而是构建一个分层系统：大模型负责语义理解和高层决策，SLAM 和规划器负责安全约束与可执行轨迹，实机平台负责闭环验证。

## 对导师提问的回答要点

如果老师问“这些论文和你之前做的项目有什么关系”，可以回答：

- 之前项目提供可靠飞行基础，OpenFly/AutoFly 提供语义决策能力。
- 传统规划和 SLAM 可以补足大模型动作不稳定、安全性不足的问题。
- 大模型可以补足传统规划缺少语言理解和开放目标识别的问题。
- 二者结合后的研究方向是 language-conditioned safe UAV navigation。

如果老师问“未来你能做什么”，可以回答：

- 先不直接训练大模型，而是做 VLA + EGO-Planner/FAST-LIO2 的安全执行框架。
- 用 SLAM keyframe 改进 OpenFly 历史帧输入。
- 用 Fast-Drone-250 采集小规模真实 image-language-action 数据。
- 做真实或半真实环境中的 prompt、keyframe、planner safety ablation。

