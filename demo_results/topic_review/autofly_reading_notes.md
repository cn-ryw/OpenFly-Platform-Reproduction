# AutoFly 论文阅读笔记

论文：AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild

来源文件：`/home/ruan/research/AutoFly2602.09657v1.pdf`

文本抽取：`demo_results/topic_review/autofly_paper.txt`

PDF SHA256：`82ca664b7f04e4a00ef455386ebf5d6ffddf277559c6266ba4dfd26db204e33d`

抽取文本 SHA256：`a8b9c4e0496f58ee52cd2e3a8ac07305e132892f7b91d72f02e6e3b76d1180fc`

## 研究问题

AutoFly 关注的是无人机在未知或半未知环境中的自主导航。论文认为，已有 UAV VLN 方法通常依赖非常详细的逐步路线指令，例如先沿路飞、到路口左转、再寻找某个目标。这种设置更像“按路线执行”，但真实无人机任务中，人类往往只能给粗粒度目标，例如“穿过障碍，到达黑色展示板附近”。

因此，AutoFly 想解决的问题不是完整复刻一条语言描述路线，而是在粗粒度语言目标、当前 RGB 观测和初始方向/位置引导下，让模型自己完成避障、目标识别和路径规划。

## 方法框架

AutoFly 是一个端到端 Vision-Language-Action 模型。输入包括：

- RGB 图像观测。
- 简短语言指令。
- 粗粒度方向或位置引导，论文中编码为 initial action。

输出是三维动作向量，和无人机速度控制相关。

模型主要由三部分组成：

1. Vision-Language backbone
   - 基于 Prismatic-VLM/OpenVLA 类似的 VLA 范式。
   - 使用 LLaMA2 7B 作为语言模型基础。
   - 通过视觉编码器和语言 token 做多模态对齐。

2. Pseudo-depth encoder
   - 用 Depth Anything V2 从 RGB 图像生成伪深度。
   - 再通过 depth projector 将深度信息投影到视觉 token 空间。
   - 目的是在没有真实深度相机的情况下增强空间和几何理解。

3. Action de-tokenizer
   - 沿用 OpenVLA 思路，把语言模型输出 token 映射成连续动作。
   - 论文使用 LLaMA2 词表末尾 token 作为动作映射空间。

## 数据集设计

AutoFly 构建了一个面向“自主导航”而不是“逐步指令跟随”的数据集。

关键数字：

| 项目 | 数值 |
|---|---:|
| 总轨迹数 | 13,476 |
| 真实世界轨迹 | 1K |
| 图像-语言-动作 triplets | 2.5M+ |
| 仿真场景 | 12 个 AirSim 场景 |
| 单场景大小 | 70m x 70m |
| 平均路径长度 | 107.43m |
| 平均指令长度 | 12 |
| 词汇量 | 147 |
| 目标实例 | 60 |
| 训练场景 | 10 |
| 测试场景 | 4，包括 seen 和 unseen |
| 仿真评估 episodes | 7,200 |

数据收集方式：

- 仿真环境中布置树、墙、岩石、建筑、盒子、柱子等障碍。
- 目标物体放在场景边界，并设置干扰物。
- 使用 SAC 训练每个场景的采集 agent，达到约 95% eval success 后批量生成轨迹。
- 再结合 expert demonstration，提高数据质量。
- 额外收集真实飞行数据，用于减轻 sim-to-real gap。

## 训练策略

AutoFly 使用两阶段训练：

1. Vision-language alignment
   - 用 prism-siglip-7b 配置初始化。
   - 建立视觉和语言表示的基础对齐。

2. Spatially-informed robot action fine-tuning
   - 将伪深度信息和 VLA backbone 联合微调。
   - 比较 Siamese MLP projector、Non-Siamese projector 和 direct depth input。
   - 实验显示 Siamese MLP projector 效果最好。

训练细节：

- VLM backbone 学习率：`2e-5`。
- Pseudo-depth projector 学习率：`1e-4`。
- 微调步数：80K gradient steps。

## 实验指标

AutoFly 使用三个主要指标：

| 指标 | 含义 |
|---|---|
| SR | Success Rate，成功率 |
| CR | Collision Rate，碰撞率，越低越好 |
| PER | Path Efficiency Rate，路径效率 |

成功条件比 OpenFly 更偏机器人控制：无人机不仅要接近目标，还要满足朝向约束。论文附录中给出阈值：距离目标 5m 内，且朝向角偏差不超过 15 度。

## 实验与主要结果

仿真整体结果：

| 方法 | SR | CR | PER |
|---|---:|---:|---:|
| RT-1 | 24.3 | 65.1 | 61.1 |
| RT-2 | 41.9 | 26.0 | 73.7 |
| OpenVLA | 44.0 | 24.5 | 75.1 |
| AutoFly | 47.9 | 21.9 | 77.3 |

论文核心结论是：AutoFly 相比 OpenVLA 提升 3.9% SR，降低 2.6% CR，并提升 2.2% PER。

真实环境结果：

| 场景 | Sim:Real | SR | CR | PER |
|---|---|---:|---:|---:|
| indoor | 0K:1K | 10 | 40 | 61.1 |
| indoor | 5K:1K | 25 | 65 | 71.3 |
| indoor | 10K:1K | 60 | 30 | 76.5 |
| outdoor | 10K:1K | 55 | 35 | 75.1 |

论文声称室内 60% 和室外 55% 的真实飞行成功率，说明方法能一定程度从仿真迁移到真实无人机。

## Ablation 结论

| 实验 | 结论 |
|---|---|
| 去掉 pseudo-depth encoder | SR 从 47.9 降到 44.0，说明伪深度有效 |
| depth projector 对比 | 专用 depth projector 明显优于直接用 SigLIP/DINOv2 处理深度图 |
| dataset rebalancing | 不做 rebalancing 时 SR 只有 16.6，说明长轨迹行为分布不平衡很严重 |
| depth-vision-language alignment | Siamese MLP 最好，direct depth input 最差 |
| vision encoder | DINO-SigLIP fusion 优于单独 CLIP/DINO/SigLIP |
| challenging scenarios | 动态障碍场景中 pseudo-depth 带来的 SR 提升最大 |

## 论文贡献

1. 将 UAV VLN 从详细路线指令推进到粗粒度目标下的自主导航。
2. 引入 pseudo-depth encoder，用 RGB 推出深度感知表示，降低对真实深度相机的依赖。
3. 构建包含仿真和真实飞行的数据集，强调避障、目标识别和规划的连续工作流。
4. 使用 rebalancing 缓解长轨迹中“避障阶段过多、目标接近阶段过少”的数据偏置。
5. 在仿真和真实环境中都给出 VLA baselines 对比。

## 优点

- 问题设定更接近真实无人机探索任务，不要求人类提供完整路线。
- 伪深度设计有现实意义：只用 RGB，减少机载硬件负担。
- 数据集包含 1K 真实飞行 episodes，比纯仿真 VLN 更重视 sim-to-real。
- 论文的 ablation 比较完整，能支撑 pseudo-depth、rebalancing、alignment 设计。
- 部署部分讨论了远程服务器、LAN 通信、ONNX/TensorRT 加速和流水线推理。

## 局限和可批判点

1. 指令语言并不复杂
   - 平均指令长度只有 12，词汇量 147。
   - 这更像目标引导和动作控制，不是长文本地标导航理解。

2. 成功率仍然有限
   - 仿真 overall SR 为 47.9%。
   - 真实 outdoor SR 为 55%，碰撞率 35%，距离安全可靠部署还有明显差距。

3. 工程复现难度可能较高
   - 论文使用自建无人机、RealSense D455、远程服务器和通信系统。
   - 部署加速依赖 ONNX/TensorRT/RTX 4090，普通本地设备很难完整复现。

4. 数据集和代码可用性需要进一步验证
   - 论文写明 model/data/code 公开，但本次任务中尚未实际复现 AutoFly 工程链路。
   - 目前只完成论文阅读分析，不能声称 AutoFly 复现。

5. 和 OpenFly 的 benchmark 目标不同
   - AutoFly 更重控制和避障。
   - OpenFly 更重 aerial VLN 平台、数据生成和地标指令跟随。
   - 两者不能只按 SR 数值直接比较。

## 个人点评

我认为 AutoFly 的亮点是把 UAV VLA 往真实自主飞行推进了一步，尤其是 pseudo-depth encoder 很符合无人机硬件受限的现实需求。相比只做语言路线跟随，它更关注避障、目标识别和路径规划的连续控制闭环。

但从学术汇报角度看，AutoFly 也要谨慎评价：它的语言复杂度比 OpenFly 低很多，真实飞行成功率和碰撞率还不能说明已经达到可靠部署水平。此外，论文的完整复现需要自建硬件和远程推理系统，本地短期内很难复刻。因此本次把它作为对照论文阅读，而不是作为主要复现对象，是比较合理的选择。

## 汇报中可以怎么说

AutoFly 的价值在于把无人机大模型导航从“跟随详细路线指令”推进到“粗粒度目标下自主避障和规划”。它的 pseudo-depth encoder 是一个很实用的设计，因为真实无人机通常希望减少传感器负担，只用 RGB 也能获得一定几何感知。但它的问题是语言复杂度较低，真实飞行成功率仍不高，工程复现门槛也比 OpenFly 更高。

## 可能被问的问题

1. AutoFly 为什么还属于 Vision-Language-Action，而不只是避障控制？
2. 伪深度和真实深度相比有什么优势和风险？
3. 为什么 AutoFly 的词汇量比 OpenFly 小很多？
4. AutoFly 的 SR 只有 47.9%，是否说明方法还不成熟？
5. AutoFly 和 OpenFly 谁更接近真实应用？
6. AutoFly 的真实飞行实验是否足够有说服力？
7. 为什么本次复现没有选择 AutoFly？
