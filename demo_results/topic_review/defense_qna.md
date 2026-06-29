# 导师答辩 Q&A 准备

## 1. 为什么选择“无人机大模型”这个 topic？

因为它结合了多模态大模型、具身智能和机器人控制三个方向。无人机任务比普通图像理解更难，因为模型不仅要理解视觉和语言，还要输出能执行的飞行动作，并面对三维空间、长距离导航、避障和安全约束。

## 2. 为什么选 OpenFly 做复现，而不是 AutoFly？

OpenFly 的开源链路更适合本地复现。它提供 HuggingFace 模型、项目仓库、AirSim 配置和 eval 入口，可以在 RTX 4060 Laptop 8GB 上通过 4bit inference 跑通核心链路。AutoFly 更依赖自建无人机、真实飞行数据、远程 4090 推理和部署系统，当前更适合作为阅读和对比论文。

## 3. 当前 OpenFly 算复现成功了吗？

算阶段性复现成功，但不是完整 benchmark 复现。已经完成了本地 4bit 模型加载、三帧视觉输入、语言指令输入、动作输出、AirSim 单场景闭环执行和指标记录。没有完成完整 100K 数据集、18 场景、官方 full benchmark 和训练复现。

## 4. 为什么不直接按 README 全部安装和训练？

完整 README 包括 ROS2、AirSim/UE、完整仿真资产、训练依赖和大规模数据，超出当前 8GB GPU 和本地存储条件。为了可控复现，我们先选择最小闭环：4bit inference + 单场景 AirSim smoke eval。这样可以快速验证核心链路，并且保留审计记录。

## 5. 为什么使用 4bit，会不会影响结果？

会有影响风险。4bit 是为了适配 RTX 4060 Laptop 8GB 显存，否则 7B 模型 BF16 全量加载大概率 OOM。本次结果应该表述为 4bit 条件下的本地 smoke eval，不等同于论文 BF16/flash-attn 设置。后续可以做 8bit 或 BF16 CPU-offload 小样本对照来评估量化影响。

## 6. 为什么本地 AirSim 结果不能和论文 test-seen/test-unseen 直接比较？

因为实验范围不同。论文结果来自完整 test split，多场景、多样本、官方设置。本地结果只在 `env_airsim_16` 单场景 30 条样本上跑，且是 4bit、max steps 20。因此只能说明本地链路可运行和初步趋势，不能代表官方 benchmark。

## 7. 本地 OpenFly smoke eval 的结果是什么？

`env_airsim_16` 30 条样本结果为：

| NE | SR | OSR | SPL |
|---:|---:|---:|---:|
| 76.457 | 16.67% | 56.67% | 13.46% |

解释：OSR 较高说明模型不少时候能进入目标半径，但 SR/SPL 较低说明最终 stop 时机和路径效率仍不稳定。

## 8. action 9 偏多是什么意思？

action 9 是长距离 forward。模型在本地 smoke eval 中 86.80% 的预测是 action 9，说明它明显倾向向前飞。OpenFly 数据本身 forward-heavy，但 ground truth 中 action 9 占比没有这么高。因此 action 9 偏多说明本地设置下存在过强 forward bias，可能和数据先验、4bit、max steps、prompt、历史 keyframe 不完整有关。

## 9. OpenFly-Agent 为什么需要 keyframe？

无人机长距离飞行中，当前帧可能看不到之前经过的关键地标。如果直接输入所有历史帧，token 数太大且冗余严重。OpenFly-Agent 用 keyframe selection 选择关键历史观测，再用 visual token merging 压缩历史 token，让模型既能记住重要地标，又不被大量图像 token 淹没。

## 10. OpenFly 的核心贡献是什么？

不是单个模型，而是完整平台：

- 集成 UE、GTA V、Google Earth、3D GS。
- 构建 100K trajectories、18 scenes。
- 提供 aerial VLN benchmark。
- 提出 OpenFly-Agent，用 keyframe selection 和 visual token merging 提升长距离空中导航。

## 11. AutoFly 的核心贡献是什么？

AutoFly 把任务从详细路线指令跟随推进到粗粒度目标下的自主导航。它提出 pseudo-depth encoder，从 RGB 中生成伪深度表示，增强无人机避障和空间理解；同时构建包含仿真和真实飞行的数据集，并用 two-stage training 对齐视觉、深度、语言和动作。

## 12. OpenFly 和 AutoFly 最大区别是什么？

OpenFly 更偏 aerial VLN 平台和 benchmark，语言指令更长，数据规模更大，重点是地标导航和历史关键帧。AutoFly 更偏真实无人机自主控制，语言更短，重点是避障、伪深度和 sim-to-real。两者分别代表 benchmark 平台化和真实自主飞行两个方向。

## 13. AutoFly 为什么不用真实深度相机，而用 pseudo-depth？

论文认为真实深度传感器会增加载荷、成本和硬件复杂度，而且 AirSim 中理想深度和真实深度存在差距。用 Depth Anything V2 从 RGB 生成伪深度，可以只依赖 RGB 摄像头，同时增强空间理解。但风险是伪深度有估计误差，极端场景下可能影响安全。

## 14. AutoFly 的局限是什么？

主要局限包括：

- 语言词汇量较小，平均指令较短，语言理解复杂度不如 OpenFly。
- 仿真 overall SR 47.9%，真实 outdoor SR 55%，碰撞率仍较高。
- 部署依赖远程服务器、通信和加速链路，普通设备不容易完整复现。
- 论文强调真实飞行，但规模和复杂度仍有限。

## 15. 本次复现最关键的工程问题是什么？

最关键的是 custom HF class import 路径和 4bit 适配。OpenFly 的 HF 自定义类实际在 `train/extern/hf/`，不是项目根目录的 `extern/hf/`。修复路径后，再用 bitsandbytes 4bit 加载模型，才在 8GB 显存上跑通。

## 16. 为什么 `CLAUDE.md` 不适合写论文 review？

`CLAUDE.md` 是给代码代理和开发工具看的项目说明，主要记录如何操作仓库、运行命令和理解代码结构。导师汇报需要的是论文阅读、复现证据、结果分析和答辩材料，应该放在 `demo_results/topic_review/` 这种面向人的报告目录。

## 17. 如果老师问“你到底做了什么工作”，怎么回答？

我做了四部分：

1. 阅读 OpenFly，并实际复现 OpenFly-Agent 4bit inference。
2. 进一步跑通 AirSim `env_airsim_16` 30 条闭环 smoke eval，记录 NE/SR/OSR/SPL。
3. 分析失败案例，尤其是 action 9 forward bias。
4. 阅读 AutoFly，整理它和 OpenFly 在任务设定、模型和评估上的差异。

## 18. 如果老师问“为什么结果比论文差”，怎么回答？

不能直接说“比论文差”，因为实验设置不同。本地只跑单场景 30 条、4bit、20 steps，而论文是完整 benchmark。SR/SPL 较低可能来自 stop 时机、max steps、量化、prompt、历史 keyframe 不完整等因素。OSR 56.67% 说明模型有接近目标区域的能力，但最终停止策略还不稳定。

## 19. 后续最值得做的改进是什么？

优先做评估对齐，而不是训练：

- 把 max steps 从 20 改到 100，先跑 5 条再扩展。
- 做 prompt wrapping ablation。
- 做最近三帧和近似 keyframe 输入对比。
- 下载第二个 AirSim 场景，观察跨场景稳定性。
- 小样本对比 4bit 和更高精度设置。

## 20. 这个 topic 的总体评价是什么？

无人机大模型导航还没有达到可靠实用阶段，但已经出现了两条重要路线：一条是 OpenFly 这样的多源仿真平台和大规模 aerial VLN benchmark，另一条是 AutoFly 这样的面向真实自主控制和 sim-to-real 的 VLA 模型。当前关键瓶颈仍然是数据、仿真真实性、安全评估、动作稳定性和真实部署成本。

## 21. 这些工作和之前做的 Fast-Drone-250、VINS-Fusion、FAST-LIO2、EGO-Planner 有什么关系？

之前做的项目主要解决无人机底层自主飞行能力，包括定位、建图、避障、轨迹规划和实机执行。OpenFly 和 AutoFly 解决的是更上层的语义导航问题，即无人机如何理解视觉和语言目标，并输出高层导航动作。两者可以组成分层系统：大模型负责语义理解和任务决策，VINS-Fusion/FAST-LIO2 负责位姿和地图，EGO-Planner/Diff-Planner 负责安全轨迹，Fast-Drone-250 负责真实飞行验证。

## 22. 未来如何把 OpenFly/AutoFly 和已有无人机项目有机结合？

一个可行方向是 language-conditioned safe UAV navigation。具体来说，用 OpenFly/AutoFly 类模型把语言指令和图像转成高层动作或 waypoint，再用 FAST-LIO2 地图和 EGO-Planner 检查安全性并生成可执行轨迹。还可以用 VINS-Fusion/FAST-LIO2 的 keyframe 改进 OpenFly 的历史帧输入，用 Fast-Drone-250 采集小规模真实 image-language-action 数据，进一步研究真实无人机上的 VLA 导航。
