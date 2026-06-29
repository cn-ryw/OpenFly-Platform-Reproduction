# OpenFly 论文阅读与复现笔记

论文：OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation

来源文件：`/home/ruan/research/Gao 等 - 2026 - Openfly A comprehensive platform ..pdf`

文本抽取：`demo_results/paper/openfly_paper.txt`

抽取文本 SHA256：`473fa046037e6e330d18231625d766896a2f6fb65da264e7c5988f3e84941d38`

## 论文定位

OpenFly 的核心不是只提出一个模型，而是提出一个面向 aerial VLN 的综合平台。它同时包含：

- 多渲染引擎数据生成。
- 大规模空中导航数据集。
- 自动化轨迹和指令生成工具链。
- OpenFly-Agent keyframe-aware VLN 模型。
- NE/SR/OSR/SPL benchmark。

因此，这篇论文更适合作为“无人机大模型 topic”中的平台型论文来讲：它关注如何构建数据、仿真、任务、模型和评估闭环。

## 研究问题

已有 aerial VLN 数据集和平台存在几个问题：

1. 场景和渲染来源单一，常依赖 AirSim 或 UE，数据多样性不足。
2. 真实航拍视角、三维空间、长距离地标导航和复杂城市环境覆盖不足。
3. 大模型/VLA 在无人机 VLN 中需要历史观测，但直接输入多帧图像会带来 token 冗余。
4. 现有模型在 aerial VLN 上成功率仍然较低，需要更适合空中视角的历史帧选择和压缩机制。

## 论文贡献

1. 提出 OpenFly 平台，把多渲染引擎、场景构建、轨迹生成、指令生成和评估流程连接起来。
2. 构建 100K trajectories、18 scenes 的大规模 aerial VLN 数据集。
3. 引入 UE、GTA V、Google Earth 和 3D GS，提升场景来源和视觉风格多样性。
4. 提出 OpenFly-Agent，用 keyframe selection 和 visual token merging 改进长距离空中 VLN。

## 平台和数据集

OpenFly 集成 4 类渲染/数据来源：

| 来源 | 作用 |
|---|---|
| Unreal Engine | 大规模高保真城市/室外场景 |
| GTA V | 类洛杉矶城市环境 |
| Google Earth | 大范围真实地理区域航拍 |
| 3D Gaussian Splatting | real-to-sim 场景重建 |

数据集关键数字：

| 项目 | 数值 |
|---|---:|
| 轨迹数 | 100K |
| 场景数 | 18 |
| 词汇量 | 15.6K |
| 平均路径长度 | 99.1m |
| 平均指令长度 | 59 |
| 动作空间 | 4 DoF aerial actions |

OpenFly 相比 AutoFly 更强调长文本地标导航和大规模多源数据生成。

## OpenFly-Agent 方法

OpenFly-Agent 基于 OpenVLA，针对空中 VLN 做了历史帧和 keyframe 设计。

关键设计：

1. 多帧历史输入
   - 原始 OpenVLA 主要处理单图像观测。
   - OpenFly-Agent 输入当前帧和历史关键帧，利用过往观察帮助长程导航。

2. Keyframe selection
   - 根据 UAV action transition 和 landmark grounding 选择关键帧。
   - 目标是保留真正有导航意义的观察，而不是均匀采样大量历史帧。

3. Visual token merging
   - 当前帧保留 256 tokens。
   - 历史 keyframes 被压缩，每个历史帧压缩到极少 token。
   - 论文实验中 history memory bank 容量 K=2。

4. 动作预测
   - 模型输出 UAV action。
   - 官方 eval 使用 `predict_action(..., unnorm_key="vlnv1", do_sample=False)`。

## 评价指标

| 指标 | 含义 |
|---|---|
| NE | Navigation Error，最终停止点和目标距离 |
| SR | Success Rate，最终停止点在目标 20m 内 |
| OSR | Oracle Success Rate，轨迹中任一点进入 20m 即算 oracle success |
| SPL | Success weighted by Path Length，考虑路径效率的成功率 |

OpenFly 的成功半径是 20m，这和 AutoFly 的 5m + 朝向约束不同，不能直接比较两篇论文的 SR。

## 实验与主要结果

OpenFly-Agent 在官方 test set 上的结果：

| Split | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| test-seen | 93m | 34.3% | 64.3% | 24.9% |
| test-unseen | 154m | 22.6% | 56.2% | 19.1% |

对比方法中，OpenFly-Agent 明显优于 Random、Seq2Seq、CMA、AerialVLN、Navid、NaVila 等方法。

Ablation 结果：

| 方法 | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| OpenVLA baseline | 231m | 2.3% | 10.8% | 2.2% |
| History | 223m | 6.9% | 23.3% | 5.6% |
| Random KS | 264m | 8.7% | 26.6% | 5.8% |
| KS | 275m | 9.2% | 28.1% | 6.1% |
| History + VTM | 215m | 16.6% | 40.5% | 9.1% |
| KS + VTM | 93m | 34.3% | 64.3% | 24.9% |

这个 ablation 是论文最有说服力的部分之一：历史信息、关键帧选择和 token merging 都对 aerial VLN 很关键。

## 我们本地复现做到的部分

本地设备：

- Ubuntu 22.04。
- RTX 4060 Laptop GPU，8GB 显存。
- conda env `openfly`。
- PyTorch 2.12.1+cu126。
- Transformers 4.47.1。
- bitsandbytes 4bit 可用。

已完成：

| 模块 | 状态 | 证据 |
|---|---|---|
| 单次 4bit inference | 完成 | `../openfly_own_video_result.json` |
| prompt/frame sensitivity | 完成 | `../extended_experiments/20260628_235508/summary.md` |
| 复现审计包 | 完成 | `../reproduction_audit.md`, `../manifest.json` |
| AirSim 单场景准备 | 完成 | `envs/airsim/env_airsim_16` |
| AirSim 30 条 smoke eval | 完成 | `../airsim_smoke/20260629_005458/analysis.md` |
| 官方 eval 对齐审计 | 完成 | `../official_eval_alignment_audit.md` |
| action 9 偏多分析 | 完成 | `../action9_bias_analysis.md` |

本地 AirSim `env_airsim_16` 30 条 smoke eval：

| Scope | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| local 4bit `env_airsim_16` 30 samples | 76.457 | 16.67% | 56.67% | 13.46% |

注意：这个结果不能和论文 test-seen/test-unseen 严格横向比较，因为我们只跑了单 AirSim 场景、30 条样本、4bit、20 max steps。

## 本地复现的关键工程修复

1. 修复 custom HF import 路径
   - 实际源码在 `train/extern/hf/`。
   - 脚本中将 `train/` 加入 `sys.path`。

2. 使用 4bit 加载适配 8GB 显存
   - `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`。
   - 避免 BF16 全量加载 OOM。

3. 处理依赖版本
   - `timm` 必须在 `>=0.9.10, <1.0.0`。
   - 本地使用 `timm==0.9.16`。

4. 对齐官方 eval 动作逻辑
   - 三帧输入。
   - `vlnv1` 反归一化。
   - raw action -> round -> discrete action template。
   - 20m 成功半径。

## 当前没有复现的部分

1. 没有完整下载 OpenFly 100K 数据集。
2. 没有完整跑官方 full benchmark。
3. 没有训练或微调 OpenFly-Agent。
4. 没有跑 18 个场景和完整 seen/unseen split。
5. 没有 BF16 + flash-attn 官方精度对照。
6. 没有完整实现论文中的 landmark grounding/keyframe selection 训练流程。
7. 没有做真实无人机部署。

## 主要失败模式

本地 AirSim smoke eval 中，主要失败模式包括：

- near-miss overshoot：曾进入 20m 半径，但继续向前飞，最终离开目标区域。
- max-steps forward bias：长时间输出 action 9，直到步数耗尽。
- immediate stop fail：部分样例第一步直接 stop。
- mixed turn near-miss：有转向动作，但最终没有稳定停在目标附近。

action 9 占预测动作的 86.80%。这不是简单 bug，因为 OpenFly 数据本身 forward-heavy，但本地预测比例明显高于 ground truth 先验，需要在汇报中披露。

## 论文优点

- 不是只做模型，而是构建了数据生成、仿真、模型、评估的完整平台。
- 数据规模大，场景来源多，覆盖 UE/GTA/Google Earth/3D GS。
- 关注 aerial VLN 的真实难点：长距离、空中视角、历史观测和地标指令。
- OpenFly-Agent 的 ablation 清晰，说明 keyframe 和 token merging 确实重要。
- 开源模型和项目使得本地受限设备也能做 4bit 推理复现。

## 论文局限

- 即使是 OpenFly-Agent，test-unseen SR 也只有 22.6%，说明任务仍很难。
- 数据和仿真资产体量大，普通本地设备难以完整复现 benchmark。
- 论文完整链路依赖多种仿真/渲染引擎，工程复杂度很高。
- 真实部署部分相比仿真 benchmark 仍有限。
- 历史 keyframe/grounding 机制对性能非常关键，但复现细节和资源门槛较高。

## 个人点评

我认为 OpenFly 的价值在于“平台化”而不只是“模型效果”。它把 UAV VLN 的数据生成、仿真来源、指令构造、模型训练和评估指标连接成一条链路，这对后续研究很重要。尤其是 3D GS 和 Google Earth 的引入，让 aerial VLN 不再局限于少数合成场景。

同时，OpenFly 的结果也说明这个方向还很难：即使最强的 OpenFly-Agent，test-unseen SR 也只有 22.6%。这意味着当前 UAV 大模型导航仍处于探索阶段，不能把论文 demo 或单场景结果误读为可靠自主飞行能力。本地复现的意义主要是验证核心链路、理解失败模式，并为后续更完整 benchmark 对齐打基础。

## 汇报中推荐表述

我复现的是 OpenFly-Agent 的本地 4bit inference 和 AirSim 单场景 smoke eval，而不是完整论文 benchmark。这个复现已经验证了模型加载、三帧视觉输入、语言指令输入、动作输出、AirSim 图像获取、动作执行和 NE/SR/OSR/SPL 记录的核心链路。在 RTX 4060 Laptop 8GB 的设备限制下，这是一个合理的阶段性复现。
