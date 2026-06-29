# OpenFly-Agent 本地复现阶段性汇报

## 一句话结论

我们已经完成 OpenFly-Agent 在 RTX 4060 Laptop 8GB 条件下的 4bit 推理复现，并进一步跑通 AirSim `env_airsim_16` 单场景 30 条 smoke evaluation。当前结果足以作为阶段性复现进展汇报，但不能表述为完整复现论文 benchmark。

## 复现目标

原始目标不是训练或完整数据集复现，而是在本机资源限制下完成：

- 三帧图像 + 英文导航指令输入。
- OpenFly-Agent 7B 4bit 推理。
- 输出 UAV action。
- 记录实验过程、依赖、显存、结果、warning。
- 在可行范围内尝试 AirSim 闭环评估。

## 当前已完成

| 模块 | 状态 | 证据 |
|---|---|---|
| 单次 4bit inference | 完成 | `demo_results/openfly_own_video_result.json` |
| prompt / frame sensitivity | 完成 | `demo_results/extended_experiments/20260628_235508/summary.md` |
| 复现审计包 | 完成 | `demo_results/reproduction_audit.md`, `demo_results/manifest.json` |
| AirSim 场景准备 | 完成单场景 | `envs/airsim/env_airsim_16` |
| AirSim 30 条 smoke eval | 完成 | `demo_results/airsim_smoke/20260629_005458/analysis.md` |
| 代表案例图页 | 完成 | `demo_results/review_assets/airsim_env16_30_cases/case_gallery.png` |
| 官方 eval 对齐审计 | 完成 | `demo_results/official_eval_alignment_audit.md` |
| action 9 偏多分析 | 完成 | `demo_results/action9_bias_analysis.md` |

## 设备与环境

- Ubuntu 22.04
- RTX 4060 Laptop GPU，8GB 显存
- NVIDIA Driver 580.159.03
- conda env: `openfly`
- Python 3.10.20
- PyTorch 2.12.1+cu126
- Transformers 4.47.1
- `bitsandbytes` 4bit 可用
- `timm==0.9.16`

## 论文对照

论文 OpenFly 的完整目标是一个 comprehensive aerial VLN platform：

- 4 rendering engines：Unreal Engine、GTA V、Google Earth、3D GS。
- 100K trajectories。
- 18 scenes。
- OpenFly-Agent 是 keyframe-aware VLN model。
- 官方 benchmark 指标包含 NE、SR、OSR、SPL。

论文 OpenFly-Agent 完整 test set 指标：

| Split | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| test-seen | 93m | 34.3% | 64.3% | 24.9% |
| test-unseen | 154m | 22.6% | 56.2% | 19.1% |

我们的当前结果不是完整 test set，而是单场景 `env_airsim_16` 30 条 smoke eval：

| Scope | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| local 4bit `env_airsim_16` 30 samples | 76.457 | 16.67% | 56.67% | 13.46% |

不能直接横向比较，但可以说明：本地链路已经跑通，OSR 接近论文 unseen 量级，SR/SPL 偏低主要集中在 stop 时机和 action 9 forward 偏置。

## 成功链路

当前已经验证以下链路：

1. 本地 custom HF class import 路径修复：`train/extern/hf`。
2. 4bit 加载 `IPEC-COMMUNITY/openfly-agent-7b`。
3. 三帧图像输入。
4. `predict_action(..., unnorm_key="vlnv1", do_sample=False)`。
5. raw action -> round -> discrete action id。
6. AirSim `front_custom` 相机图像获取。
7. 根据 action 更新 UAV pose。
8. 记录 NE/SR/OSR/SPL、显存、日志、截图。

30 条 AirSim eval 没有 OOM、Traceback 或图像错误，峰值 PyTorch allocated 显存约 4526.9 MB。

## 失败模式

代表图页：

- `demo_results/review_assets/airsim_env16_30_cases/README.md`
- `demo_results/review_assets/airsim_env16_30_cases/case_gallery.png`

主要失败模式：

| 类型 | 表现 | 解释 |
|---|---|---|
| near-miss overshoot | 进入 20m 半径后继续飞远 | 模型看到目标附近但 stop 时机不稳 |
| max-steps forward bias | 多步 action 9，直到步数耗尽 | forward 先验强，转向/停止不足 |
| immediate stop fail | step 0 输出 stop | 可能与 prompt、图像、action fallback 或量化有关 |
| mixed turn near-miss | 有 turn/up/down，但最终没停在目标内 | 局部动作有反应，但全局导航不稳 |

## Action 9 偏多解释

预测分布中 action 9 占 86.80%。这不是简单 bug，但偏置明显过强。

证据：

- 论文指出 Forward action 通常占飞行轨迹较大比例。
- `configs/eval_test.json` 全部 eval 样本中 action 9 占 50.66%。
- `env_airsim_16` 非负 ground-truth action 中 action 9 占 65.72%。
- 本地预测 action 9 占 86.80%，明显高于数据先验。

可能原因：

- 数据分布本身 forward-heavy。
- 本地 smoke eval 用 20 steps，而官方 eval 是 100 steps。
- 本地未完整实现论文的 keyframe/grounding selection。
- prompt 形式存在训练/官方 eval 差异。
- 4bit 量化可能放大 greedy decoding 偏置。

## 当前还缺什么

还缺：

1. 多场景完整 benchmark。
2. 官方 `train/eval.py` BF16/flash-attn 原样评估。
3. 训练复现。
4. 完整 OpenFly dataset 下载和处理。
5. collision checking 与点云地图评估。
6. 论文 keyframe selection / landmark grounding 的完整评估对齐。
7. 4bit vs BF16/8bit 量化影响对照。

## 下一步计划

短期，建议 1-2 天内完成：

1. `env_airsim_16` 用 `--max-steps 100` 小规模复跑，先取 5 条，再决定是否 30 条。
2. 做 prompt wrapping ablation：原始 `gpt_instruction` vs `What action should the robot take to ...?`。
3. 做 history ablation：最近三帧 vs 近似 keyframe/拐点帧。
4. 统计 raw action 到 action 9 与其他模板的 margin。

中期：

1. 下载第二个 AirSim 场景，做跨场景 smoke eval。
2. 如果有 HF dataset 权限，按官方 seen/unseen split 扩展。
3. 尝试 8bit 或 BF16/CPU-offload 小样本对照。

长期：

1. 完整复现官方 benchmark 多场景指标。
2. 再考虑训练或微调复现。
3. 暂不建议在当前 8GB 显存机器上做完整训练。

## 汇报建议措辞

推荐说：

> 我们已经在本地 RTX 4060 Laptop 8GB 条件下完成 OpenFly-Agent 4bit 推理复现，并跑通 AirSim `env_airsim_16` 单场景 30 条闭环 smoke evaluation。当前复现覆盖模型加载、三帧视觉输入、指令输入、动作输出、AirSim 图像获取、动作执行和 NE/SR/OSR/SPL 记录。该结果证明本地受限设备下的核心链路可运行，但仍不是完整论文 benchmark 复现。下一步将重点对齐官方 eval 的 max steps、prompt、历史关键帧与量化设置，并扩展到更多 AirSim 场景。
