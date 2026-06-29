# OpenFly 官方 Eval 对齐审计

## 结论

当前 `eval_airsim_4bit_smoke.py` 已经对齐了官方 `train/eval.py` 的主要 AirSim 推理闭环：三帧输入、`vlnv1` 反归一化、`do_sample=False`、round 后精确匹配离散动作模板、位姿更新公式、20m 成功半径和 NE/SR/OSR/SPL 指标。

仍未完全对齐的部分主要有四类：模型加载精度、最大步数、训练时 keyframe 逻辑、完整 benchmark 范围。它们足以解释当前 smoke eval 只能作为阶段性复现结果，不能等同论文完整 benchmark。

## 论文依据

论文 `Gao 等 - 2026 - Openfly A comprehensive platform ..pdf` 表明：

- OpenFly 是一个 aerial VLN 平台，包含 4 类渲染/数据来源：Unreal Engine、GTA V、Google Earth、3D Gaussian Splatting。
- 数据集规模为 100K trajectories、18 scenes。
- OpenFly-Agent 是 keyframe-aware VLN 模型，当前帧保留 256 tokens，历史 keyframes 压缩进入 memory bank，容量 K=2。
- 评价指标为 NE、SR、OSR、SPL；成功条件是 UAV 在目标 20m 内停止。
- 论文 test-seen OpenFly-Agent 指标为 NE 93m、SR 34.3%、OSR 64.3%、SPL 24.9%；test-unseen 为 NE 154m、SR 22.6%、OSR 56.2%、SPL 19.1%。

## 源码对齐表

| 项目 | 官方 `train/eval.py` | 本地 smoke eval | 状态 |
|---|---|---|---|
| 模型 | `IPEC-COMMUNITY/openfly-agent-7b` | 同一 HF 模型 | 已对齐 |
| custom HF 类 | `train/extern/hf` 注册 `OpenFlyConfig` 等 | 同样注册 | 已对齐 |
| 输入图像 | `get_images(..., if_his=True, his_step=2)` 最近三帧，不足时复制 | `history_images()` 最近三帧，不足时复制 | 已对齐官方 eval |
| prompt | `item['gpt_instruction']` 原文 | 同样使用 `gpt_instruction` | 已对齐官方 eval |
| 训练 prompt | collator 中为 `What action should the robot take to {instruction.lower()}?` | smoke eval 未额外包装 | 与官方 eval 一致，但与训练构造存在差异 |
| 推理 dtype | BF16 + flash-attn，全量加载 | 4bit NF4 + float16 compute | 未完全对齐，适配 8GB 显存 |
| `unnorm_key` | `vlnv1` | `vlnv1` | 已对齐 |
| sampling | `do_sample=False` | `do_sample=False` | 已对齐 |
| 动作解码 | raw action -> round -> exact template match，失败默认 0 | 同样执行，并额外记录 nearest template | 已对齐 |
| 动作模板 | 0 stop；1/8/9 forward 3/6/9m；2/3 left/right；4/5 up/down；6/7 lateral | 同一模板 | 已对齐 |
| 位姿更新 | `getPoseAfterMakeAction()` step_size=3，action 9 = 9m forward | 同一公式 | 已对齐 |
| AirSim 相机 | `front_custom` | `front_custom`，`vehicle_name=drone_1` | 基本对齐；本地显式指定 vehicle |
| 成功半径 | 20m | 20m | 已对齐 |
| max steps | 官方脚本 `MAX_STEP=100` | 本地 smoke eval `max_steps=20` | 未完全对齐，为轻量 smoke 设置 |
| 场景范围 | `configs/eval_test.json` 所有环境组 | 仅 `env_airsim_16` 30 条 | 未完全对齐 |
| 碰撞检查 | 论文要求 collision failure；官方脚本当前主要按距离/图像异常记录 | 本地未接入点云 collision checking | 未完全覆盖 |

## 与训练/论文 keyframe 逻辑的差异

官方 `train/eval.py` 使用最近三帧作为模型输入，这一点本地 smoke eval 已对齐。但论文和训练数据构造强调的是 keyframe-aware 机制：

- 论文提出基于动作转折和 landmark grounding 的 keyframe selection。
- 训练数据构造里 `image_1` 是当前帧，`image_2/image_3` 不是简单最近两帧，而会根据起点、拐点、关键动作变化等规则选取。
- ablation 显示历史帧和 keyframe selection 对 SR 提升很明显。

因此，当前本地 smoke eval 对齐了仓库官方 eval 脚本，但还没有复刻论文意义上的完整 keyframe/grounding 评估逻辑。

## 当前本地 AirSim 30 条结果

结果路径：`demo_results/airsim_smoke/20260629_005458/analysis.md`

| Scope | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| 本地 `env_airsim_16` 30 条 smoke eval | 76.457 | 16.67% | 56.67% | 13.46% |
| 论文 OpenFly-Agent test-seen | 93m | 34.3% | 64.3% | 24.9% |
| 论文 OpenFly-Agent test-unseen | 154m | 22.6% | 56.2% | 19.1% |

注意：这三行不能直接横向比较。我们的结果是单 AirSim 场景、4bit、20 step smoke eval；论文结果是完整 test split benchmark。

## 当前判断

可以说：

- 已经完成本机 8GB GPU 条件下 OpenFly-Agent 4bit inference 复现。
- 已经完成 AirSim `env_airsim_16` 单场景 30 条 smoke eval。
- 已经初步对齐官方 eval 入口的动作映射、三帧输入、指标计算。

不能说：

- 已经完整复现论文 benchmark。
- 当前 SR/SPL 能代表论文结果。
- 当前失败完全来自模型能力，而不是 4bit、max steps、keyframe、prompt/历史帧差异的组合影响。

## 建议修正/下一步

1. 增加一个 `--max-steps 100` 的复跑选项，用同一 `env_airsim_16` 比较 20 step 与 100 step 差异。
2. 增加 prompt 包装 ablation：原始 `gpt_instruction` vs `What action should the robot take to ...?`。
3. 增加历史帧 ablation：最近三帧 vs 训练 builder 风格的起点/拐点/当前帧近似策略。
4. 如果显存允许，抽 3-5 条样本对比 4bit 与 BF16/CPU-offload 或 8bit，评估量化影响。
5. 在有 HF dataset 权限后，下载第二个 AirSim 场景，验证跨场景趋势。
