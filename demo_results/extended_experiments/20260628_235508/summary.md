# OpenFly-Agent 4bit Extended Experiments

## 实验目的

本轮实验补充两类更接近汇报需求的轻量验证：使用 OpenFly annotation 中真实 `gpt_instruction` 风格的 prompt，以及从自有 `demo_video.mp4` 抽取多组连续三帧。实验仍然是本机 4bit inference，不是完整 benchmark 复现。

## 复现元数据

- Repo commit: `c075075497a7122bad82f5b76b9be926ad5a81b3`
- Git dirty: `True`
- HF model revision: `21dcce235f1f2d40fc23c76998051abc5434cc99`
- Command: `/home/ruan/miniconda3/envs/openfly/bin/python run_openfly_extended_experiments.py`
- pip freeze: `demo_results/extended_experiments/20260628_235508/pip_freeze.txt`
- Contact sheet: `demo_results/extended_experiments/20260628_235508/video_frame_contact_sheet.png`

## 环境摘要

- Model: `IPEC-COMMUNITY/openfly-agent-7b`
- Unnorm key: `vln_norm`
- Python: `3.10.20`
- PyTorch: `2.12.1+cu126`
- Transformers: `4.47.1`
- Tokenizers: `0.21.4`
- TIMM: `0.9.16`
- bitsandbytes: `0.49.2`
- CUDA device: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Peak PyTorch allocated GPU memory: `4466.9 MB`

## 实验数量

- Baseline: `2`
- 官方风格 prompt: `10`
- 自有视频帧组: `5`

## Warning 审计

- non_fatal_warnings_only: `True`
- FutureWarning: `3`
- UserWarning: `392`
- dependency version warning: `1`
- no-op warning: `392`
- Traceback/Exception/OOM: `0/0/0`

## 结果表

| name | group | action | rounded | nearest template | L2 vs baseline | max abs |
|---|---|---|---|---|---:|---:|
| baseline_1 | baseline | `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 0, 0, 0, 0]` | 0_stop | 0.000000 | 0.000000 |
| baseline_2 | baseline | `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 0, 0, 0, 0]` | 0_stop | 0.000000 | 0.000000 |
| official_seen_0 | official_prompt | `[0.0020, 4.9902, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 1, 0, 0, 0]` | 8_move_forward_6 | 5.175775 | 4.980392 |
| official_seen_1 | official_prompt | `[0.0020, 4.9902, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 1, 0, 0, 0]` | 8_move_forward_6 | 5.175775 | 4.980392 |
| official_seen_2 | official_prompt | `[0.0020, 4.9902, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 1, 0, 0, 0]` | 8_move_forward_6 | 5.175775 | 4.980392 |
| official_seen_3 | official_prompt | `[0.0020, 4.9902, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 1, 0, 0, 0]` | 8_move_forward_6 | 5.175775 | 4.980392 |
| official_seen_4 | official_prompt | `[0.0020, 0.0098, 14.9706, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[0, 0, 15, 0, 1, 0, 0, 0]` | 2_turn_left | 15.007435 | 14.941176 |
| official_unseen_0 | official_prompt | `[0.9980, 0.0098, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 1, 0, 0, 0]` | 0_stop | 0.996078 | 0.996078 |
| official_unseen_1 | official_prompt | `[0.9980, 0.0098, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 1, 0, 0, 0]` | 0_stop | 0.996078 | 0.996078 |
| official_unseen_2 | official_prompt | `[0.0020, 4.9902, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 1, 0, 0, 0]` | 8_move_forward_6 | 5.175775 | 4.980392 |
| official_unseen_3 | official_prompt | `[0.0020, 4.9902, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 1, 0, 0, 0]` | 8_move_forward_6 | 5.175775 | 4.980392 |
| official_unseen_4 | official_prompt | `[0.0020, 4.9902, 0.0294, 0.0294, 1.0000, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 1, 0, 0, 0]` | 8_move_forward_6 | 5.175775 | 4.980392 |
| video_frames_00_0.8s | video_frames | `[0.0020, 4.9902, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 0, 0, 0, 0]` | 8_move_forward_6 | 5.079023 | 4.980392 |
| video_frames_01_2.0s | video_frames | `[0.0020, 4.9902, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 0, 0, 0, 0]` | 8_move_forward_6 | 5.079023 | 4.980392 |
| video_frames_02_3.2s | video_frames | `[0.0020, 4.9902, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 0, 0, 0, 0]` | 8_move_forward_6 | 5.079023 | 4.980392 |
| video_frames_03_4.4s | video_frames | `[0.0020, 4.9902, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 0, 0, 0, 0]` | 8_move_forward_6 | 5.079023 | 4.980392 |
| video_frames_04_5.6s | video_frames | `[0.0020, 4.9902, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 0, 0, 0, 0]` | 8_move_forward_6 | 5.079023 | 4.980392 |

## 解释

- Baseline 复跑差异 L2 = `0.000000`。
- 官方风格 prompt 最大 L2 差异 = `15.007435`。
- 自有视频帧组最大 L2 差异 = `5.079023`。
- Baseline action = `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]`。
- 官方 prompt 与自有视频图像不匹配，因此这些结果只能说明语言分布变化下 action 输出是否敏感，不能作为官方轨迹评测指标。
- 视频帧实验固定 prompt，只观察视觉输入变化对 action 的影响。
