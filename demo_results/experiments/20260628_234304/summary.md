# OpenFly-Agent 4bit 后续实验记录

## 实验目的

本轮实验在不训练、不仿真、不下载完整 OpenFly 数据集的前提下，复用当前三张无人机视频截帧和已缓存模型，检查本地 4bit inference 的稳定性，以及输出是否会随 prompt 和帧序变化。

## 复现元数据

- Repo commit: `c075075497a7122bad82f5b76b9be926ad5a81b3`
- Git dirty: `True`
- HF model revision: `21dcce235f1f2d40fc23c76998051abc5434cc99`
- Command: `/home/ruan/miniconda3/envs/openfly/bin/python run_openfly_experiments.py`
- pip freeze: `demo_results/experiments/20260628_234304/pip_freeze.txt`

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
- CUDA capability: `(8, 9)`
- Peak PyTorch allocated GPU memory: `4383.9 MB`

## 输入帧

| alias | path | sha256 |
|---|---|---|
| hist_1 | `demo_imgs/hist_1.png` | `151b4d027ff8997ad56a316aaa05307a0bba60fdb38136a1740a6e42370e383d` |
| hist_2 | `demo_imgs/hist_2.png` | `320819abce4045a3404d15e6cbe9930302f4ff3a33be2a79b0b836a76ba32cea` |
| current | `demo_imgs/current.png` | `379150b2c4b94cc0b2265d0794b4e7356a557fe86dc078e6d190cadcb6ff7692` |

## Warning 审计

- non_fatal_warnings_only: `True`
- FutureWarning: `3`
- UserWarning: `392`
- dependency version warning: `1`
- no-op warning: `392`
- Traceback/Exception/OOM: `0/0/0`
- 这些 warning 会保留在 `run.log` 中；本轮未观察到 Traceback、Exception 或 OOM。

## 结果表

| name | group | action | rounded | nearest template | L1 vs baseline | L2 vs baseline | max abs |
|---|---|---|---|---|---:|---:|---:|
| baseline_1 | baseline | `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 0, 0, 0, 0]` | 0_stop | 0.000000 | 0.000000 | 0.000000 |
| baseline_2 | baseline | `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 0, 0, 0, 0]` | 0_stop | 0.000000 | 0.000000 | 0.000000 |
| forward_careful | prompt | `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 0, 0, 0, 0]` | 0_stop | 0.000000 | 0.000000 | 0.000000 |
| turn_left | prompt | `[0.0020, 4.9902, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 0, 0, 0, 0]` | 8_move_forward_6 | 5.976471 | 5.079023 | 4.980392 |
| turn_right | prompt | `[0.0020, 0.0098, 0.0294, 0.0294, 1.9961, 0.0000, 0.0000, 0.0000]` | `[0, 0, 0, 0, 2, 0, 0, 0]` | 4_go_up | 2.988235 | 2.227299 | 1.992157 |
| hover_stop | prompt | `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 0, 0, 0, 0]` | 0_stop | 0.000000 | 0.000000 | 0.000000 |
| frames_normal | frames | `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 0, 0, 0, 0]` | 0_stop | 0.000000 | 0.000000 | 0.000000 |
| frames_reversed | frames | `[0.0020, 4.9902, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[0, 5, 0, 0, 0, 0, 0, 0]` | 8_move_forward_6 | 5.976471 | 5.079023 | 4.980392 |
| frames_static_current | frames | `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]` | `[1, 0, 0, 0, 0, 0, 0, 0]` | 0_stop | 0.000000 | 0.000000 | 0.000000 |

## 解释

- Baseline 复跑差异 L2 = `0.000000`；在 `do_sample=False` 下，如果该值为 0，说明同输入推理是确定性的。
- Prompt 对照中的最大 L2 差异为 `5.079023`；它反映同一视觉输入下，语言指令对 action 向量的影响。
- 帧序对照中的最大 L2 差异为 `5.079023`；它反映历史帧顺序或历史信息变化对 action 向量的影响。
- Baseline action 为 `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]`。
- `nearest template` 来自仓库中的 10 个离散动作模板，只作为辅助解释；当前模型输出仍应视为 8 维连续 action，不应把 nearest template 当作严格分类标签。

## 注意事项

- 本实验只验证本地 4bit inference 和输入扰动敏感性，不代表完整 OpenFly benchmark 评测。
- Transformers / bitsandbytes 的 cache warning 或 future warning 不影响本轮结果，除非推理中断或 action 缺失。
- 若后续更换视频帧，应保留同样的 JSON/Markdown 记录结构，便于横向比较。
