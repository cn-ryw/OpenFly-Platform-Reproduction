# OpenFly-Agent 本地复现与无人机大模型论文 Review

本仓库是我针对“无人机大模型”topic 做的本地复现、论文阅读和导师汇报材料归档。

上游官方项目：

- OpenFly-Platform 官方仓库：https://github.com/SHAILAB-IPEC/OpenFly-Platform
- OpenFly-Agent HuggingFace 模型：https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b
- 本仓库保留的官方原始 README 备份：[README_OpenFly_Platform_Official.md](README_OpenFly_Platform_Official.md)

注意：本仓库不是官方 OpenFly 项目，而是基于官方仓库做的本地复现记录和论文 review 归档。

## 本仓库包含什么

本次工作聚焦于在本机资源受限条件下完成可审计的 OpenFly-Agent 复现：

- 硬件：RTX 4060 Laptop GPU，8GB 显存。
- 系统：Ubuntu 22.04。
- 环境：conda `openfly`，Python 3.10。
- 模型：`IPEC-COMMUNITY/openfly-agent-7b`。
- 推理方式：bitsandbytes NF4 4bit，`device_map="auto"`。
- 复现范围：OpenFly-Agent 4bit inference + AirSim `env_airsim_16` 单场景 smoke evaluation。
- 论文 review 范围：OpenFly 与 AutoFly 两篇论文的阅读笔记、对比、汇报大纲和答辩 Q&A。

本仓库不声称完成完整 OpenFly benchmark、完整训练、完整 100K trajectory 数据集复现或 18 个场景的全量评估。

## 主要复现结果

### 单次 4bit 推理

输入：

- `demo_imgs/hist_1.png`
- `demo_imgs/hist_2.png`
- `demo_imgs/current.png`
- Prompt: `Fly forward carefully, avoid obstacles, and move towards the open area.`

输出：

- `demo_results/openfly_own_video_result.json`

### AirSim 30 条 Smoke Eval

当前最完整的本地 AirSim 评估结果：

- 结果目录：`demo_results/airsim_smoke/20260629_005458`
- 场景：`env_airsim_16`
- 样本数：30
- 最大步数：20
- 模型设置：OpenFly-Agent 4bit local inference

| Scope | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| local 4bit `env_airsim_16` 30 samples | 76.457 | 16.67% | 56.67% | 13.46% |

解释：OSR 较高说明模型多次进入目标半径，但 SR/SPL 较低说明最终停止时机和路径效率仍不稳定。这是本地 smoke eval，不是官方完整 benchmark。

## 重要文件导航

### 复现脚本

| 文件 | 用途 |
|---|---|
| `infer_openfly_4bit.py` | 三帧图片 + prompt 的单次 4bit 推理 |
| `run_openfly_experiments.py` | prompt/frame sensitivity 实验 |
| `run_openfly_extended_experiments.py` | 官方风格 prompt 和自有视频多帧实验 |
| `eval_airsim_4bit_smoke.py` | AirSim `env_airsim_16` smoke eval |
| `analyze_airsim_smoke_results.py` | AirSim 结果统计分析 |
| `prepare_airsim_review_assets.py` | 生成案例图页和 review assets |
| `generate_reproduction_audit.py` | 生成复现审计包和 manifest |

### 复现报告

| 文件 | 用途 |
|---|---|
| `demo_results/advisor_review_report.md` | 给导师看的阶段性复现报告 |
| `demo_results/reproduction_audit.md` | 复现审计报告 |
| `demo_results/manifest.json` | 环境、模型、hash、结果路径等机器可读记录 |
| `demo_results/official_eval_alignment_audit.md` | 和官方 `train/eval.py` 的动作、prompt、历史帧逻辑对齐 |
| `demo_results/action9_bias_analysis.md` | action 9 forward bias 分析 |
| `demo_results/simulation_eval_plan.md` | 后续仿真评估路线 |

### 论文 Review 材料

| 文件 | 用途 |
|---|---|
| `demo_results/topic_review/README.md` | 汇报材料索引 |
| `demo_results/topic_review/openfly_reading_notes.md` | OpenFly 阅读笔记与复现总结 |
| `demo_results/topic_review/autofly_reading_notes.md` | AutoFly 阅读笔记 |
| `demo_results/topic_review/openfly_vs_autofly_comparison.md` | OpenFly 与 AutoFly 对比 |
| `demo_results/topic_review/presentation_outline.md` | 16 页左右的学术分享大纲 |
| `demo_results/topic_review/defense_qna.md` | 22 个答辩问题与回答 |
| `demo_results/topic_review/prior_work_integration.md` | 与 Fast-Drone-250、VINS-Fusion、FAST-LIO2、EGO-Planner、Diff-Planner 等既有工作的联系 |

### 重要图片结果

| 文件 | 用途 |
|---|---|
| `demo_results/input_frames_contact_sheet.png` | 三张输入图片汇总 |
| `demo_results/extended_experiments/20260628_235508/video_frame_contact_sheet.png` | 自有视频抽帧实验图页 |
| `demo_results/airsim_smoke/20260629_005458/contact_sheet.png` | AirSim smoke eval 图页 |
| `demo_results/review_assets/airsim_env16_30_cases/case_gallery.png` | 成功/失败案例图页 |

## 如何运行关键脚本

先进入 conda 环境：

```bash
conda activate openfly
cd ~/research/OpenFly-Platform
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

单次 4bit 推理：

```bash
python -u infer_openfly_4bit.py
```

轻量实验矩阵：

```bash
python -u run_openfly_experiments.py
```

扩展 prompt / 视频帧实验：

```bash
python -u run_openfly_extended_experiments.py
```

AirSim smoke eval 需要先准备好 AirSim 场景和设置：

```bash
python -u eval_airsim_4bit_smoke.py --env-name env_airsim_16 --max-samples 30 --max-steps 20
```

## License and Attribution

OpenFly-Platform 上游代码和资产属于原作者。完整官方说明、安装方式和许可信息请参考：

https://github.com/SHAILAB-IPEC/OpenFly-Platform
