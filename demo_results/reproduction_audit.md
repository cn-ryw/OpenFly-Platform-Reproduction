# OpenFly-Agent 本地 4bit 复现审计记录

## 结论摘要

- 已在本机 RTX 4060 Laptop 8GB 上跑通 `IPEC-COMMUNITY/openfly-agent-7b` 的 4bit inference。
- 已完成单次推理和 `17` 组轻量对照实验，并保存 JSON、Markdown 与日志。
- 当前结果属于“本机 4bit 推理复现与轻量敏感性验证”，不是完整 OpenFly benchmark 复现。
- Baseline action: `[0.9980, 0.0098, 0.0294, 0.0294, 0.0039, 0.0000, 0.0000, 0.0000]`
- Baseline repeat L2: `0.000000`
- Max prompt L2: `15.007435`
- Max frame-order L2: `5.079023`
- Peak PyTorch allocated GPU memory: `4466.9 MB`
- Group counts: `{'baseline': 2, 'official_prompt': 10, 'video_frames': 5}`

## Extended 实验覆盖

- 官方风格 prompt 实验数量: `10`
- 自有视频多帧实验数量: `5`
- 官方风格 prompt 来自 `seen.json[0:5]` 和 `unseen.json[0:5]`；图像仍使用本地 demo 帧，因此只能验证语言分布敏感性。
- 自有视频多帧实验从 `demo_video.mp4` 抽取连续三帧组，固定 baseline prompt，用于观察视觉输入变化。

## 代码与模型

- Repo commit: `c075075497a7122bad82f5b76b9be926ad5a81b3`
- Branch: `main`
- Git dirty: `True`
- Model: `IPEC-COMMUNITY/openfly-agent-7b`
- HF revision: `21dcce235f1f2d40fc23c76998051abc5434cc99`
- Unnorm key: `vln_norm`
- do_sample: `False`

## 环境

- Python: `3.10.20`
- PyTorch: `2.12.1+cu126`
- Transformers: `4.47.1`
- Tokenizers: `0.21.4`
- TIMM: `0.9.16`
- bitsandbytes: `0.49.2`
- CUDA available: `True`
- CUDA device: `NVIDIA GeForce RTX 4060 Laptop GPU`
- CUDA capability: `(8, 9)`
- `pip freeze`: `demo_results/pip_freeze_openfly.txt`

## 输入与脚本 Hash

| item | path | sha256 |
|---|---|---|
| input:hist_1 | `demo_imgs/hist_1.png` | `151b4d027ff8997ad56a316aaa05307a0bba60fdb38136a1740a6e42370e383d` |
| input:hist_2 | `demo_imgs/hist_2.png` | `320819abce4045a3404d15e6cbe9930302f4ff3a33be2a79b0b836a76ba32cea` |
| input:current | `demo_imgs/current.png` | `379150b2c4b94cc0b2265d0794b4e7356a557fe86dc078e6d190cadcb6ff7692` |
| input:demo_video | `demo_video.mp4` | `636f2344c5a0127c576e426e91a15961b1f2c5b80557e11bf04df9ca42d86d7c` |
| input:openfly_paper_pdf | `/home/ruan/research/Gao 等 - 2026 - Openfly A comprehensive platform ..pdf` | `0c1ab59f3027ff8f74d5857036efa749217dfc502f4c3feed6e8964d67e33571` |
| script:single_inference | `infer_openfly_4bit.py` | `796459e46a3377609ea7e404fc18695f0a985ac29e09d7eaad22ffd5c319e02b` |
| script:experiment_runner | `run_openfly_experiments.py` | `f563e240e3b19b62de17175d6884e474f37c76b331b5af450aeb34b4f55beff2` |
| script:extended_experiment_runner | `run_openfly_extended_experiments.py` | `26bdc8f5165b49190795210828d885aefba1e66432674e601aa9563be7874836` |
| script:airsim_smoke_eval | `eval_airsim_4bit_smoke.py` | `6b84f8b1428c9b8126182586e698e898dc29f685253321c66262fe88b97dce57` |
| script:airsim_smoke_analyzer | `analyze_airsim_smoke_results.py` | `ba0bd3b53be48ca8d988849e3af05a91290ab29c0a55ef35ee417c8e61169280` |
| script:airsim_review_assets | `prepare_airsim_review_assets.py` | `f9b1fa8b7503bfcda6de63a191a09e5945680576575c38f84cff3ec8da34e6d9` |
| script:airsim_scene_downloader | `download_openfly_airsim_scene.py` | `5ac4495a44bb7be99ff983f7fedcf9393c317b2bf8f66b6f17b2a519de64c74d` |
| script:audit_generator | `generate_reproduction_audit.py` | `49b1bfdd894c6feca89bf0e275a7de9c85cfc166747261246b9c1d828e169145` |
| result:single_inference_result | `demo_results/openfly_own_video_result.json` | `08e173cee1bf17f25ae1825cc999a52066acb215eb7ca56af361fb024a429161` |
| result:experiment_results | `demo_results/extended_experiments/20260628_235508/results.json` | `277724f47fa255392693f5b70ee6c7bd848d7d30d4a8b9f772547426243da74f` |
| result:experiment_summary | `demo_results/extended_experiments/20260628_235508/summary.md` | `2d5767465e4073964a0b3a0562e365a22ccc058ad69398c20497deb0a0005159` |
| result:experiment_log | `demo_results/extended_experiments/20260628_235508/run.log` | `ab18a588b502acd86565f9e57483f03290f6d240dc7d5415f1726e9e9aa3b8ea` |
| result:top_level_pip_freeze | `demo_results/pip_freeze_openfly.txt` | `98da7826cc14effda2c355a8f325def36d75ee9e8026571de7e0a42b889ad030` |
| result:simulation_eval_plan | `demo_results/simulation_eval_plan.md` | `104fd51a0a135325d8d1ac6639b9af978043edf5c991c9b43e417a0771bcdd7c` |
| result:openfly_paper_text_extract | `demo_results/paper/openfly_paper.txt` | `473fa046037e6e330d18231625d766896a2f6fb65da264e7c5988f3e84941d38` |
| result:official_eval_alignment_audit | `demo_results/official_eval_alignment_audit.md` | `b1bb4c6884b3504ed388c8cb07082672873a7158fdec73c0fa0256b957949c2c` |
| result:action9_bias_analysis | `demo_results/action9_bias_analysis.md` | `d029d78c3cd718f6605aead399a643c654a6f4ff320da9d35c7bc8571b27b6b9` |
| result:advisor_review_report | `demo_results/advisor_review_report.md` | `a5f3699b347c104b78e50713577831f18297f4238b0a7aa9fc41151a62cb4ff3` |
| result:experiment_pip_freeze | `demo_results/extended_experiments/20260628_235508/pip_freeze.txt` | `9961bdac9933a204507968c007c562cad9a7a9609564c394c2c43f5046c9c404` |
| result:contact_sheet | `demo_results/extended_experiments/20260628_235508/video_frame_contact_sheet.png` | `fcbe5d7a255c53daffe8e4af14e55cb704eccbde9d6e9b11720df5c4e02a008b` |
| result:airsim_smoke_results | `demo_results/airsim_smoke/20260629_005458/results.json` | `e14b73e1c0fcd2f30637af8e5b293aac6710735ade31b8d063ad3561c23f7f36` |
| result:airsim_smoke_summary | `demo_results/airsim_smoke/20260629_005458/summary.md` | `a2562759058138f5e6f6dad8f3cc7bea3eb5c7ec5cc43820e7459805d5984eeb` |
| result:airsim_smoke_log | `demo_results/airsim_smoke/20260629_005458/run.log` | `dd56a7ee029d881cab28abb95cd535f3a38a4eb824cad5fd8ff47eabe05a7a06` |
| result:airsim_smoke_analysis | `demo_results/airsim_smoke/20260629_005458/analysis.md` | `a6295e9e2527fa694a786da7f159e95f683dd793c0606fe560ed2e6e4503b959` |
| result:airsim_smoke_analysis_json | `demo_results/airsim_smoke/20260629_005458/analysis.json` | `c6eb34afd7c759a71a78f2f92da9179964d7b1f783c7ff23701daafa86300eb8` |
| result:airsim_scene_download_manifest | `demo_results/downloads/env_airsim_16/20260629_002430.json` | `f6fdfda134f2abd867c2578b9eaa4e20213aee3dfa1511a2a64281352537777e` |
| result:airsim_smoke_contact_sheet | `demo_results/airsim_smoke/20260629_005458/contact_sheet.png` | `01d33b2506826d847fd11a5f36427b49b2a55a90824f71b50848a16e0ccc997d` |
| result:airsim_review_cases | `demo_results/review_assets/airsim_env16_30_cases/cases.json` | `ee9ce3f20fa8c76265199a813816df91b916ec2246b8e147720800678c60ddbc` |
| result:airsim_review_case_gallery | `demo_results/review_assets/airsim_env16_30_cases/case_gallery.png` | `47ae8283dea508e2cf8614f18bb50c27b7bb0c4a476f890106c79eb662e033cb` |
| result:airsim_review_readme | `demo_results/review_assets/airsim_env16_30_cases/README.md` | `b76bdbbc9f957ff8cac75e8280f55b680998b6daac61335ad4c86e15f22f572e` |
| result:airsim_review_source_analysis | `demo_results/review_assets/airsim_env16_30_cases/source_analysis.md` | `a6295e9e2527fa694a786da7f159e95f683dd793c0606fe560ed2e6e4503b959` |

## 运行命令

```bash
cd ~/research/OpenFly-Platform && conda activate openfly && export HF_HUB_ENABLE_HF_TRANSFER=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -u infer_openfly_4bit.py
cd ~/research/OpenFly-Platform && conda activate openfly && export HF_HUB_ENABLE_HF_TRANSFER=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -u run_openfly_experiments.py
cd ~/research/OpenFly-Platform && conda activate openfly && export HF_HUB_ENABLE_HF_TRANSFER=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -u run_openfly_extended_experiments.py
cd ~/research/OpenFly-Platform && conda activate openfly && python -u eval_airsim_4bit_smoke.py --dry-run --env-name env_airsim_16 --limit 3 --max-steps 20
hf auth login
cd ~/research/OpenFly-Platform && conda activate openfly && export HF_HUB_ENABLE_HF_TRANSFER=1 && python -u download_openfly_airsim_scene.py --env-name env_airsim_16
cd ~/research/OpenFly-Platform && conda activate openfly && python -u download_openfly_airsim_scene.py --env-name env_airsim_16 --zip-path /home/ruan/Downloads/env_airsim_16.zip
cd ~/research/OpenFly-Platform && conda activate openfly && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -u eval_airsim_4bit_smoke.py --env-name env_airsim_16 --limit 30 --max-steps 20 --save-frames --prepare-settings
cd ~/research/OpenFly-Platform && conda activate openfly && python -u generate_reproduction_audit.py
```

## Hugging Face CLI 状态

- Standalone `hf` CLI path: `/home/ruan/.local/bin/hf`
- Standalone `hf` CLI exists: `True`
- `hf version`: `version=1.21.0`
- `hf auth whoami`: `Error: Not logged in`
- HF CLI skill installed: `True`
- HF CLI skill path: `/home/ruan/.agents/skills/hf-cli`

## AirSim 仿真评估准备状态

- 已新增 `eval_airsim_4bit_smoke.py`，用于单场景、少样例、4bit AirSim smoke eval。
- 已新增 `download_openfly_airsim_scene.py`，用于只下载并解压单个 `env_airsim_16` 场景。
- AirSim Python client installed: `True`
- HF CLI skill installed: `True`
- `env_airsim_16` ready: `True`
- `env_airsim_16` start script exists: `True`
- Start script path: `envs/airsim/env_airsim_16/LinuxNoEditor/start.sh`
- Current blocker: `None`
- Latest AirSim smoke result: `demo_results/airsim_smoke/20260629_005458/results.json`
- Latest AirSim smoke metrics: `NE=76.45658647164754`, `SR=0.16666666666666666`, `OSR=0.5666666666666667`, `SPL=0.1346144473653307`
- Latest AirSim smoke actions: `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]`
- Latest AirSim contact sheet: `demo_results/airsim_smoke/20260629_005458/contact_sheet.png`
- 该 AirSim 结果是单场景 smoke eval，不等同于完整 OpenFly benchmark。

| sample | steps | NE | SR | OSR | SPL | actions |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 12 | 7.436 | 1 | 1 | 1.0461 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 1 | 14 | 90.853 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 9, 9, 1, 0]` |
| 2 | 20 | 37.882 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 3, 9, 9, 9, 3, 9, 3, 3, 9, 9]` |
| 3 | 1 | 136.392 | 0 | 0 | 0.0000 | `[0]` |
| 4 | 20 | 83.588 | 0 | 0 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 5 | 1 | 104.498 | 0 | 0 | 0.0000 | `[0]` |
| 6 | 11 | 19.726 | 1 | 1 | 0.7246 | `[9, 9, 9, 9, 9, 9, 9, 1, 1, 2, 0]` |
| 7 | 9 | 19.042 | 1 | 1 | 0.7400 | `[9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 8 | 8 | 101.425 | 0 | 0 | 0.0000 | `[9, 9, 9, 9, 1, 1, 1, 0]` |
| 9 | 20 | 90.254 | 0 | 0 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 10 | 20 | 64.856 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 1, 9]` |
| 11 | 12 | 57.000 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 12 | 20 | 71.608 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3]` |
| 13 | 20 | 60.472 | 0 | 0 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 14 | 1 | 105.256 | 0 | 0 | 0.0000 | `[0]` |
| 15 | 20 | 26.776 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 16 | 1 | 104.505 | 0 | 0 | 0.0000 | `[0]` |
| 17 | 1 | 164.224 | 0 | 0 | 0.0000 | `[0]` |
| 18 | 20 | 166.716 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 19 | 7 | 7.565 | 1 | 1 | 1.0277 | `[9, 9, 9, 9, 9, 9, 0]` |
| 20 | 20 | 43.823 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9]` |
| 21 | 1 | 80.007 | 0 | 0 | 0.0000 | `[0]` |
| 22 | 10 | 39.079 | 0 | 0 | 0.0000 | `[9, 9, 9, 9, 3, 9, 9, 9, 9, 0]` |
| 23 | 20 | 41.173 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 3, 1, 3, 9, 9, 9, 1, 3, 3]` |
| 24 | 20 | 115.410 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 25 | 1 | 139.102 | 0 | 0 | 0.0000 | `[0]` |
| 26 | 1 | 89.080 | 0 | 0 | 0.0000 | `[0]` |
| 27 | 5 | 18.000 | 1 | 1 | 0.5000 | `[9, 9, 9, 9, 0]` |
| 28 | 20 | 131.224 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 29 | 20 | 76.726 | 0 | 1 | 0.0000 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 9, 9, 9, 1]` |

## Warning 审计

- non_fatal_warnings_only: `True`
- FutureWarning: `3`
- UserWarning: `392`
- dependency version warning: `1`
- deprecated: `1`
- no-op: `392`
- Traceback: `0`
- Exception: `0`
- out of memory/OOM: `0/0`

说明：`no-op` warning 来自 timm/meta tensor checkpoint loading。当前实验没有 Traceback、Exception 或 OOM，因此记录为非致命 warning，但在导师 review 中应披露。

## 结果路径

- Experiment dir: `demo_results/extended_experiments/20260628_235508`
- Results JSON: `demo_results/extended_experiments/20260628_235508/results.json`
- Summary Markdown: `demo_results/extended_experiments/20260628_235508/summary.md`
- Run log: `demo_results/extended_experiments/20260628_235508/run.log`
- Manifest: `demo_results/manifest.json`
- Simulation eval plan: `demo_results/simulation_eval_plan.md`

## 未覆盖内容

- 未复现完整 OpenFly dataset、训练流程或仿真环境。
- 未运行官方 `train/eval.py` 全量 benchmark，因此还没有完整 benchmark 级别的 NE/SR/OSR/SPL 汇总指标。
- 当前 demo 使用自有视频截帧，不能替代官方 seen/unseen 图像轨迹评测。
