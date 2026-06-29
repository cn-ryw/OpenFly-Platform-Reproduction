# OpenFly-Agent Local Reproduction and UAV Large-Model Paper Review

Read this in Chinese: [README.zh-CN.md](README.zh-CN.md)

Upstream official OpenFly README backup: [README_OpenFly_Platform_Official.md](README_OpenFly_Platform_Official.md)

This repository is a public archive of my local reproduction and paper-review work for the UAV large-model topic.

The upstream project is Shanghai AI Lab IPEC's OpenFly-Platform:

- Official repository: https://github.com/SHAILAB-IPEC/OpenFly-Platform
- OpenFly-Agent model: https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b

This repository is not the official OpenFly project. It keeps the upstream source tree together with my local scripts, experiment outputs, audit notes, and review materials.

## What This Archive Contains

The work here focuses on a feasible local reproduction under limited hardware:

- Hardware: RTX 4060 Laptop GPU, 8GB VRAM.
- Environment: Ubuntu 22.04, conda `openfly`, Python 3.10.
- Model: `IPEC-COMMUNITY/openfly-agent-7b`.
- Inference mode: bitsandbytes NF4 4bit, `device_map="auto"`.
- Reproduction scope: OpenFly-Agent inference plus AirSim `env_airsim_16` smoke evaluation.
- Paper-review scope: OpenFly and AutoFly reading notes, comparison, presentation outline, and defense Q&A.

This archive does not claim to reproduce the full OpenFly benchmark, full training pipeline, complete 100K trajectory dataset, or all 18 scenes.

## Main Results

### Single 4bit Inference

The single-image-sequence inference pipeline was run with:

- Input images:
  - `demo_imgs/hist_1.png`
  - `demo_imgs/hist_2.png`
  - `demo_imgs/current.png`
- Prompt:
  - `Fly forward carefully, avoid obstacles, and move towards the open area.`
- Output:
  - `demo_results/openfly_own_video_result.json`

### AirSim Smoke Evaluation

The most complete local AirSim smoke eval is:

- Run directory: `demo_results/airsim_smoke/20260629_005458`
- Scene: `env_airsim_16`
- Samples: 30
- Max steps: 20
- Mode: OpenFly-Agent 4bit local inference

| Scope | NE | SR | OSR | SPL |
|---|---:|---:|---:|---:|
| local 4bit `env_airsim_16` 30 samples | 76.457 | 16.67% | 56.67% | 13.46% |

Interpretation: the model often entered the goal radius, but final stopping behavior and path efficiency remained unstable. This is a local smoke eval result, not a full benchmark result.

## Important Files

### Reproduction Scripts

| File | Purpose |
|---|---|
| `infer_openfly_4bit.py` | Single 4bit OpenFly-Agent inference on three demo images |
| `run_openfly_experiments.py` | Prompt/frame sensitivity experiments |
| `run_openfly_extended_experiments.py` | Official-style prompt and own-video frame experiments |
| `eval_airsim_4bit_smoke.py` | AirSim `env_airsim_16` smoke evaluation |
| `analyze_airsim_smoke_results.py` | Aggregate AirSim smoke-eval analysis |
| `prepare_airsim_review_assets.py` | Generates review assets and case gallery |
| `generate_reproduction_audit.py` | Generates reproducibility manifest/audit files |

### Reproduction Reports

| File | Purpose |
|---|---|
| `demo_results/advisor_review_report.md` | Main advisor-facing reproduction report |
| `demo_results/reproduction_audit.md` | Reproducibility audit |
| `demo_results/manifest.json` | Machine-readable manifest |
| `demo_results/official_eval_alignment_audit.md` | Alignment with official `train/eval.py` logic |
| `demo_results/action9_bias_analysis.md` | Analysis of action 9 forward-bias |
| `demo_results/simulation_eval_plan.md` | Simulation evaluation roadmap |

### Paper Review Materials

| File | Purpose |
|---|---|
| `demo_results/topic_review/README.md` | Review-material index |
| `demo_results/topic_review/openfly_reading_notes.md` | OpenFly paper notes and reproduction summary |
| `demo_results/topic_review/autofly_reading_notes.md` | AutoFly paper notes |
| `demo_results/topic_review/openfly_vs_autofly_comparison.md` | Comparison between OpenFly and AutoFly |
| `demo_results/topic_review/presentation_outline.md` | 16-slide academic sharing outline |
| `demo_results/topic_review/defense_qna.md` | 22 defense questions and answers |
| `demo_results/topic_review/prior_work_integration.md` | Links to prior UAV work such as Fast-Drone-250, VINS-Fusion, FAST-LIO2, EGO-Planner, and Diff-Planner |

### Visual Results

| File | Purpose |
|---|---|
| `demo_results/input_frames_contact_sheet.png` | Three-image input contact sheet |
| `demo_results/extended_experiments/20260628_235508/video_frame_contact_sheet.png` | Own-video frame groups |
| `demo_results/airsim_smoke/20260629_005458/contact_sheet.png` | AirSim smoke-eval contact sheet |
| `demo_results/review_assets/airsim_env16_30_cases/case_gallery.png` | Representative success/failure cases |

## How To Run The Key Scripts

Activate the environment first:

```bash
conda activate openfly
cd ~/research/OpenFly-Platform
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Run single 4bit inference:

```bash
python -u infer_openfly_4bit.py
```

Run the lightweight experiment matrix:

```bash
python -u run_openfly_experiments.py
```

Run extended prompt/video-frame experiments:

```bash
python -u run_openfly_extended_experiments.py
```

Run AirSim smoke evaluation only after the AirSim scene and settings are prepared:

```bash
python -u eval_airsim_4bit_smoke.py --env-name env_airsim_16 --max-samples 30 --max-steps 20
```

## License and Attribution

The upstream OpenFly-Platform code and assets belong to their original authors. Please refer to the official upstream repository for the original license, installation instructions, and full project documentation:

https://github.com/SHAILAB-IPEC/OpenFly-Platform
