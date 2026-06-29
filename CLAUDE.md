# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenFly is a research platform for **aerial Vision-Language Navigation (VLN)** — navigating a drone outdoors using natural language instructions and visual observations. Developed by Shanghai Artificial Intelligence Laboratory (SHAILAB-IPEC). It includes a 100k+ trajectory benchmark dataset, the OpenFly-Agent VLA model (based on OpenVLA), and a full data-generation toolchain across four simulator types (UE, AirSim, 3DGS, GTA V).

- **Paper**: [arXiv 2502.18041](https://arxiv.org/abs/2502.18041)
- **Website**: [shailab-ipec.github.io/openfly](https://shailab-ipec.github.io/openfly/)
- **License**: MIT

## Prerequisites

- Ubuntu 22.04 (recommended), CUDA GPU
- Python 3.10 (conda env `openfly`)
- ROS2 Humble
- System deps: `xvfb`, `libgoogle-glog-dev`, `ros-humble-pcl-ros`, `nlohmann-json3-dev`

## Commands

### Build

```bash
# Build ROS2 C++ toolchain packages (from repo root)
cd tool_ws
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

# Build SIBR_viewers for 3DGS rendering (from repo root)
cd envs/gs/SIBR_viewers
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_IBR_HIERARCHYVIEWER=ON -DBUILD_IBR_ULR=OFF -DBUILD_IBR_DATASET_TOOLS=OFF -DBUILD_IBR_GAUSSIANVIEWER=OFF
cmake --build build -j --target install --config Release
```

### Install Python dependencies

```bash
conda create -n openfly python=3.10 -y
conda activate openfly
pip install -r requirements.txt
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
pip install huggingface_hub          # for scene download & HF model access
git clone https://github.com/kvablack/dlimp && cd dlimp && pip install -e .
```

### Run simulation

```bash
conda activate openfly
python scripts/sim/env_bridge.py --env env_xx_xxx   # e.g. env_airsim_16b
```

### Run toolchain (data generation)

```bash
# Point cloud generation (AirSim only)
bash scripts/toolchain/pcdgen_tool.sh env_xx_xxx

# Segmentation generation (BEV or manual mode)
bash scripts/toolchain/seggen_tool.sh env_xx_xxx bev
bash scripts/toolchain/seggen_tool.sh env_xx_xxx manual

# Trajectory generation
bash scripts/toolchain/trajgen_tool.sh env_xx_xxx

# GPT instruction generation
python3 tool_ws/src/ins_gen/gpt_generation.py --json <path.json> --type train
```

### Training

```bash
cd train
# Edit train.sh to set model_name_or_path, grid_size, history_frames, batch_size, etc.
bash train.sh
```

Training uses `torchrun` with FSDP across 8 GPUs (HYBRID_SHARD). Key hyperparameters:
- `grid_size` — token compression ratio for vision features
- `history_frames` — number of past frames (should match dataset)
- `model_name_or_path` — path to pretrained OpenVLA checkpoint
- `data_mix` — dataset mixture name defined in `datasets/dataset.py`

### Evaluation

```bash
python train/eval.py   # reads eval_test.json for environment configs
```

### Build TFDS dataset from custom data

```bash
cd train/dataset_builder/vln
tfds build --data_dir <TFDS_DATA_DIR>
```

### Inference

```bash
# 4-bit quantized inference (standalone script)
python infer_openfly_4bit.py

# Or use the HuggingFace pipeline (see README Test section)
```

### Testing

There is one TFDS dataset builder test:

```bash
cd train/dataset_builder/vln
python vln_dataset_builder_test.py
```

### Root-level reproduction scripts

These standalone scripts at the repo root are the primary entry points for inference, evaluation, and reproduction:

```bash
# 4-bit quantized single inference (needs demo_imgs/)
python infer_openfly_4bit.py

# Batch experiment runner — tests multiple prompts and action templates
python run_openfly_experiments.py
python run_openfly_extended_experiments.py

# End-to-end AirSim smoke evaluation (requires AirSim simulator running)
python eval_airsim_4bit_smoke.py --env env_airsim_16

# Analyze smoke test results
python analyze_airsim_smoke_results.py

# Download AirSim scene assets from HuggingFace
python download_openfly_airsim_scene.py --env env_airsim_16

# Generate comprehensive reproduction audit report
python generate_reproduction_audit.py
```

All root scripts follow the same pattern to access `train/` modules:
```python
ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
sys.path.insert(0, str(TRAIN_DIR))
```

### Other dependencies

The root download/experiment scripts additionally require:

```bash
pip install huggingface_hub
```

## Architecture

### Four-layer architecture

1. **Simulation Layer** (`scripts/sim/`) — bridges to four simulator types:
   - `env_bridge.py` — main entry point; reads YAML config, spawns per-thread bridges, exposes TCP server for pose commands
   - `airsim_bridge.py` — AirSim (Microsoft AirSim plugin)
   - `ue_bridge.py` — Unreal Engine (UnrealCV plugin)
   - `gs_bridge.py` — 3D Gaussian Splatting (SIBR_viewers HTTP API)
   - `gtav_bridge.py` — GTA V (DeepGTAV)
   - `common.py` — coordinate transforms (quaternion/Euler/rotation matrix)

2. **Toolchain Layer** (`tool_ws/src/`, `scripts/toolchain/`) — ROS2-based data generation:
   - `pcd_gen/` — point cloud maps from AirSim lidar scans
   - `seg_gen/` — BEV and manual semantic segmentation (C++ ROS2 + Python)
   - `traj_gen/` — path planning on voxel maps (C++ ROS2)
   - `ins_gen/` — GPT-based NL instruction generation from trajectories

3. **Model / Training Layer** (`train/`) — OpenFly-Agent VLA model:
   - **Vision**: `DinoSigLIPViTBackbone` — fused DINOv2 ViT-L + SigLIP ViT-SO400M, multi-frame input (current + N history), grid-pooled features
   - **Projector**: `FusedMLPProjector` — 3-layer GELU MLP projecting vision features to LLM embedding space
   - **LLM**: `LLaMa2LLMBackbone` — LLaMA-2-7B via HuggingFace
   - **VLM**: `PrismaticVLM` — patches visual embeddings after BOS token in the token stream
   - **Action head**: `OpenFly` class extends `PrismaticVLM` with `predict_action()` — discretizes 8-dim continuous actions into 256 bins mapped to last vocabulary tokens. Action space: stop, move_forward, turn_left, turn_right, go_up, go_down, move_left, move_right (each with magnitude variants)
   - **Training strategy** (`strategy.py`): FSDP with HYBRID_SHARD, gradient checkpointing, bf16 mixed precision. Supports fine-tuning stages: full, frozen-vision, last-layer only
   - **Data pipeline** (`datasets/`): RLDS/TFDS → `dlimp` → PyTorch `IterableDataset`. Supports interleaved dataset mixtures with weighted sampling. Dataset mixtures defined in `dataset.py` (`OXE_NAMED_MIXTURES` dict)
   - **Metrics** (`metrics.py`): JSONL + Weights & Biases logging; tracks loss, L1 loss, action token accuracy
   - **Model loading** (`load_model.py`): `OpenFly` class extends `PrismaticVLM` with `predict_action()` — performs full inference pipeline (vision encoding → LLM generation → action token decoding → denormalization); `load_vla()` loads pretrained checkpoints from local disk or HF Hub

4. **HF Integration Layer** (`train/extern/hf/`) — HuggingFace `PreTrainedModel` wrappers so the model can be loaded via `AutoModelForVision2Seq.from_pretrained()`:
   - `configuration_prismatic.py` — `OpenFlyConfig`, `PrismaticConfig`
   - `modeling_prismatic.py` — `OpenVLAForActionPrediction`, `PrismaticForConditionalGeneration`
   - `processing_prismatic.py` — `PrismaticProcessor`, `PrismaticImageProcessor`

### Key data flow

1. Scene configs (`configs/env_*.yaml`) define simulation parameters (map bounds, voxel sizes, thread counts, landmark configs)
2. Toolchain generates PCD maps → segmentation maps → trajectories → GPT instructions
3. Trajectories + instructions are converted to RLDS/TFDS format via `dataset_builder/`
4. Training reads TFDS datasets through `dlimp`, feeds (image sequence, instruction) pairs to the VLA model
5. The model outputs discretized action tokens, which are de-tokenized to continuous 8-dim actions
6. Evaluation runs the model in closed-loop simulation, measuring NE (Navigation Error), SR (Success Rate), OSR (Oracle SR), SPL (Success weighted by Path Length)

### Input format

- 3 RGB images at 224×224 (current frame + 2 history frames), processed through both DINOv2 and SigLIP encoders
- Language instruction as text prompt
- Model predicts the next action as an 8-dim vector

### Configuration & data files

- `configs/env_*.yaml` — one per scene, shared by simulation and toolchain
- `configs/eval_test.json` — ~970KB evaluation trajectory definitions
- `openfly_ann/Annotation/seen.json` — annotated trajectories for seen scenes (train)
- `openfly_ann/Annotation/unseen.json` — annotated trajectories for unseen scenes (eval)
- `tool_ws/src/ins_gen/gpt_api_config.json` — GPT API key and model for instruction generation
- Training uses `draccus`/`HfArgumentParser` with dataclasses (`DataArguments`, `TrainingArguments`)

### Environment variables

- `WORLD_SIZE`, `LOCAL_RANK` — set by `torchrun` for distributed training
- `HUGGINGFACE_HUB_TOKEN` — required for gated models (LLaMA-2); also passed via `hf_token` in `TrainingArguments`

### Pretrained weights

- OpenFly-Agent: [huggingface.co/IPEC-COMMUNITY/openfly-agent-7b](https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b)
- Base OpenVLA: [huggingface.co/openvla/openvla-7b-prismatic](https://huggingface.co/openvla/openvla-7b-prismatic)
- Dataset: [huggingface.co/datasets/IPEC-COMMUNITY/OpenFly](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly)
- Scene data: [huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen)

### Import path conventions

The `train/` directory is the Python root for model code. Imports use the form:
```python
from model.config import OpenFlyConfig
from model.prismatic import PrismaticVLM
from datasets import get_vla_dataset_and_collator
from extern.hf.configuration_prismatic import OpenFlyConfig
```

When running training or eval, ensure `train/` is on `PYTHONPATH` or run from within `train/`.
