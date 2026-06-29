import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


MODEL_ID = "IPEC-COMMUNITY/openfly-agent-7b"
UNNORM_KEY = "vln_norm"
BASE_PROMPT = "Fly forward carefully, avoid obstacles, and move towards the open area."
IMAGE_ALIASES = {
    "hist_1": ROOT / "demo_imgs" / "hist_1.png",
    "hist_2": ROOT / "demo_imgs" / "hist_2.png",
    "current": ROOT / "demo_imgs" / "current.png",
}
SCRIPT_PATHS = {
    "single_inference": ROOT / "infer_openfly_4bit.py",
    "experiment_runner": ROOT / "run_openfly_experiments.py",
}


ACTION_TEMPLATES = {
    "0_stop": [1, 0, 0, 0, 0, 0, 0, 0],
    "1_move_forward_3": [0, 3, 0, 0, 0, 0, 0, 0],
    "2_turn_left": [0, 0, 15, 0, 0, 0, 0, 0],
    "3_turn_right": [0, 0, 0, 15, 0, 0, 0, 0],
    "4_go_up": [0, 0, 0, 0, 2, 0, 0, 0],
    "5_go_down": [0, 0, 0, 0, 0, 2, 0, 0],
    "6_move_left": [0, 0, 0, 0, 0, 0, 5, 0],
    "7_move_right": [0, 0, 0, 0, 0, 0, 0, 5],
    "8_move_forward_6": [0, 6, 0, 0, 0, 0, 0, 0],
    "9_move_forward_9": [0, 9, 0, 0, 0, 0, 0, 0],
}


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    prompt: str
    image_order: list[str]
    note: str


class Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc), "args": args}

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "args": args,
    }


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": relative_path(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }
        for name, path in paths.items()
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def print_gpu(label: str) -> None:
    snap = gpu_snapshot()
    if not snap["cuda_available"]:
        print(f"{label}: CUDA unavailable")
        return
    print(
        f"{label}: allocated={snap['allocated_mb']:.1f}MB "
        f"reserved={snap['reserved_mb']:.1f}MB "
        f"max_allocated={snap['max_allocated_mb']:.1f}MB"
    )


def get_git_info() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"])
    branch = run_command(["git", "branch", "--show-current"])
    status = run_command(["git", "status", "--short"])
    remote = run_command(["git", "remote", "-v"])

    status_lines = status["stdout"].splitlines() if status["stdout"] else []
    return {
        "commit": commit["stdout"] if commit["ok"] else None,
        "branch": branch["stdout"] if branch["ok"] else None,
        "dirty": bool(status_lines),
        "status_short": status_lines,
        "remote": remote["stdout"].splitlines() if remote["stdout"] else [],
    }


def get_hf_revision() -> str | None:
    repo_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--IPEC-COMMUNITY--openfly-agent-7b"
    ref_path = repo_cache / "refs" / "main"
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    snapshots = repo_cache / "snapshots"
    if snapshots.exists():
        dirs = sorted(path.name for path in snapshots.iterdir() if path.is_dir())
        if dirs:
            return dirs[-1]
    return None


def get_environment() -> dict[str, Any]:
    import bitsandbytes
    import timm
    import tokenizers
    import transformers

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "timm": timm.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "hf_hub_enable_hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def get_nvidia_smi_snapshot() -> dict[str, Any]:
    gpu = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    apps = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )

    return {
        "gpu_query": gpu,
        "compute_apps_query": apps,
    }


def write_pip_freeze(path: Path) -> None:
    freeze = run_command([sys.executable, "-m", "pip", "freeze"])
    text = freeze["stdout"]
    if freeze["stderr"]:
        text += "\n# stderr:\n" + freeze["stderr"]
    path.write_text(text + "\n", encoding="utf-8")


def summarize_warnings(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    counts = {
        "FutureWarning": text.count("FutureWarning"),
        "UserWarning": text.count("UserWarning"),
        "dependency_version_warning": text.count("Expected `transformers==4.48.1`"),
        "deprecated": text.lower().count("deprecated"),
        "no_op": text.count("no-op"),
        "Traceback": text.count("Traceback"),
        "Exception": text.count("Exception"),
        "out_of_memory": text.lower().count("out of memory"),
        "OOM": text.count("OOM"),
    }
    fatal_count = counts["Traceback"] + counts["Exception"] + counts["out_of_memory"] + counts["OOM"]
    return {
        "counts": counts,
        "non_fatal_warnings_only": fatal_count == 0,
        "notes": [
            "The timm/meta tensor 'no-op' warnings were observed during vision-backbone checkpoint loading.",
            "The run is considered successful only when no Traceback/Exception/OOM appears and all actions are recorded.",
        ],
    }


def validate_inputs() -> None:
    missing = [str(path) for path in IMAGE_ALIASES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing demo image(s): {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; OpenFly 4bit experiments require GPU inference.")

    env = get_environment()
    if env["timm"] != "0.9.16":
        raise RuntimeError(f"Expected timm==0.9.16 for this OpenFly code, got timm=={env['timm']}.")


def build_experiments() -> list[Experiment]:
    normal = ["hist_1", "hist_2", "current"]
    return [
        Experiment("baseline_1", "baseline", BASE_PROMPT, normal, "原始三帧顺序和原 prompt。"),
        Experiment("baseline_2", "baseline", BASE_PROMPT, normal, "完全相同输入复跑，检查 do_sample=False 的稳定性。"),
        Experiment("forward_careful", "prompt", BASE_PROMPT, normal, "forward prompt 对照项。"),
        Experiment("turn_left", "prompt", "Turn left carefully and move toward the open area.", normal, "只改变导航指令为左转。"),
        Experiment("turn_right", "prompt", "Turn right carefully and avoid obstacles.", normal, "只改变导航指令为右转。"),
        Experiment("hover_stop", "prompt", "Stop and hover carefully, avoid moving forward.", normal, "只改变导航指令为停止/悬停。"),
        Experiment("frames_normal", "frames", BASE_PROMPT, normal, "帧顺序正常的帧序对照项。"),
        Experiment(
            "frames_reversed",
            "frames",
            BASE_PROMPT,
            ["current", "hist_2", "hist_1"],
            "反转历史和当前帧，观察时序敏感性。",
        ),
        Experiment(
            "frames_static_current",
            "frames",
            BASE_PROMPT,
            ["current", "current", "current"],
            "三帧都使用 current，观察历史信息被抹掉后的变化。",
        ),
    ]


def load_images_for_experiment(experiment: Experiment) -> list[Image.Image]:
    return [Image.open(IMAGE_ALIASES[alias]).convert("RGB") for alias in experiment.image_order]


def nearest_action_template(action: np.ndarray) -> dict[str, Any]:
    rounded = np.rint(action).astype(float)
    distances = []
    for name, template in ACTION_TEMPLATES.items():
        template_arr = np.array(template, dtype=float)
        diff = rounded - template_arr
        distances.append(
            {
                "template": name,
                "l2_distance": float(np.linalg.norm(diff)),
                "l1_distance": float(np.abs(diff).sum()),
                "template_action": template,
            }
        )
    return min(distances, key=lambda item: item["l2_distance"])


def action_delta(action: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    diff = action - baseline
    return {
        "l1": float(np.abs(diff).sum()),
        "l2": float(np.linalg.norm(diff)),
        "max_abs": float(np.abs(diff).max()),
    }


def register_openfly_classes() -> None:
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


def load_model_and_processor():
    print("[0/5] Registering OpenFly custom classes")
    register_openfly_classes()

    print("[1/5] Loading processor")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("[2/5] Loading model in 4bit")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print_gpu("[After model load]")
    return model, processor


def run_single_experiment(model, processor, experiment: Experiment, baseline: np.ndarray | None) -> dict[str, Any]:
    started = time.perf_counter()
    print(f"[Experiment] {experiment.name}: group={experiment.group}")
    print(f"  prompt: {experiment.prompt}")
    print(f"  images: {experiment.image_order}")

    images = load_images_for_experiment(experiment)
    input_device = next(model.parameters()).device
    inputs = processor(experiment.prompt, images).to(input_device, dtype=torch.float16)
    before = gpu_snapshot()

    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
        )

    action_arr = np.array(action, dtype=float)
    elapsed = time.perf_counter() - started
    after = gpu_snapshot()
    rounded = np.rint(action_arr).astype(int)
    nearest = nearest_action_template(action_arr)
    delta = action_delta(action_arr, baseline) if baseline is not None else {"l1": 0.0, "l2": 0.0, "max_abs": 0.0}

    print(f"  action: {action_arr.tolist()}")
    print(f"  rounded: {rounded.tolist()}")
    print(f"  nearest_template: {nearest['template']} l2={nearest['l2_distance']:.4f}")
    print(f"  delta_vs_baseline: l1={delta['l1']:.6f} l2={delta['l2']:.6f} max={delta['max_abs']:.6f}")
    print(f"  elapsed_sec: {elapsed:.2f}")
    print_gpu("  [GPU]")

    return {
        "name": experiment.name,
        "group": experiment.group,
        "note": experiment.note,
        "prompt": experiment.prompt,
        "image_order": experiment.image_order,
        "image_paths": [relative_path(IMAGE_ALIASES[alias]) for alias in experiment.image_order],
        "action": action_arr.tolist(),
        "rounded_action": rounded.tolist(),
        "nearest_action_template": nearest,
        "delta_vs_baseline_1": delta,
        "elapsed_sec": round(elapsed, 3),
        "gpu_before_inference": before,
        "gpu_after_inference": after,
    }


def write_results_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)


def format_action(action: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in action) + "]"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    baseline = results[0]["action"]
    env = payload["environment"]
    warning_summary = payload["warning_summary"]
    peak_gpu = max(item["gpu_after_inference"].get("max_allocated_mb", 0.0) for item in results)

    lines = [
        "# OpenFly-Agent 4bit 后续实验记录",
        "",
        "## 实验目的",
        "",
        "本轮实验在不训练、不仿真、不下载完整 OpenFly 数据集的前提下，复用当前三张无人机视频截帧和已缓存模型，检查本地 4bit inference 的稳定性，以及输出是否会随 prompt 和帧序变化。",
        "",
        "## 复现元数据",
        "",
        f"- Repo commit: `{payload['git']['commit']}`",
        f"- Git dirty: `{payload['git']['dirty']}`",
        f"- HF model revision: `{payload['model_revision']}`",
        f"- Command: `{payload['run_command']}`",
        f"- pip freeze: `{payload['pip_freeze_path']}`",
        "",
        "## 环境摘要",
        "",
        f"- Model: `{payload['model']}`",
        f"- Unnorm key: `{payload['unnorm_key']}`",
        f"- Python: `{env['python']}`",
        f"- PyTorch: `{env['torch']}`",
        f"- Transformers: `{env['transformers']}`",
        f"- Tokenizers: `{env['tokenizers']}`",
        f"- TIMM: `{env['timm']}`",
        f"- bitsandbytes: `{env['bitsandbytes']}`",
        f"- CUDA device: `{env['cuda_device']}`",
        f"- CUDA capability: `{env['cuda_capability']}`",
        f"- Peak PyTorch allocated GPU memory: `{peak_gpu:.1f} MB`",
        "",
        "## 输入帧",
        "",
        "| alias | path | sha256 |",
        "|---|---|---|",
    ]

    for alias, meta in payload["input_image_hashes"].items():
        lines.append(f"| {alias} | `{meta['path']}` | `{meta['sha256']}` |")

    lines.extend(
        [
            "",
            "## Warning 审计",
            "",
            f"- non_fatal_warnings_only: `{warning_summary['non_fatal_warnings_only']}`",
            f"- FutureWarning: `{warning_summary['counts']['FutureWarning']}`",
            f"- UserWarning: `{warning_summary['counts']['UserWarning']}`",
            f"- dependency version warning: `{warning_summary['counts']['dependency_version_warning']}`",
            f"- no-op warning: `{warning_summary['counts']['no_op']}`",
            f"- Traceback/Exception/OOM: `{warning_summary['counts']['Traceback']}/{warning_summary['counts']['Exception']}/{warning_summary['counts']['OOM']}`",
            "- 这些 warning 会保留在 `run.log` 中；本轮未观察到 Traceback、Exception 或 OOM。",
            "",
            "## 结果表",
            "",
            "| name | group | action | rounded | nearest template | L1 vs baseline | L2 vs baseline | max abs |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )

    for item in results:
        delta = item["delta_vs_baseline_1"]
        lines.append(
            "| "
            f"{item['name']} | "
            f"{item['group']} | "
            f"`{format_action(item['action'])}` | "
            f"`{item['rounded_action']}` | "
            f"{item['nearest_action_template']['template']} | "
            f"{delta['l1']:.6f} | "
            f"{delta['l2']:.6f} | "
            f"{delta['max_abs']:.6f} |"
        )

    baseline_2 = next(item for item in results if item["name"] == "baseline_2")
    prompt_items = [item for item in results if item["group"] == "prompt"]
    frame_items = [item for item in results if item["group"] == "frames"]
    max_prompt_l2 = max(item["delta_vs_baseline_1"]["l2"] for item in prompt_items)
    max_frame_l2 = max(item["delta_vs_baseline_1"]["l2"] for item in frame_items)

    lines.extend(
        [
            "",
            "## 解释",
            "",
            f"- Baseline 复跑差异 L2 = `{baseline_2['delta_vs_baseline_1']['l2']:.6f}`；在 `do_sample=False` 下，如果该值为 0，说明同输入推理是确定性的。",
            f"- Prompt 对照中的最大 L2 差异为 `{max_prompt_l2:.6f}`；它反映同一视觉输入下，语言指令对 action 向量的影响。",
            f"- 帧序对照中的最大 L2 差异为 `{max_frame_l2:.6f}`；它反映历史帧顺序或历史信息变化对 action 向量的影响。",
            f"- Baseline action 为 `{format_action(baseline)}`。",
            "- `nearest template` 来自仓库中的 10 个离散动作模板，只作为辅助解释；当前模型输出仍应视为 8 维连续 action，不应把 nearest template 当作严格分类标签。",
            "",
            "## 注意事项",
            "",
            "- 本实验只验证本地 4bit inference 和输入扰动敏感性，不代表完整 OpenFly benchmark 评测。",
            "- Transformers / bitsandbytes 的 cache warning 或 future warning 不影响本轮结果，除非推理中断或 action 缺失。",
            "- 若后续更换视频帧，应保留同样的 JSON/Markdown 记录结构，便于横向比较。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    run_dir = ROOT / "demo_results" / "experiments" / now_timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    pip_freeze_path = run_dir / "pip_freeze.txt"

    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"[Run dir] {run_dir}")
            print("[Precheck] Validating inputs and environment")
            validate_inputs()
            environment = get_environment()
            git_info = get_git_info()
            model_revision = get_hf_revision()
            nvidia_smi_before = get_nvidia_smi_snapshot()
            write_pip_freeze(pip_freeze_path)

            print(json.dumps(environment, ensure_ascii=False, indent=2))
            print("[Metadata] repo commit:", git_info["commit"])
            print("[Metadata] HF model revision:", model_revision)
            print("[Metadata] pip freeze:", relative_path(pip_freeze_path))

            model, processor = load_model_and_processor()

            print("[3/5] Running experiment matrix")
            experiments = build_experiments()
            results = []
            baseline_action = None

            for experiment in experiments:
                result = run_single_experiment(model, processor, experiment, baseline_action)
                if experiment.name == "baseline_1":
                    baseline_action = np.array(result["action"], dtype=float)
                    result["delta_vs_baseline_1"] = {"l1": 0.0, "l2": 0.0, "max_abs": 0.0}
                results.append(result)

            print("[4/5] Collecting audit metadata")
            log_file.flush()
            warning_summary = summarize_warnings(log_path)
            nvidia_smi_after = get_nvidia_smi_snapshot()

            payload = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "repo_root": str(ROOT),
                "model": MODEL_ID,
                "model_revision": model_revision,
                "unnorm_key": UNNORM_KEY,
                "do_sample": False,
                "git": git_info,
                "environment": environment,
                "input_images": {alias: relative_path(path) for alias, path in IMAGE_ALIASES.items()},
                "input_image_hashes": hash_paths(IMAGE_ALIASES),
                "script_hashes": hash_paths(SCRIPT_PATHS),
                "nvidia_smi_before": nvidia_smi_before,
                "nvidia_smi_after": nvidia_smi_after,
                "pip_freeze_path": relative_path(pip_freeze_path),
                "run_command": " ".join([sys.executable, *sys.argv]),
                "warning_summary": warning_summary,
                "action_template_note": "Nearest template is only an explanatory approximation; model output is an 8D continuous action.",
                "action_templates": ACTION_TEMPLATES,
                "results": results,
                "final_gpu": gpu_snapshot(),
            }

            print("[5/5] Writing results")
            write_results_json(run_dir / "results.json", payload)
            write_summary(run_dir / "summary.md", payload)
            print(f"[Done] results: {run_dir / 'results.json'}")
            print(f"[Done] summary: {run_dir / 'summary.md'}")
            print(f"[Done] log: {log_path}")
            print(f"[Done] pip freeze: {pip_freeze_path}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
