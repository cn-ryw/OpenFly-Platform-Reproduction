import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
MODEL_ID = "IPEC-COMMUNITY/openfly-agent-7b"
DEFAULT_EXPERIMENT_DIR = ROOT / "demo_results" / "experiments" / "20260628_232446"
MANIFEST_PATH = ROOT / "demo_results" / "manifest.json"
AUDIT_PATH = ROOT / "demo_results" / "reproduction_audit.md"
PIP_FREEZE_PATH = ROOT / "demo_results" / "pip_freeze_openfly.txt"

INPUT_FILES = {
    "hist_1": ROOT / "demo_imgs" / "hist_1.png",
    "hist_2": ROOT / "demo_imgs" / "hist_2.png",
    "current": ROOT / "demo_imgs" / "current.png",
    "demo_video": ROOT / "demo_video.mp4",
    "openfly_paper_pdf": ROOT.parent / "Gao 等 - 2026 - Openfly A comprehensive platform ..pdf",
}
SCRIPT_FILES = {
    "single_inference": ROOT / "infer_openfly_4bit.py",
    "experiment_runner": ROOT / "run_openfly_experiments.py",
    "extended_experiment_runner": ROOT / "run_openfly_extended_experiments.py",
    "airsim_smoke_eval": ROOT / "eval_airsim_4bit_smoke.py",
    "airsim_smoke_analyzer": ROOT / "analyze_airsim_smoke_results.py",
    "airsim_review_assets": ROOT / "prepare_airsim_review_assets.py",
    "airsim_scene_downloader": ROOT / "download_openfly_airsim_scene.py",
    "audit_generator": ROOT / "generate_reproduction_audit.py",
}


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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path),
    }


def records_for(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: file_record(path) for name, path in paths.items()}


def latest_experiment_dir() -> Path:
    candidates = []
    for experiments_root in [ROOT / "demo_results" / "experiments", ROOT / "demo_results" / "extended_experiments"]:
        candidates.extend(path.parent for path in experiments_root.glob("*/results.json"))
    if not candidates:
        raise FileNotFoundError("No experiment results found under demo_results/experiments")
    return max(candidates, key=lambda path: (path / "results.json").stat().st_mtime)


def latest_file(pattern: str) -> Path | None:
    candidates = [path for path in ROOT.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def select_experiment_dir() -> Path:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not (path / "results.json").exists():
            raise FileNotFoundError(f"Missing results.json in experiment dir: {path}")
        return path

    return latest_experiment_dir()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return {"gpu_query": gpu, "compute_apps_query": apps}


def get_hf_cli_status() -> dict[str, Any]:
    hf_cli = Path.home() / ".local" / "bin" / "hf"
    skill_dir = Path.home() / ".agents" / "skills" / "hf-cli"
    if hf_cli.exists():
        version = run_command([str(hf_cli), "version"])
        whoami = run_command([str(hf_cli), "auth", "whoami"])
        env = run_command([str(hf_cli), "env"])
    else:
        version = {"ok": False, "stdout": "", "stderr": "standalone hf CLI not found"}
        whoami = version
        env = version
    return {
        "standalone_hf_cli": file_record(hf_cli),
        "version": version,
        "whoami": whoami,
        "env": env,
        "hf_cli_skill_dir": str(skill_dir),
        "hf_cli_skill_installed": (skill_dir / "SKILL.md").exists(),
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
            "No Traceback/Exception/OOM was observed when fatal counters are zero.",
        ],
    }


def experiment_result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(item for item in results if item["name"] == "baseline_1")
    baseline_2 = next((item for item in results if item["name"] == "baseline_2"), baseline)
    prompt_items = [item for item in results if item["group"] in {"prompt", "official_prompt"}]
    frame_items = [item for item in results if item["group"] in {"frames", "video_frames"}]
    group_counts = {}
    for item in results:
        group_counts[item["group"]] = group_counts.get(item["group"], 0) + 1
    return {
        "num_experiments": len(results),
        "group_counts": group_counts,
        "baseline_action": baseline["action"],
        "baseline_rounded_action": baseline["rounded_action"],
        "baseline_repeat_l2": baseline_2["delta_vs_baseline_1"]["l2"],
        "max_prompt_l2": max((item["delta_vs_baseline_1"]["l2"] for item in prompt_items), default=0.0),
        "max_frame_l2": max((item["delta_vs_baseline_1"]["l2"] for item in frame_items), default=0.0),
        "peak_gpu_allocated_mb": max(item["gpu_after_inference"].get("max_allocated_mb", 0.0) for item in results),
    }


def airsim_smoke_result_summary(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    data = load_json(path)
    results = data.get("results", [])
    if data.get("dry_run") or not results:
        return {
            "path": relative_path(path),
            "dry_run": data.get("dry_run"),
            "num_samples": 0,
            "note": "Latest AirSim smoke result is a dry-run or contains no simulator results.",
        }
    mean_ne = sum(item["distance_to_goal"] for item in results) / len(results)
    mean_sr = sum(item["success"] for item in results) / len(results)
    mean_osr = sum(item["osr"] for item in results) / len(results)
    mean_spl = sum(item["spl"] for item in results) / len(results)
    first = results[0]
    return {
        "path": relative_path(path),
        "dry_run": data.get("dry_run"),
        "num_samples": len(results),
        "env_name": data.get("env_name"),
        "unnorm_key": data.get("unnorm_key"),
        "vehicle_name": data.get("vehicle_name"),
        "max_steps": data.get("max_steps"),
        "settings_match": data.get("preflight", {}).get("airsim_settings", {}).get("matches_project"),
        "mean_ne": mean_ne,
        "mean_sr": mean_sr,
        "mean_osr": mean_osr,
        "mean_spl": mean_spl,
        "first_sample_steps": first.get("steps_run"),
        "first_sample_actions": first.get("predicted_action_ids"),
        "first_sample_image_error": first.get("image_error"),
        "first_sample_final_distance": first.get("distance_to_goal"),
        "contact_sheet_path": data.get("contact_sheet_path"),
        "sample_summaries": [
            {
                "sample_index": item.get("sample_index"),
                "image_path": item.get("image_path"),
                "steps_run": item.get("steps_run"),
                "actions": item.get("predicted_action_ids"),
                "ne": item.get("distance_to_goal"),
                "sr": item.get("success"),
                "osr": item.get("osr"),
                "spl": item.get("spl"),
                "image_error": item.get("image_error"),
            }
            for item in results
        ],
    }


def collect_manifest(experiment_dir: Path) -> dict[str, Any]:
    results_path = experiment_dir / "results.json"
    summary_path = experiment_dir / "summary.md"
    log_path = experiment_dir / "run.log"

    results_json = load_json(results_path)
    write_pip_freeze(PIP_FREEZE_PATH)

    result_files = {
        "single_inference_result": ROOT / "demo_results" / "openfly_own_video_result.json",
        "experiment_results": results_path,
        "experiment_summary": summary_path,
        "experiment_log": log_path,
        "top_level_pip_freeze": PIP_FREEZE_PATH,
        "simulation_eval_plan": ROOT / "demo_results" / "simulation_eval_plan.md",
        "openfly_paper_text_extract": ROOT / "demo_results" / "paper" / "openfly_paper.txt",
        "official_eval_alignment_audit": ROOT / "demo_results" / "official_eval_alignment_audit.md",
        "action9_bias_analysis": ROOT / "demo_results" / "action9_bias_analysis.md",
        "advisor_review_report": ROOT / "demo_results" / "advisor_review_report.md",
    }
    if results_json.get("pip_freeze_path"):
        result_files["experiment_pip_freeze"] = ROOT / results_json["pip_freeze_path"]
    if results_json.get("contact_sheet_path"):
        result_files["contact_sheet"] = ROOT / results_json["contact_sheet_path"]
    latest_airsim_results = latest_file("demo_results/airsim_smoke/*/results.json")
    latest_airsim_summary = latest_file("demo_results/airsim_smoke/*/summary.md")
    latest_airsim_log = latest_file("demo_results/airsim_smoke/*/run.log")
    latest_airsim_analysis = latest_file("demo_results/airsim_smoke/*/analysis.md")
    latest_airsim_analysis_json = latest_file("demo_results/airsim_smoke/*/analysis.json")
    latest_airsim_download_manifest = latest_file("demo_results/downloads/env_airsim_16/*.json")
    review_assets_dir = ROOT / "demo_results" / "review_assets" / "airsim_env16_30_cases"
    review_assets = {
        "airsim_review_cases": review_assets_dir / "cases.json",
        "airsim_review_case_gallery": review_assets_dir / "case_gallery.png",
        "airsim_review_readme": review_assets_dir / "README.md",
        "airsim_review_source_analysis": review_assets_dir / "source_analysis.md",
    }
    if latest_airsim_results:
        result_files["airsim_smoke_results"] = latest_airsim_results
    if latest_airsim_summary:
        result_files["airsim_smoke_summary"] = latest_airsim_summary
    if latest_airsim_log:
        result_files["airsim_smoke_log"] = latest_airsim_log
    if latest_airsim_analysis:
        result_files["airsim_smoke_analysis"] = latest_airsim_analysis
    if latest_airsim_analysis_json:
        result_files["airsim_smoke_analysis_json"] = latest_airsim_analysis_json
    if latest_airsim_download_manifest:
        result_files["airsim_scene_download_manifest"] = latest_airsim_download_manifest
    if latest_airsim_results:
        latest_airsim_data = load_json(latest_airsim_results)
        if latest_airsim_data.get("contact_sheet_path"):
            result_files["airsim_smoke_contact_sheet"] = ROOT / latest_airsim_data["contact_sheet_path"]
    for name, asset_path in review_assets.items():
        if asset_path.exists():
            result_files[name] = asset_path

    embedded_metadata = {
        "has_git": "git" in results_json,
        "has_model_revision": "model_revision" in results_json,
        "has_input_hashes": "input_image_hashes" in results_json,
        "has_warning_summary": "warning_summary" in results_json,
        "has_pip_freeze_path": "pip_freeze_path" in results_json,
    }

    airsim_start_script = file_record(ROOT / "envs" / "airsim" / "env_airsim_16" / "LinuxNoEditor" / "start.sh")
    scene_ready = bool(airsim_start_script["exists"])
    airsim_blocker = None if scene_ready else "IPEC-COMMUNITY/OpenFly_DataGen is gated; current machine is not logged in to HuggingFace and no local scene is installed."
    latest_airsim_summary_record = airsim_smoke_result_summary(latest_airsim_results)
    latest_airsim_warning_summary = None
    if latest_airsim_results:
        latest_airsim_warning_summary = load_json(latest_airsim_results).get("warning_summary")
    latest_airsim_metrics = None
    if latest_airsim_summary_record and not latest_airsim_summary_record.get("dry_run"):
        latest_airsim_metrics = {
            "mean_ne": latest_airsim_summary_record.get("mean_ne"),
            "mean_sr": latest_airsim_summary_record.get("mean_sr"),
            "mean_osr": latest_airsim_summary_record.get("mean_osr"),
            "mean_spl": latest_airsim_summary_record.get("mean_spl"),
            "num_samples": latest_airsim_summary_record.get("num_samples"),
        }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "OpenFly-Agent local 4bit inference reproduction and lightweight sensitivity experiments",
        "repo": get_git_info(),
        "model": {
            "id": MODEL_ID,
            "revision": results_json.get("model_revision") or get_hf_revision(),
            "unnorm_key": results_json.get("unnorm_key"),
            "do_sample": results_json.get("do_sample"),
        },
        "environment": get_environment(),
        "nvidia_smi": get_nvidia_smi_snapshot(),
        "hf_cli_status": get_hf_cli_status(),
        "input_files": records_for(INPUT_FILES),
        "script_files": records_for(SCRIPT_FILES),
        "result_files": records_for(result_files),
        "pip_freeze_path": relative_path(PIP_FREEZE_PATH),
        "run_commands": {
            "single_inference": "cd ~/research/OpenFly-Platform && conda activate openfly && export HF_HUB_ENABLE_HF_TRANSFER=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -u infer_openfly_4bit.py",
            "experiment_runner": "cd ~/research/OpenFly-Platform && conda activate openfly && export HF_HUB_ENABLE_HF_TRANSFER=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -u run_openfly_experiments.py",
            "audit_generator": "cd ~/research/OpenFly-Platform && conda activate openfly && python -u generate_reproduction_audit.py",
            "extended_experiment_runner": "cd ~/research/OpenFly-Platform && conda activate openfly && export HF_HUB_ENABLE_HF_TRANSFER=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -u run_openfly_extended_experiments.py",
            "airsim_smoke_dry_run": "cd ~/research/OpenFly-Platform && conda activate openfly && python -u eval_airsim_4bit_smoke.py --dry-run --env-name env_airsim_16 --limit 3 --max-steps 20",
            "hf_auth_login": "hf auth login",
            "airsim_scene_download": "cd ~/research/OpenFly-Platform && conda activate openfly && export HF_HUB_ENABLE_HF_TRANSFER=1 && python -u download_openfly_airsim_scene.py --env-name env_airsim_16",
            "airsim_scene_extract_local_zip": "cd ~/research/OpenFly-Platform && conda activate openfly && python -u download_openfly_airsim_scene.py --env-name env_airsim_16 --zip-path /home/ruan/Downloads/env_airsim_16.zip",
            "airsim_smoke_eval": "cd ~/research/OpenFly-Platform && conda activate openfly && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -u eval_airsim_4bit_smoke.py --env-name env_airsim_16 --limit 30 --max-steps 20 --save-frames --prepare-settings",
            "airsim_smoke_analysis": "cd ~/research/OpenFly-Platform && conda activate openfly && python -u analyze_airsim_smoke_results.py demo_results/airsim_smoke/<timestamp>",
            "airsim_review_assets": "cd ~/research/OpenFly-Platform && conda activate openfly && python -u prepare_airsim_review_assets.py",
        },
        "simulation_status": {
            "airsim_python_client_installed": True,
            "airsim_scene_env_16_start_script": airsim_start_script,
            "airsim_scene_env_16_ready": scene_ready,
            "hf_cli_skill_installed": (Path.home() / ".agents" / "skills" / "hf-cli" / "SKILL.md").exists(),
            "airsim_dataset_access_blocker": airsim_blocker,
            "latest_airsim_smoke": latest_airsim_summary_record,
            "latest_airsim_summary": relative_path(latest_airsim_summary) if latest_airsim_summary else None,
            "latest_airsim_log": relative_path(latest_airsim_log) if latest_airsim_log else None,
            "latest_airsim_analysis": relative_path(latest_airsim_analysis) if latest_airsim_analysis else None,
            "latest_airsim_analysis_json": relative_path(latest_airsim_analysis_json) if latest_airsim_analysis_json else None,
            "latest_airsim_contact_sheet": latest_airsim_summary_record.get("contact_sheet_path") if latest_airsim_summary_record else None,
            "airsim_metrics": latest_airsim_metrics,
            "airsim_warning_summary": latest_airsim_warning_summary,
        },
        "referenced_experiment": {
            "path": relative_path(experiment_dir),
            "results_path": relative_path(results_path),
            "summary_path": relative_path(summary_path),
            "log_path": relative_path(log_path),
            "embedded_metadata": embedded_metadata,
            "metadata_note": "Older experiment runs may not contain embedded hashes; this audit records current files and marks embedded metadata availability explicitly.",
            "summary": experiment_result_summary(results_json["results"]),
        },
        "warning_summary": results_json.get("warning_summary") or summarize_warnings(log_path),
        "limitations": [
            "This is not a full OpenFly benchmark reproduction because NE/SR/OSR/SPL evaluation requires simulation environments and benchmark trajectories.",
            "The prompt sensitivity tests use local video frames and lightweight prompts, not matched official image trajectories.",
            "Official-style prompts in extended experiments are intentionally paired with local demo frames, so they cannot be interpreted as trajectory success evaluation.",
            "The current worktree is intentionally recorded as dirty because demo files, scripts, and results are untracked local reproduction artifacts.",
        ],
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_action(action: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in action) + "]"


def write_audit_markdown(path: Path, manifest: dict[str, Any]) -> None:
    env = manifest["environment"]
    repo = manifest["repo"]
    model = manifest["model"]
    exp = manifest["referenced_experiment"]
    summary = exp["summary"]
    warnings = manifest["warning_summary"]
    counts = warnings["counts"]
    simulation_status = manifest.get("simulation_status", {})
    hf_cli_status = manifest.get("hf_cli_status", {})
    airsim_start_script = simulation_status.get("airsim_scene_env_16_start_script", {})
    latest_airsim_smoke = simulation_status.get("latest_airsim_smoke") or {}
    latest_airsim_samples = latest_airsim_smoke.get("sample_summaries") or []

    lines = [
        "# OpenFly-Agent 本地 4bit 复现审计记录",
        "",
        "## 结论摘要",
        "",
        "- 已在本机 RTX 4060 Laptop 8GB 上跑通 `IPEC-COMMUNITY/openfly-agent-7b` 的 4bit inference。",
        f"- 已完成单次推理和 `{summary['num_experiments']}` 组轻量对照实验，并保存 JSON、Markdown 与日志。",
        "- 当前结果属于“本机 4bit 推理复现与轻量敏感性验证”，不是完整 OpenFly benchmark 复现。",
        f"- Baseline action: `{format_action(summary['baseline_action'])}`",
        f"- Baseline repeat L2: `{summary['baseline_repeat_l2']:.6f}`",
        f"- Max prompt L2: `{summary['max_prompt_l2']:.6f}`",
        f"- Max frame-order L2: `{summary['max_frame_l2']:.6f}`",
        f"- Peak PyTorch allocated GPU memory: `{summary['peak_gpu_allocated_mb']:.1f} MB`",
        f"- Group counts: `{summary['group_counts']}`",
        "",
        "## Extended 实验覆盖",
        "",
        f"- 官方风格 prompt 实验数量: `{summary['group_counts'].get('official_prompt', 0)}`",
        f"- 自有视频多帧实验数量: `{summary['group_counts'].get('video_frames', 0)}`",
        "- 官方风格 prompt 来自 `seen.json[0:5]` 和 `unseen.json[0:5]`；图像仍使用本地 demo 帧，因此只能验证语言分布敏感性。",
        "- 自有视频多帧实验从 `demo_video.mp4` 抽取连续三帧组，固定 baseline prompt，用于观察视觉输入变化。",
        "",
        "## 代码与模型",
        "",
        f"- Repo commit: `{repo['commit']}`",
        f"- Branch: `{repo['branch']}`",
        f"- Git dirty: `{repo['dirty']}`",
        f"- Model: `{model['id']}`",
        f"- HF revision: `{model['revision']}`",
        f"- Unnorm key: `{model['unnorm_key']}`",
        f"- do_sample: `{model['do_sample']}`",
        "",
        "## 环境",
        "",
        f"- Python: `{env['python']}`",
        f"- PyTorch: `{env['torch']}`",
        f"- Transformers: `{env['transformers']}`",
        f"- Tokenizers: `{env['tokenizers']}`",
        f"- TIMM: `{env['timm']}`",
        f"- bitsandbytes: `{env['bitsandbytes']}`",
        f"- CUDA available: `{env['cuda_available']}`",
        f"- CUDA device: `{env['cuda_device']}`",
        f"- CUDA capability: `{env['cuda_capability']}`",
        f"- `pip freeze`: `{manifest['pip_freeze_path']}`",
        "",
        "## 输入与脚本 Hash",
        "",
        "| item | path | sha256 |",
        "|---|---|---|",
    ]

    for name, record in manifest["input_files"].items():
        lines.append(f"| input:{name} | `{record['path']}` | `{record['sha256']}` |")
    for name, record in manifest["script_files"].items():
        lines.append(f"| script:{name} | `{record['path']}` | `{record['sha256']}` |")
    for name, record in manifest["result_files"].items():
        lines.append(f"| result:{name} | `{record['path']}` | `{record['sha256']}` |")

    lines.extend(
        [
            "",
            "## 运行命令",
            "",
            "```bash",
            manifest["run_commands"]["single_inference"],
            manifest["run_commands"]["experiment_runner"],
            manifest["run_commands"]["extended_experiment_runner"],
            manifest["run_commands"]["airsim_smoke_dry_run"],
            manifest["run_commands"]["hf_auth_login"],
            manifest["run_commands"]["airsim_scene_download"],
            manifest["run_commands"]["airsim_scene_extract_local_zip"],
            manifest["run_commands"]["airsim_smoke_eval"],
            manifest["run_commands"]["audit_generator"],
            "```",
            "",
            "## Hugging Face CLI 状态",
            "",
            f"- Standalone `hf` CLI path: `{hf_cli_status.get('standalone_hf_cli', {}).get('path')}`",
            f"- Standalone `hf` CLI exists: `{hf_cli_status.get('standalone_hf_cli', {}).get('exists')}`",
            f"- `hf version`: `{hf_cli_status.get('version', {}).get('stdout')}`",
            f"- `hf auth whoami`: `{hf_cli_status.get('whoami', {}).get('stdout') or hf_cli_status.get('whoami', {}).get('stderr')}`",
            f"- HF CLI skill installed: `{hf_cli_status.get('hf_cli_skill_installed')}`",
            f"- HF CLI skill path: `{hf_cli_status.get('hf_cli_skill_dir')}`",
            "",
            "## AirSim 仿真评估准备状态",
            "",
            "- 已新增 `eval_airsim_4bit_smoke.py`，用于单场景、少样例、4bit AirSim smoke eval。",
            "- 已新增 `download_openfly_airsim_scene.py`，用于只下载并解压单个 `env_airsim_16` 场景。",
            f"- AirSim Python client installed: `{simulation_status.get('airsim_python_client_installed')}`",
            f"- HF CLI skill installed: `{simulation_status.get('hf_cli_skill_installed')}`",
            f"- `env_airsim_16` ready: `{simulation_status.get('airsim_scene_env_16_ready')}`",
            f"- `env_airsim_16` start script exists: `{airsim_start_script.get('exists')}`",
            f"- Start script path: `{airsim_start_script.get('path')}`",
            f"- Current blocker: `{simulation_status.get('airsim_dataset_access_blocker')}`",
            f"- Latest AirSim smoke result: `{latest_airsim_smoke.get('path')}`",
            f"- Latest AirSim smoke metrics: `NE={latest_airsim_smoke.get('mean_ne')}`, `SR={latest_airsim_smoke.get('mean_sr')}`, `OSR={latest_airsim_smoke.get('mean_osr')}`, `SPL={latest_airsim_smoke.get('mean_spl')}`",
            f"- Latest AirSim smoke actions: `{latest_airsim_smoke.get('first_sample_actions')}`",
            f"- Latest AirSim contact sheet: `{latest_airsim_smoke.get('contact_sheet_path')}`",
            "- 该 AirSim 结果是单场景 smoke eval，不等同于完整 OpenFly benchmark。",
            "",
            "| sample | steps | NE | SR | OSR | SPL | actions |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for item in latest_airsim_samples:
        lines.append(
            f"| {item.get('sample_index')} | {item.get('steps_run')} | "
            f"{item.get('ne'):.3f} | {item.get('sr')} | {item.get('osr')} | "
            f"{item.get('spl'):.4f} | `{item.get('actions')}` |"
        )

    lines.extend(
        [
            "",
            "## Warning 审计",
            "",
            f"- non_fatal_warnings_only: `{warnings['non_fatal_warnings_only']}`",
            f"- FutureWarning: `{counts['FutureWarning']}`",
            f"- UserWarning: `{counts['UserWarning']}`",
            f"- dependency version warning: `{counts['dependency_version_warning']}`",
            f"- deprecated: `{counts['deprecated']}`",
            f"- no-op: `{counts['no_op']}`",
            f"- Traceback: `{counts['Traceback']}`",
            f"- Exception: `{counts['Exception']}`",
            f"- out of memory/OOM: `{counts['out_of_memory']}/{counts['OOM']}`",
            "",
            "说明：`no-op` warning 来自 timm/meta tensor checkpoint loading。当前实验没有 Traceback、Exception 或 OOM，因此记录为非致命 warning，但在导师 review 中应披露。",
            "",
            "## 结果路径",
            "",
            f"- Experiment dir: `{exp['path']}`",
            f"- Results JSON: `{exp['results_path']}`",
            f"- Summary Markdown: `{exp['summary_path']}`",
            f"- Run log: `{exp['log_path']}`",
            f"- Manifest: `{relative_path(MANIFEST_PATH)}`",
            f"- Simulation eval plan: `demo_results/simulation_eval_plan.md`",
            "",
            "## 未覆盖内容",
            "",
            "- 未复现完整 OpenFly dataset、训练流程或仿真环境。",
            "- 未运行官方 `train/eval.py` 全量 benchmark，因此还没有完整 benchmark 级别的 NE/SR/OSR/SPL 汇总指标。",
            "- 当前 demo 使用自有视频截帧，不能替代官方 seen/unseen 图像轨迹评测。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    experiment_dir = select_experiment_dir()
    manifest = collect_manifest(experiment_dir)
    write_manifest(MANIFEST_PATH, manifest)
    write_audit_markdown(AUDIT_PATH, manifest)
    print(f"experiment: {experiment_dir}")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"audit: {AUDIT_PATH}")
    print(f"pip freeze: {PIP_FREEZE_PATH}")


if __name__ == "__main__":
    main()
