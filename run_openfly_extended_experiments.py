import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from run_openfly_experiments import (
    ACTION_TEMPLATES,
    BASE_PROMPT,
    MODEL_ID,
    ROOT,
    SCRIPT_PATHS,
    UNNORM_KEY,
    Tee,
    action_delta,
    format_action,
    get_environment,
    get_git_info,
    get_hf_revision,
    get_nvidia_smi_snapshot,
    gpu_snapshot,
    hash_paths,
    load_model_and_processor,
    nearest_action_template,
    now_timestamp,
    print_gpu,
    relative_path,
    run_command,
    sha256_file,
    summarize_warnings,
    to_jsonable,
    write_pip_freeze,
    write_results_json,
)


BASE_IMAGE_PATHS = {
    "hist_1": ROOT / "demo_imgs" / "hist_1.png",
    "hist_2": ROOT / "demo_imgs" / "hist_2.png",
    "current": ROOT / "demo_imgs" / "current.png",
}
ANNOTATION_PATHS = {
    "seen": ROOT / "openfly_ann" / "Annotation" / "seen.json",
    "unseen": ROOT / "openfly_ann" / "Annotation" / "unseen.json",
}
VIDEO_PATH = ROOT / "demo_video.mp4"
CENTER_TIMES_SEC = [0.8, 2.0, 3.2, 4.4, 5.6]
EXTENDED_SCRIPT_PATHS = {
    **SCRIPT_PATHS,
    "extended_experiment_runner": ROOT / "run_openfly_extended_experiments.py",
}


@dataclass(frozen=True)
class ExtendedExperiment:
    name: str
    group: str
    prompt: str
    image_paths: list[Path]
    note: str
    prompt_source: dict[str, Any]
    frame_source: dict[str, Any]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs() -> None:
    missing = [str(path) for path in [*BASE_IMAGE_PATHS.values(), *ANNOTATION_PATHS.values(), VIDEO_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input(s): {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; OpenFly 4bit experiments require GPU inference.")

    env = get_environment()
    if env["timm"] != "0.9.16":
        raise RuntimeError(f"Expected timm==0.9.16 for this OpenFly code, got timm=={env['timm']}.")


def select_official_prompts(limit_per_split: int = 5) -> list[dict[str, Any]]:
    selected = []
    for split, path in ANNOTATION_PATHS.items():
        data = load_json(path)
        for index, item in enumerate(data[:limit_per_split]):
            selected.append(
                {
                    "split": split,
                    "index": index,
                    "annotation_path": relative_path(path),
                    "annotation_image_path": item.get("image_path"),
                    "prompt": item["gpt_instruction"],
                    "action_sequence": item.get("action"),
                    "action_sequence_len": len(item.get("action", [])),
                }
            )
    return selected


def video_metadata() -> dict[str, Any]:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "path": relative_path(VIDEO_PATH),
        "sha256": sha256_file(VIDEO_PATH),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps else None,
    }


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> Image.Image:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {VIDEO_PATH}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def extract_video_frame_groups(run_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    meta = video_metadata()
    fps = float(meta["fps"])
    frame_count = int(meta["frame_count"])
    frames_dir = run_dir / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    groups = []
    for group_idx, center_time in enumerate(CENTER_TIMES_SEC):
        center_frame = int(round(center_time * fps))
        center_frame = min(max(center_frame, 2), frame_count - 1)
        frame_indices = [center_frame - 2, center_frame - 1, center_frame]
        group_dir = frames_dir / f"group_{group_idx:02d}_{center_time:.1f}s"
        group_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for alias, frame_idx in zip(["hist_1", "hist_2", "current"], frame_indices):
            image = read_video_frame(cap, frame_idx)
            out_path = group_dir / f"{alias}_frame_{frame_idx:04d}.png"
            image.save(out_path)
            saved_paths.append(out_path)

        groups.append(
            {
                "group_index": group_idx,
                "center_time_sec": center_time,
                "center_frame": center_frame,
                "frame_indices": frame_indices,
                "image_paths": saved_paths,
                "image_hashes": hash_paths({f"frame_{idx}": path for idx, path in zip(frame_indices, saved_paths)}),
            }
        )

    cap.release()
    contact_sheet = write_contact_sheet(groups, run_dir / "video_frame_contact_sheet.png")
    return groups, contact_sheet


def write_contact_sheet(groups: list[dict[str, Any]], out_path: Path) -> Path:
    thumb_w, thumb_h = 320, 180
    label_h = 32
    margin = 12
    cols = 3
    rows = len(groups)
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin, rows * (thumb_h + label_h) + (rows + 1) * margin), "white")
    draw = ImageDraw.Draw(sheet)

    for row, group in enumerate(groups):
        for col, path in enumerate(group["image_paths"]):
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            x = margin + col * (thumb_w + margin)
            y = margin + row * (thumb_h + label_h + margin)
            sheet.paste(img, (x, y))
            label = f"g{group['group_index']} f{group['frame_indices'][col]} {['hist1','hist2','cur'][col]}"
            draw.text((x, y + thumb_h + 4), label, fill=(0, 0, 0))

    sheet.save(out_path)
    return out_path


def load_images(paths: list[Path]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def build_experiments(frame_groups: list[dict[str, Any]]) -> list[ExtendedExperiment]:
    experiments = [
        ExtendedExperiment(
            "baseline_1",
            "baseline",
            BASE_PROMPT,
            [BASE_IMAGE_PATHS["hist_1"], BASE_IMAGE_PATHS["hist_2"], BASE_IMAGE_PATHS["current"]],
            "原始三帧顺序和 baseline prompt。",
            {"type": "manual_baseline"},
            {"type": "demo_imgs", "image_aliases": ["hist_1", "hist_2", "current"]},
        ),
        ExtendedExperiment(
            "baseline_2",
            "baseline",
            BASE_PROMPT,
            [BASE_IMAGE_PATHS["hist_1"], BASE_IMAGE_PATHS["hist_2"], BASE_IMAGE_PATHS["current"]],
            "完全相同输入复跑，检查 do_sample=False 的稳定性。",
            {"type": "manual_baseline"},
            {"type": "demo_imgs", "image_aliases": ["hist_1", "hist_2", "current"]},
        ),
    ]

    for item in select_official_prompts(limit_per_split=5):
        experiments.append(
            ExtendedExperiment(
                name=f"official_{item['split']}_{item['index']}",
                group="official_prompt",
                prompt=item["prompt"],
                image_paths=[BASE_IMAGE_PATHS["hist_1"], BASE_IMAGE_PATHS["hist_2"], BASE_IMAGE_PATHS["current"]],
                note="官方 annotation 风格导航指令；图像仍使用自有三帧，因此只验证语言分布敏感性。",
                prompt_source=item,
                frame_source={"type": "demo_imgs", "image_aliases": ["hist_1", "hist_2", "current"]},
            )
        )

    for group in frame_groups:
        experiments.append(
            ExtendedExperiment(
                name=f"video_frames_{group['group_index']:02d}_{group['center_time_sec']:.1f}s",
                group="video_frames",
                prompt=BASE_PROMPT,
                image_paths=group["image_paths"],
                note="自有视频连续三帧；prompt 固定为 baseline prompt，用于观察视觉变化对 action 的影响。",
                prompt_source={"type": "manual_baseline"},
                frame_source={
                    "type": "demo_video",
                    "center_time_sec": group["center_time_sec"],
                    "center_frame": group["center_frame"],
                    "frame_indices": group["frame_indices"],
                    "image_hashes": group["image_hashes"],
                },
            )
        )

    return experiments


def run_single_experiment(model, processor, experiment: ExtendedExperiment, baseline: np.ndarray | None) -> dict[str, Any]:
    started = time.perf_counter()
    print(f"[Experiment] {experiment.name}: group={experiment.group}")
    print(f"  prompt: {experiment.prompt[:220]}")
    print(f"  images: {[relative_path(path) for path in experiment.image_paths]}")

    images = load_images(experiment.image_paths)
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

    image_hashes = hash_paths({f"image_{idx}": path for idx, path in enumerate(experiment.image_paths)})
    return {
        "name": experiment.name,
        "group": experiment.group,
        "note": experiment.note,
        "prompt": experiment.prompt,
        "prompt_source": experiment.prompt_source,
        "frame_source": experiment.frame_source,
        "image_paths": [relative_path(path) for path in experiment.image_paths],
        "image_hashes": image_hashes,
        "action": action_arr.tolist(),
        "rounded_action": rounded.tolist(),
        "nearest_action_template": nearest,
        "delta_vs_baseline_1": delta,
        "elapsed_sec": round(elapsed, 3),
        "gpu_before_inference": before,
        "gpu_after_inference": after,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    baseline = next(item for item in results if item["name"] == "baseline_1")
    baseline_2 = next(item for item in results if item["name"] == "baseline_2")
    official = [item for item in results if item["group"] == "official_prompt"]
    video = [item for item in results if item["group"] == "video_frames"]
    env = payload["environment"]
    warning_summary = payload["warning_summary"]
    peak_gpu = max(item["gpu_after_inference"].get("max_allocated_mb", 0.0) for item in results)
    max_official_l2 = max(item["delta_vs_baseline_1"]["l2"] for item in official)
    max_video_l2 = max(item["delta_vs_baseline_1"]["l2"] for item in video)

    lines = [
        "# OpenFly-Agent 4bit Extended Experiments",
        "",
        "## 实验目的",
        "",
        "本轮实验补充两类更接近汇报需求的轻量验证：使用 OpenFly annotation 中真实 `gpt_instruction` 风格的 prompt，以及从自有 `demo_video.mp4` 抽取多组连续三帧。实验仍然是本机 4bit inference，不是完整 benchmark 复现。",
        "",
        "## 复现元数据",
        "",
        f"- Repo commit: `{payload['git']['commit']}`",
        f"- Git dirty: `{payload['git']['dirty']}`",
        f"- HF model revision: `{payload['model_revision']}`",
        f"- Command: `{payload['run_command']}`",
        f"- pip freeze: `{payload['pip_freeze_path']}`",
        f"- Contact sheet: `{payload['contact_sheet_path']}`",
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
        f"- Peak PyTorch allocated GPU memory: `{peak_gpu:.1f} MB`",
        "",
        "## 实验数量",
        "",
        f"- Baseline: `{payload['group_counts']['baseline']}`",
        f"- 官方风格 prompt: `{payload['group_counts']['official_prompt']}`",
        f"- 自有视频帧组: `{payload['group_counts']['video_frames']}`",
        "",
        "## Warning 审计",
        "",
        f"- non_fatal_warnings_only: `{warning_summary['non_fatal_warnings_only']}`",
        f"- FutureWarning: `{warning_summary['counts']['FutureWarning']}`",
        f"- UserWarning: `{warning_summary['counts']['UserWarning']}`",
        f"- dependency version warning: `{warning_summary['counts']['dependency_version_warning']}`",
        f"- no-op warning: `{warning_summary['counts']['no_op']}`",
        f"- Traceback/Exception/OOM: `{warning_summary['counts']['Traceback']}/{warning_summary['counts']['Exception']}/{warning_summary['counts']['OOM']}`",
        "",
        "## 结果表",
        "",
        "| name | group | action | rounded | nearest template | L2 vs baseline | max abs |",
        "|---|---|---|---|---|---:|---:|",
    ]

    for item in results:
        delta = item["delta_vs_baseline_1"]
        lines.append(
            "| "
            f"{item['name']} | "
            f"{item['group']} | "
            f"`{format_action(item['action'])}` | "
            f"`{item['rounded_action']}` | "
            f"{item['nearest_action_template']['template']} | "
            f"{delta['l2']:.6f} | "
            f"{delta['max_abs']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 解释",
            "",
            f"- Baseline 复跑差异 L2 = `{baseline_2['delta_vs_baseline_1']['l2']:.6f}`。",
            f"- 官方风格 prompt 最大 L2 差异 = `{max_official_l2:.6f}`。",
            f"- 自有视频帧组最大 L2 差异 = `{max_video_l2:.6f}`。",
            f"- Baseline action = `{format_action(baseline['action'])}`。",
            "- 官方 prompt 与自有视频图像不匹配，因此这些结果只能说明语言分布变化下 action 输出是否敏感，不能作为官方轨迹评测指标。",
            "- 视频帧实验固定 prompt，只观察视觉输入变化对 action 的影响。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    run_dir = ROOT / "demo_results" / "extended_experiments" / now_timestamp()
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

            video_meta = video_metadata()
            frame_groups, contact_sheet = extract_video_frame_groups(run_dir)
            experiments = build_experiments(frame_groups)

            print(json.dumps(environment, ensure_ascii=False, indent=2))
            print("[Metadata] repo commit:", git_info["commit"])
            print("[Metadata] HF model revision:", model_revision)
            print("[Metadata] video:", video_meta)
            print("[Metadata] extracted frame groups:", len(frame_groups))
            print("[Metadata] contact sheet:", contact_sheet)

            model, processor = load_model_and_processor()

            print("[3/5] Running extended experiment matrix")
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
            group_counts = {
                "baseline": sum(item.group == "baseline" for item in experiments),
                "official_prompt": sum(item.group == "official_prompt" for item in experiments),
                "video_frames": sum(item.group == "video_frames" for item in experiments),
            }

            payload = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "experiment_type": "extended_official_prompt_and_video_frames",
                "repo_root": str(ROOT),
                "model": MODEL_ID,
                "model_revision": model_revision,
                "unnorm_key": UNNORM_KEY,
                "do_sample": False,
                "git": git_info,
                "environment": environment,
                "base_input_images": {alias: relative_path(path) for alias, path in BASE_IMAGE_PATHS.items()},
                "base_input_image_hashes": hash_paths(BASE_IMAGE_PATHS),
                "annotation_paths": {name: relative_path(path) for name, path in ANNOTATION_PATHS.items()},
                "video_metadata": video_meta,
                "video_frame_groups": [
                    {
                        **group,
                        "image_paths": [relative_path(path) for path in group["image_paths"]],
                    }
                    for group in frame_groups
                ],
                "contact_sheet_path": relative_path(contact_sheet),
                "script_hashes": hash_paths(EXTENDED_SCRIPT_PATHS),
                "nvidia_smi_before": nvidia_smi_before,
                "nvidia_smi_after": nvidia_smi_after,
                "pip_freeze_path": relative_path(pip_freeze_path),
                "run_command": " ".join([sys.executable, *sys.argv]),
                "warning_summary": warning_summary,
                "group_counts": group_counts,
                "action_template_note": "Nearest template is only an explanatory approximation; model output is an 8D continuous action.",
                "action_templates": ACTION_TEMPLATES,
                "results": results,
                "final_gpu": gpu_snapshot(),
                "limitations": [
                    "Official-style prompts are paired with local demo video frames, not their original OpenFly images.",
                    "Video-frame experiments use a fixed prompt and only test visual sensitivity.",
                    "This is not a benchmark evaluation and does not produce NE/SR/OSR/SPL.",
                ],
            }

            print("[5/5] Writing results")
            write_results_json(run_dir / "results.json", payload)
            write_summary(run_dir / "summary.md", payload)
            print(f"[Done] results: {run_dir / 'results.json'}")
            print(f"[Done] summary: {run_dir / 'summary.md'}")
            print(f"[Done] log: {log_path}")
            print(f"[Done] pip freeze: {pip_freeze_path}")
            print(f"[Done] contact sheet: {contact_sheet}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
