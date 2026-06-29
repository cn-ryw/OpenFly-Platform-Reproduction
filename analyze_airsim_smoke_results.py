import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SUCCESS_RADIUS = 20.0


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_results(run_dir: Path) -> dict[str, Any]:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json: {results_path}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def classify_sample(item: dict[str, Any]) -> str:
    if item.get("image_error"):
        return "image_error"
    if item.get("success") == 1:
        return "success"
    if item.get("osr") == 1:
        return "near_miss_osr"
    if item.get("steps_run") == item.get("max_steps"):
        return "max_steps_fail"
    return "fail"


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_samples": 0,
            "mean_ne": None,
            "mean_sr": None,
            "mean_osr": None,
            "mean_spl": None,
        }

    action_hist = Counter()
    class_hist = Counter()
    stop_count = 0
    max_step_count = 0
    image_error_count = 0
    for item in results:
        actions = item.get("predicted_action_ids", [])
        action_hist.update(actions)
        class_hist.update([classify_sample(item)])
        if actions and actions[-1] == 0:
            stop_count += 1
        if item.get("steps_run") == item.get("max_steps"):
            max_step_count += 1
        if item.get("image_error"):
            image_error_count += 1

    return {
        "num_samples": len(results),
        "mean_ne": sum(item["distance_to_goal"] for item in results) / len(results),
        "mean_sr": sum(item["success"] for item in results) / len(results),
        "mean_osr": sum(item["osr"] for item in results) / len(results),
        "mean_spl": sum(item["spl"] for item in results) / len(results),
        "action_histogram": dict(sorted(action_hist.items())),
        "sample_class_histogram": dict(sorted(class_hist.items())),
        "stop_count": stop_count,
        "max_step_count": max_step_count,
        "image_error_count": image_error_count,
    }


def sample_summary(item: dict[str, Any]) -> dict[str, Any]:
    actions = item.get("predicted_action_ids", [])
    gt_actions = item.get("ground_truth_action") or []
    return {
        "sample_index": item.get("sample_index"),
        "class": classify_sample(item),
        "image_path": item.get("image_path"),
        "gt_action_len": len(gt_actions),
        "gt_actions": gt_actions,
        "steps_run": item.get("steps_run"),
        "predicted_actions": actions,
        "predicted_action_len": len(actions),
        "stop_predicted": bool(actions and actions[-1] == 0),
        "distance_to_goal": item.get("distance_to_goal"),
        "success": item.get("success"),
        "osr": item.get("osr"),
        "spl": item.get("spl"),
        "image_error": item.get("image_error"),
    }


def write_analysis(run_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    results = payload.get("results", [])
    samples = [sample_summary(item) for item in results]
    summary = aggregate(results)
    analysis = {
        "run_dir": relative_path(run_dir),
        "results_path": relative_path(run_dir / "results.json"),
        "summary_path": relative_path(run_dir / "summary.md"),
        "contact_sheet_path": payload.get("contact_sheet_path"),
        "dry_run": payload.get("dry_run"),
        "env_name": payload.get("env_name"),
        "unnorm_key": payload.get("unnorm_key"),
        "vehicle_name": payload.get("vehicle_name"),
        "max_steps": payload.get("max_steps"),
        "success_radius": SUCCESS_RADIUS,
        "aggregate": summary,
        "samples": samples,
        "notes": [
            "This analysis is for local 4bit AirSim smoke evaluation, not a full OpenFly benchmark.",
            "OSR means the trajectory entered the success radius at least once; SR requires final distance below the radius.",
            "SPL is copied from the local smoke-eval calculation and should be interpreted as preliminary for smoke testing.",
        ],
    }

    analysis_json = run_dir / "analysis.json"
    analysis_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# AirSim Smoke Eval Analysis",
        "",
        "## Scope",
        "",
        "This is a local OpenFly-Agent 4bit AirSim smoke-eval analysis. It is not a full OpenFly benchmark report.",
        "",
        "## Run Metadata",
        "",
        f"- Run dir: `{analysis['run_dir']}`",
        f"- Env: `{analysis['env_name']}`",
        f"- Unnorm key: `{analysis['unnorm_key']}`",
        f"- Vehicle: `{analysis['vehicle_name']}`",
        f"- Max steps: `{analysis['max_steps']}`",
        f"- Contact sheet: `{analysis['contact_sheet_path']}`",
        "",
        "## Aggregate",
        "",
        f"- Samples: `{summary['num_samples']}`",
        f"- Mean NE: `{summary['mean_ne']:.3f}`" if summary["mean_ne"] is not None else "- Mean NE: `None`",
        f"- Mean SR: `{summary['mean_sr']:.4f}`" if summary["mean_sr"] is not None else "- Mean SR: `None`",
        f"- Mean OSR: `{summary['mean_osr']:.4f}`" if summary["mean_osr"] is not None else "- Mean OSR: `None`",
        f"- Mean SPL: `{summary['mean_spl']:.4f}`" if summary["mean_spl"] is not None else "- Mean SPL: `None`",
        f"- Action histogram: `{summary['action_histogram']}`",
        f"- Sample classes: `{summary['sample_class_histogram']}`",
        f"- Stop predicted count: `{summary['stop_count']}`",
        f"- Max-step count: `{summary['max_step_count']}`",
        f"- Image error count: `{summary['image_error_count']}`",
        "",
        "## Per Sample",
        "",
        "| sample | class | steps | GT len | NE | SR | OSR | SPL | stop | predicted actions |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for item in samples:
        lines.append(
            f"| {item['sample_index']} | {item['class']} | {item['steps_run']} | {item['gt_action_len']} | "
            f"{item['distance_to_goal']:.3f} | {item['success']} | {item['osr']} | "
            f"{item['spl']:.4f} | {item['stop_predicted']} | `{item['predicted_actions']}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `success` samples end within the 20m success radius.",
            "- `near_miss_osr` samples entered the radius at some point but did not finish there.",
            "- `max_steps_fail` samples reached the step budget without success.",
        ]
    )

    analysis_md = run_dir / "analysis.md"
    analysis_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return analysis_json, analysis_md


def resolve_run_dir() -> Path:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path
    candidates = [path.parent for path in (ROOT / "demo_results" / "airsim_smoke").glob("*/results.json")]
    if not candidates:
        raise FileNotFoundError("No AirSim smoke results found.")
    return max(candidates, key=lambda path: (path / "results.json").stat().st_mtime)


def main() -> None:
    run_dir = resolve_run_dir()
    payload = load_results(run_dir)
    analysis_json, analysis_md = write_analysis(run_dir, payload)
    print(f"run_dir: {run_dir}")
    print(f"analysis_json: {analysis_json}")
    print(f"analysis_md: {analysis_md}")


if __name__ == "__main__":
    main()
