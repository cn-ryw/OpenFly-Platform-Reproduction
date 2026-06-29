import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "demo_results" / "airsim_smoke" / "20260629_005458"
OUT_ROOT = ROOT / "demo_results" / "review_assets" / "airsim_env16_30_cases"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def distances(result: dict[str, Any]) -> list[float]:
    return [float(step["distance_to_goal"]) for step in result.get("action_steps", [])]


def min_distance(result: dict[str, Any]) -> float | None:
    values = distances(result)
    return min(values) if values else None


def best_step(result: dict[str, Any]) -> int:
    values = distances(result)
    if not values:
        return 0
    return min(range(len(values)), key=lambda idx: values[idx])


def image_for_step(run_dir: Path, sample_index: int, step: int) -> Path:
    return run_dir / f"sample_{sample_index:02d}" / f"step_{step:03d}.png"


def pick_cases(results: list[dict[str, Any]], analysis_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index = {item["sample_index"]: item for item in results}
    analysis_by_index = {item["sample_index"]: item for item in analysis_samples}

    successes = [item for item in results if item.get("success") == 1]
    osr_failures = [item for item in results if item.get("osr") == 1 and item.get("success") == 0]
    immediate_stops = [
        item
        for item in results
        if item.get("steps_run") == 1 and item.get("predicted_action_ids") == [0] and item.get("success") == 0
    ]
    max_step_failures = [
        item
        for item in results
        if item.get("steps_run") == item.get("max_steps") and item.get("success") == 0
    ]
    mixed_turn_failures = [
        item
        for item in results
        if item.get("success") == 0 and any(action in {1, 2, 3} for action in item.get("predicted_action_ids", []))
    ]

    chosen: list[tuple[str, str, dict[str, Any]]] = []

    if successes:
        item = min(successes, key=lambda sample: sample.get("distance_to_goal", 9999))
        chosen.append(("success_best", "Successful stop inside the 20m success radius.", item))

    if successes:
        item = max(successes, key=lambda sample: len(set(sample.get("predicted_action_ids", []))))
        if item["sample_index"] not in {case[2]["sample_index"] for case in chosen}:
            chosen.append(("success_with_turns", "Successful sample that used more than only forward/stop.", item))

    if osr_failures:
        item = max(
            osr_failures,
            key=lambda sample: sample.get("distance_to_goal", 0.0) - (min_distance(sample) or 0.0),
        )
        chosen.append(("near_miss_overshoot", "Entered the success radius but ended far away.", item))

    if immediate_stops:
        item = max(immediate_stops, key=lambda sample: sample.get("distance_to_goal", 0.0))
        chosen.append(("immediate_stop_fail", "Predicted stop at step 0 while still far from target.", item))

    if max_step_failures:
        item = max(
            max_step_failures,
            key=lambda sample: sample.get("predicted_action_ids", []).count(9),
        )
        chosen.append(("max_steps_forward_bias", "Hit max steps with a strong action-9 forward bias.", item))

    if mixed_turn_failures:
        item = max(
            mixed_turn_failures,
            key=lambda sample: len([a for a in sample.get("predicted_action_ids", []) if a in {1, 2, 3}]),
        )
        if item["sample_index"] not in {case[2]["sample_index"] for case in chosen}:
            chosen.append(("mixed_turn_near_miss", "Failure case with several turn/vertical actions.", item))

    cases = []
    for label, note, item in chosen:
        sample_index = item["sample_index"]
        steps_run = int(item.get("steps_run") or 1)
        first_step = 0
        closest_step = best_step(item)
        last_step = max(0, steps_run - 1)
        cases.append(
            {
                "label": label,
                "note": note,
                "sample_index": sample_index,
                "analysis_class": analysis_by_index.get(sample_index, {}).get("class"),
                "image_path": item.get("image_path"),
                "steps_run": steps_run,
                "gt_len": item.get("ground_truth_action_len"),
                "ne": item.get("distance_to_goal"),
                "min_ne": min_distance(item),
                "sr": item.get("success"),
                "osr": item.get("osr"),
                "spl": item.get("spl"),
                "actions": item.get("predicted_action_ids"),
                "frames": [
                    {"name": "first", "step": first_step},
                    {"name": "closest", "step": closest_step},
                    {"name": "last", "step": last_step},
                ],
            }
        )
    return cases


def make_gallery(run_dir: Path, cases: list[dict[str, Any]], out_path: Path) -> None:
    thumb_w, thumb_h = 320, 180
    label_h = 56
    gap = 12
    row_h = thumb_h + label_h
    col_w = thumb_w
    width = 3 * col_w + 4 * gap
    height = len(cases) * row_h + (len(cases) + 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for row, case in enumerate(cases):
        y = gap + row * (row_h + gap)
        header = (
            f"{case['label']} | sample {case['sample_index']:02d} | "
            f"class={case['analysis_class']} | NE={case['ne']:.2f} | "
            f"min={case['min_ne']:.2f} | actions={case['actions'][:8]}"
        )
        draw.text((gap, y), header, fill=(20, 20, 20))
        for col, frame in enumerate(case["frames"]):
            img_path = image_for_step(run_dir, case["sample_index"], frame["step"])
            x = gap + col * (col_w + gap)
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb_w, thumb_h))
                tile = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
                tile.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
            else:
                tile = Image.new("RGB", (thumb_w, thumb_h), (230, 230, 230))
                ImageDraw.Draw(tile).text((12, 80), "missing frame", fill=(120, 0, 0))
            canvas.paste(tile, (x, y + label_h))
            draw.text((x, y + 18), f"{frame['name']} step={frame['step']}", fill=(70, 70, 70))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_markdown(run_dir: Path, cases: list[dict[str, Any]], gallery_path: Path, out_path: Path) -> None:
    lines = [
        "# AirSim 30 条 Smoke Eval 代表案例",
        "",
        "用途：给导师 review 时快速展示本机 4bit AirSim 单场景评估的成功与失败模式。",
        "",
        f"- Run dir: `{run_dir.relative_to(ROOT)}`",
        f"- Gallery: `{gallery_path.relative_to(ROOT)}`",
        "- 说明：这是单场景 smoke eval，不等同于完整 OpenFly benchmark。",
        "",
        "## 案例表",
        "",
        "| case | sample | class | steps | GT len | NE | min NE | SR | OSR | SPL | actions | 解释 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for case in cases:
        actions = case["actions"]
        action_text = str(actions[:12]) + ("..." if len(actions) > 12 else "")
        row = dict(case)
        row["action_text"] = action_text
        lines.append(
            "| {label} | {sample_index} | {analysis_class} | {steps_run} | {gt_len} | "
            "{ne:.3f} | {min_ne:.3f} | {sr} | {osr} | {spl:.4f} | `{action_text}` | {note} |".format(
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            "- 成功样例证明 4bit 模型、AirSim 图像获取、动作执行、NE/SR/OSR/SPL 记录链路是可运行的。",
            "- near-miss/overshoot 样例说明模型有时进入成功半径，但 stop 时机不稳定。",
            "- immediate stop 样例说明部分场景会过早输出 stop，需要检查 prompt、历史帧和 action decoding 是否与官方流程完全一致。",
            "- max-steps forward-bias 样例说明 action 9 占比偏高，后续要和官方 `train/eval.py` 的 action mapping 与 frame history 构造对齐。",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_dir = DEFAULT_RUN_DIR
    results = load_json(run_dir / "results.json")["results"]
    analysis = load_json(run_dir / "analysis.json")
    cases = pick_cases(results, analysis["samples"])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gallery_path = OUT_ROOT / "case_gallery.png"
    make_gallery(run_dir, cases, gallery_path)
    write_markdown(run_dir, cases, gallery_path, OUT_ROOT / "README.md")

    shutil.copy2(run_dir / "analysis.md", OUT_ROOT / "source_analysis.md")
    print(f"cases: {OUT_ROOT / 'cases.json'}")
    print(f"gallery: {gallery_path}")
    print(f"markdown: {OUT_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
