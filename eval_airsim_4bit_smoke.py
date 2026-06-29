import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
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
DEFAULT_ENV = "env_airsim_16"
DEFAULT_EVAL_JSON = ROOT / "configs" / "eval_test.json"
DEFAULT_OUT_ROOT = ROOT / "demo_results" / "airsim_smoke"
PROJECT_AIRSIM_SETTINGS = ROOT / "envs" / "airsim" / "AirSim" / "settings.json"
USER_AIRSIM_SETTINGS = Path.home() / "Documents" / "AirSim" / "settings.json"
SUCCESS_RADIUS = 20.0
ACTION_TEMPLATES = {
    0: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    2: np.array([0, 0, 15, 0, 0, 0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 0, 15, 0, 0, 0, 0], dtype=np.float32),
    4: np.array([0, 0, 0, 0, 2, 0, 0, 0], dtype=np.float32),
    5: np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.float32),
    6: np.array([0, 0, 0, 0, 0, 0, 5, 0], dtype=np.float32),
    7: np.array([0, 0, 0, 0, 0, 0, 0, 5], dtype=np.float32),
    8: np.array([0, 6, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    9: np.array([0, 9, 0, 0, 0, 0, 0, 0], dtype=np.float32),
}


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


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def package_status(module_name: str) -> dict[str, Any]:
    try:
        __import__(module_name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True}


def scene_status(env_name: str) -> dict[str, Any]:
    env_dir = ROOT / "envs" / "airsim" / env_name
    start_script = env_dir / "LinuxNoEditor" / "start.sh"
    return {
        "env_dir": relative_path(env_dir),
        "env_dir_exists": env_dir.exists(),
        "start_script": relative_path(start_script),
        "start_script_exists": start_script.exists(),
    }


def airsim_settings_status() -> dict[str, Any]:
    project_hash = sha256_file(PROJECT_AIRSIM_SETTINGS)
    user_hash = sha256_file(USER_AIRSIM_SETTINGS)
    return {
        "project_settings": relative_path(PROJECT_AIRSIM_SETTINGS),
        "project_exists": PROJECT_AIRSIM_SETTINGS.exists(),
        "project_sha256": project_hash,
        "user_settings": str(USER_AIRSIM_SETTINGS),
        "user_exists": USER_AIRSIM_SETTINGS.exists(),
        "user_sha256": user_hash,
        "matches_project": bool(project_hash and user_hash and project_hash == user_hash),
    }


def prepare_airsim_settings(run_dir: Path) -> dict[str, Any]:
    status_before = airsim_settings_status()
    if not PROJECT_AIRSIM_SETTINGS.exists():
        raise FileNotFoundError(f"Missing project AirSim settings: {PROJECT_AIRSIM_SETTINGS}")
    USER_AIRSIM_SETTINGS.parent.mkdir(parents=True, exist_ok=True)

    backup_path = None
    if USER_AIRSIM_SETTINGS.exists() and status_before["user_sha256"] != status_before["project_sha256"]:
        backup_path = USER_AIRSIM_SETTINGS.with_name(f"settings.json.openfly_backup_{now_timestamp()}")
        shutil.copy2(USER_AIRSIM_SETTINGS, backup_path)
        shutil.copy2(USER_AIRSIM_SETTINGS, run_dir / backup_path.name)

    shutil.copy2(PROJECT_AIRSIM_SETTINGS, USER_AIRSIM_SETTINGS)
    status_after = airsim_settings_status()
    return {
        "changed": not status_before["matches_project"],
        "backup_path": str(backup_path) if backup_path else None,
        "backup_copied_to_run_dir": relative_path(run_dir / backup_path.name) if backup_path else None,
        "before": status_before,
        "after": status_after,
    }


def load_eval_items(eval_json: Path, env_name: str, limit: int) -> list[dict[str, Any]]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    items = [item for item in data if item.get("image_path", "").split("/", 1)[0] == env_name]
    return items[:limit]


def register_openfly_classes() -> None:
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


def load_model_and_processor(model_id: str):
    print("[Model] Registering OpenFly classes")
    register_openfly_classes()
    print("[Model] Loading processor")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("[Model] Loading 4bit model")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print_gpu("[Model loaded]")
    return model, processor


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * cr + cy * sp * sr
    z = sy * cp * sr - cy * sp * cr
    return [x, y, z, w]


class AirsimSmokeBridge:
    def __init__(
        self,
        env_name: str,
        startup_wait_sec: float,
        connection_timeout_sec: float,
        vehicle_name: str,
        run_dir: Path,
    ):
        import airsim

        self.airsim = airsim
        self.env_name = env_name
        self.vehicle_name = vehicle_name
        self.run_dir = run_dir
        self.process: subprocess.Popen | None = None
        self._stdout_handle = None
        self._stderr_handle = None
        try:
            self._start_env(startup_wait_sec)
            self.client = airsim.MultirotorClient()
            self._wait_for_connection(connection_timeout_sec)
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        except Exception:
            self.close()
            raise

    def _start_env(self, startup_wait_sec: float) -> None:
        env_dir = ROOT / "envs" / "airsim" / self.env_name
        start_script = env_dir / "LinuxNoEditor" / "start.sh"
        if not start_script.exists():
            raise FileNotFoundError(f"Missing AirSim start script: {start_script}")

        stdout_path = self.run_dir / "airsim_stdout.log"
        stderr_path = self.run_dir / "airsim_stderr.log"
        self._stdout_handle = stdout_path.open("w", encoding="utf-8")
        self._stderr_handle = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            ["bash", str(start_script)],
            cwd=env_dir,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            preexec_fn=os.setsid,
        )
        print(f"[AirSim] Started {self.env_name}; waiting {startup_wait_sec:.1f}s")
        time.sleep(startup_wait_sec)

    def _wait_for_connection(self, timeout_sec: float) -> None:
        started = time.perf_counter()
        last_error = None
        while time.perf_counter() - started < timeout_sec:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"AirSim process exited early with code {self.process.returncode}")
            try:
                if self.client.ping():
                    print("[AirSim] Connected")
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(1.0)
        detail = f"; last error: {last_error}" if last_error else ""
        raise TimeoutError(f"Timed out waiting for AirSim RPC connection after {timeout_sec:.1f}s{detail}")

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            print("[AirSim] Terminating spawned process group")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        for handle in [self._stdout_handle, self._stderr_handle]:
            if handle and not handle.closed:
                handle.close()

    def set_camera_pose(self, x: float, y: float, z: float, pitch: float, yaw: float, roll: float) -> None:
        pose = self.airsim.Pose(
            self.airsim.Vector3r(x, -y, -z),
            self.airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)),
        )
        self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02, vehicle_name=self.vehicle_name)
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.vehicle_name)

    def set_drone_pos(self, x: float, y: float, z: float, pitch: float, yaw: float, roll: float) -> None:
        qua = euler_to_quaternion(pitch, -yaw, roll)
        pose = self.airsim.Pose(
            self.airsim.Vector3r(x, y, z),
            self.airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]),
        )
        self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02, vehicle_name=self.vehicle_name)
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.vehicle_name)
        self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02, vehicle_name=self.vehicle_name)

    def get_camera_data(self) -> np.ndarray:
        responses = self.client.simGetImages(
            [self.airsim.ImageRequest("front_custom", self.airsim.ImageType.Scene, False, False)],
            vehicle_name=self.vehicle_name,
        )
        if not responses:
            raise RuntimeError("AirSim returned no image response")
        response = responses[0]
        if response.height <= 0 or response.width <= 0:
            raise RuntimeError(f"Invalid AirSim image size: {response.width}x{response.height}")
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        return img.reshape(response.height, response.width, 3)


def history_images(image_list: list[np.ndarray]) -> list[Image.Image]:
    if len(image_list) >= 3:
        selected = image_list[-3:]
    elif len(image_list) == 2:
        selected = [image_list[0], image_list[0], image_list[1]]
    elif len(image_list) == 1:
        selected = [image_list[0], image_list[0], image_list[0]]
    else:
        raise ValueError("image_list must contain at least one frame")
    return [Image.fromarray(img).convert("RGB") for img in selected]


def convert_to_action_id(action: np.ndarray) -> int:
    for action_id, template in ACTION_TEMPLATES.items():
        if np.array_equal(action, template.astype(action.dtype)):
            return action_id
    return 0


def nearest_action_template(action: np.ndarray) -> dict[str, Any]:
    distances = {
        str(action_id): float(np.linalg.norm(action - template.astype(float)))
        for action_id, template in ACTION_TEMPLATES.items()
    }
    best_id = min(distances, key=distances.get)
    return {"action_id": int(best_id), "l2_distance": distances[best_id], "all_l2": distances}


def get_pose_after_action(pose: list[float], action_id: int) -> list[float]:
    x, y, z, yaw = pose
    step_size = 3.0
    if action_id == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action_id == 2:
        yaw += math.radians(30)
    elif action_id == 3:
        yaw -= math.radians(30)
    elif action_id == 4:
        z += step_size
    elif action_id == 5:
        z -= step_size
    elif action_id == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action_id == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action_id == 8:
        x += step_size * math.cos(yaw) * 2
        y += step_size * math.sin(yaw) * 2
    elif action_id == 9:
        x += step_size * math.cos(yaw) * 3
        y += step_size * math.sin(yaw) * 3
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return [x, y, z, yaw]


def distance(point1: list[float], point2: list[float]) -> float:
    return math.sqrt(
        (point2[0] - point1[0]) ** 2
        + (point2[1] - point1[1]) ** 2
        + (point2[2] - point1[2]) ** 2
    )


def predict_action(model, processor, prompt: str, image_list: list[np.ndarray], unnorm_key: str) -> dict[str, Any]:
    images = history_images(image_list)
    input_device = next(model.parameters()).device
    inputs = processor(prompt, images).to(input_device, dtype=torch.float16)
    with torch.inference_mode():
        raw = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    raw_arr = np.array(raw, dtype=float)
    rounded = np.rint(raw_arr).astype(int)
    action_id = convert_to_action_id(rounded)
    return {
        "raw_action": raw_arr.tolist(),
        "rounded_action": rounded.tolist(),
        "action_id": action_id,
        "nearest_template": nearest_action_template(raw_arr),
    }


def save_frame(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image).convert("RGB").save(path)


def write_contact_sheet(run_dir: Path, results: list[dict[str, Any]]) -> Path | None:
    rows = []
    for result in results:
        sample_dir = run_dir / f"sample_{result['sample_index']:02d}"
        frames = sorted(sample_dir.glob("step_*.png"))
        if not frames:
            continue
        rows.append((result, [frames[0], frames[len(frames) // 2], frames[-1]]))
    if not rows:
        return None

    thumb_w, thumb_h = 320, 180
    label_h = 40
    margin = 12
    cols = 3
    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * margin, len(rows) * (thumb_h + label_h) + (len(rows) + 1) * margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_idx, (result, frame_paths) in enumerate(rows):
        for col_idx, path in enumerate(frame_paths):
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            x = margin + col_idx * (thumb_w + margin)
            y = margin + row_idx * (thumb_h + label_h + margin)
            sheet.paste(img, (x, y))
            label = f"sample {result['sample_index']} {path.stem} SR={result['success']} OSR={result['osr']}"
            draw.text((x, y + thumb_h + 4), label, fill=(0, 0, 0))

    out_path = run_dir / "contact_sheet.png"
    sheet.save(out_path)
    return out_path


def run_sample(
    sample: dict[str, Any],
    sample_index: int,
    bridge: AirsimSmokeBridge,
    model,
    processor,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    prompt = sample["gpt_instruction"]
    pos_list = sample["pos"]
    start_position = pos_list[0]
    end_position = pos_list[-1]
    start_yaw = sample["yaw"][0]
    new_pose = [start_position[0], start_position[1], start_position[2], start_yaw]
    old_pose = list(new_pose)
    pass_len = 1e-3
    osr = 0
    action_steps = []
    image_list: list[np.ndarray] = []
    image_error = None
    pitch = -45.0 if "high" in sample["image_path"] else 0.0
    sample_dir = run_dir / f"sample_{sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    bridge.set_camera_pose(start_position[0], start_position[1], start_position[2], pitch, np.rad2deg(start_yaw), 0)
    time.sleep(args.step_sleep_sec)

    for step in range(args.max_steps):
        try:
            frame = bridge.get_camera_data()
            image_list.append(frame)
            if args.save_frames:
                save_frame(frame, sample_dir / f"step_{step:03d}.png")

            pred = predict_action(model, processor, prompt, image_list, args.unnorm_key)
            action_id = pred["action_id"]
            new_pose = get_pose_after_action(new_pose, action_id)
            bridge.set_camera_pose(new_pose[0], new_pose[1], new_pose[2], pitch, np.rad2deg(new_pose[3]), 0)
            pass_len += distance(old_pose, new_pose)
            dis = distance(end_position, new_pose)
            if dis < SUCCESS_RADIUS:
                osr = 1
            old_pose = list(new_pose)

            step_record = {
                "step": step,
                **pred,
                "pose_after_action": new_pose,
                "distance_to_goal": dis,
                "gpu": gpu_snapshot(),
            }
            action_steps.append(step_record)
            print(
                f"[Sample {sample_index} step {step}] action_id={action_id} "
                f"dist={dis:.3f} raw={pred['raw_action']}"
            )
            if action_id == 0:
                break
            time.sleep(args.step_sleep_sec)
        except Exception as exc:
            image_error = f"{type(exc).__name__}: {exc}"
            print(f"[Sample {sample_index}] error at step {step}: {image_error}")
            break

    final_distance = distance(end_position, new_pose)
    traj_len = distance(start_position, end_position)
    success = final_distance < SUCCESS_RADIUS
    spl = traj_len / pass_len if success else 0.0
    return {
        "sample_index": sample_index,
        "image_path": sample["image_path"],
        "prompt": prompt,
        "ground_truth_action": sample.get("action"),
        "ground_truth_action_len": len(sample.get("action", [])),
        "start_position": start_position,
        "end_position": end_position,
        "start_yaw": start_yaw,
        "pitch": pitch,
        "max_steps": args.max_steps,
        "steps_run": len(action_steps),
        "action_steps": action_steps,
        "predicted_action_ids": [step["action_id"] for step in action_steps],
        "final_pose": new_pose,
        "traj_len": traj_len,
        "pass_len": pass_len,
        "distance_to_goal": final_distance,
        "success": int(success),
        "osr": osr,
        "spl": spl,
        "image_error": image_error,
    }


def write_pip_freeze(path: Path) -> None:
    freeze = run_command([sys.executable, "-m", "pip", "freeze"])
    text = freeze["stdout"]
    if freeze["stderr"]:
        text += "\n# stderr:\n" + freeze["stderr"]
    path.write_text(text + "\n", encoding="utf-8")


def warning_summary(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    counts = {
        "FutureWarning": text.count("FutureWarning"),
        "UserWarning": text.count("UserWarning"),
        "deprecated": text.lower().count("deprecated"),
        "Traceback": text.count("Traceback"),
        "Exception": text.count("Exception"),
        "out_of_memory": text.lower().count("out of memory"),
        "OOM": text.count("OOM"),
    }
    fatal = counts["Traceback"] + counts["Exception"] + counts["out_of_memory"] + counts["OOM"]
    return {"counts": counts, "non_fatal_warnings_only": fatal == 0}


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    checks = payload["preflight"]
    results = payload.get("results", [])
    lines = [
        "# AirSim 4bit Smoke Eval",
        "",
        "## Scope",
        "",
        "This is a minimal AirSim-first smoke evaluation for OpenFly-Agent on local 4bit inference. It is not a full OpenFly benchmark run.",
        "",
        "## Preflight",
        "",
        f"- Environment: `{payload['env_name']}`",
        f"- Scene start script exists: `{checks['scene']['start_script_exists']}`",
        f"- Python airsim import: `{checks['packages']['airsim']['ok']}`",
        f"- CUDA available: `{torch.cuda.is_available()}`",
        f"- Samples selected: `{len(payload['selected_samples'])}`",
        f"- AirSim settings matches project: `{checks['airsim_settings']['matches_project']}`",
        f"- Dry run: `{payload['dry_run']}`",
        "",
        "## Settings",
        "",
        f"- Model: `{payload['model']}`",
        f"- Unnorm key: `{payload['unnorm_key']}`",
        f"- Max steps per sample: `{payload['max_steps']}`",
        f"- Success radius: `{SUCCESS_RADIUS}`",
        f"- Vehicle name: `{payload['vehicle_name']}`",
        "",
    ]
    if results:
        lines.extend(
            [
                "## Results",
                "",
                "| sample | steps | actions | NE | SR | OSR | SPL | image_error |",
                "|---:|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in results:
            lines.append(
                f"| {item['sample_index']} | {item['steps_run']} | `{item['predicted_action_ids']}` | "
                f"{item['distance_to_goal']:.3f} | {item['success']} | {item['osr']} | "
                f"{item['spl']:.4f} | `{item['image_error']}` |"
            )
        mean_ne = sum(item["distance_to_goal"] for item in results) / len(results)
        mean_sr = sum(item["success"] for item in results) / len(results)
        mean_osr = sum(item["osr"] for item in results) / len(results)
        mean_spl = sum(item["spl"] for item in results) / len(results)
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- Mean NE: `{mean_ne:.3f}`",
                f"- Mean SR: `{mean_sr:.4f}`",
                f"- Mean OSR: `{mean_osr:.4f}`",
                f"- Mean SPL: `{mean_spl:.4f}`",
            ]
        )
    else:
        lines.extend(
            [
                "## Dry-run result",
                "",
                "No simulator or model execution was run. Resolve failed preflight items before actual smoke evaluation.",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Official `train/eval.py` uses `unnorm_key=\"vlnv1\"`; this script follows that by default.",
            "- The action-id conversion intentionally mirrors official eval: unmatched rounded 8D actions default to action id 0.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_preflight(args: argparse.Namespace, selected_samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scene": scene_status(args.env_name),
        "packages": {
            "airsim": package_status("airsim"),
            "msgpackrpc": package_status("msgpackrpc"),
        },
        "airsim_settings": airsim_settings_status(),
        "nvidia_smi": get_nvidia_smi_snapshot(),
        "selected_samples": [
            {
                "sample_index": idx,
                "image_path": item.get("image_path"),
                "prompt": item.get("gpt_instruction"),
                "action_len": len(item.get("action", [])),
            }
            for idx, item in enumerate(selected_samples)
        ],
    }


def dry_run_payload(args: argparse.Namespace, selected_samples: list[dict[str, Any]]) -> dict[str, Any]:
    preflight = build_preflight(args, selected_samples)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": True,
        "repo_root": str(ROOT),
        "model": args.model_id,
        "env_name": args.env_name,
        "eval_json": relative_path(args.eval_json),
        "unnorm_key": args.unnorm_key,
        "max_steps": args.max_steps,
        "selected_samples": preflight["selected_samples"],
        "preflight": preflight,
        "next_commands": [
            "hf auth login",
            "HF_XET_HIGH_PERFORMANCE=1 python -u download_openfly_airsim_scene.py --env-name env_airsim_16",
            "python -u eval_airsim_4bit_smoke.py --env-name env_airsim_16 --limit 1 --max-steps 20 --prepare-settings",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OpenFly-Agent 4bit AirSim smoke evaluation")
    parser.add_argument("--env-name", default=DEFAULT_ENV)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--startup-wait-sec", type=float, default=20.0)
    parser.add_argument("--connection-timeout-sec", type=float, default=60.0)
    parser.add_argument("--step-sleep-sec", type=float, default=0.1)
    parser.add_argument("--unnorm-key", default="vlnv1")
    parser.add_argument("--vehicle-name", default="drone_1")
    parser.add_argument("--prepare-settings", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.eval_json.is_absolute():
        args.eval_json = ROOT / args.eval_json
    if not args.out_root.is_absolute():
        args.out_root = ROOT / args.out_root

    selected_samples = load_eval_items(args.eval_json, args.env_name, args.limit)
    run_dir = args.out_root / now_timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    pip_freeze_path = run_dir / "pip_freeze.txt"
    summary_path = run_dir / "summary.md"
    results_path = run_dir / "results.json"

    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"[Run dir] {run_dir}")
            write_pip_freeze(pip_freeze_path)
            settings_preparation = prepare_airsim_settings(run_dir) if args.prepare_settings else None
            preflight = build_preflight(args, selected_samples)
            print(json.dumps(preflight, ensure_ascii=False, indent=2))

            if args.dry_run:
                payload = dry_run_payload(args, selected_samples)
                payload["vehicle_name"] = args.vehicle_name
                payload["airsim_settings_preparation"] = settings_preparation
                payload["pip_freeze_path"] = relative_path(pip_freeze_path)
                payload["run_command"] = " ".join([sys.executable, *sys.argv])
                payload["warning_summary"] = warning_summary(log_path)
                results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                write_summary(summary_path, payload)
                print(f"[Dry run done] results: {results_path}")
                print(f"[Dry run done] summary: {summary_path}")
                return

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; this smoke eval requires GPU 4bit inference.")
            if not preflight["scene"]["start_script_exists"]:
                raise FileNotFoundError(f"Missing scene package start script: {preflight['scene']['start_script']}")
            if not preflight["packages"]["airsim"]["ok"]:
                raise ModuleNotFoundError(f"airsim import failed: {preflight['packages']['airsim']['error']}")
            if not preflight["airsim_settings"]["matches_project"]:
                raise RuntimeError("AirSim user settings do not match OpenFly settings; rerun with --prepare-settings.")
            if not selected_samples:
                raise RuntimeError(f"No samples selected for env {args.env_name} in {args.eval_json}")

            model, processor = load_model_and_processor(args.model_id)
            bridge = None
            results = []
            try:
                bridge = AirsimSmokeBridge(
                    args.env_name,
                    args.startup_wait_sec,
                    args.connection_timeout_sec,
                    args.vehicle_name,
                    run_dir,
                )
                for sample_index, sample in enumerate(selected_samples):
                    print(f"[Sample {sample_index}] {sample['image_path']}")
                    results.append(run_sample(sample, sample_index, bridge, model, processor, args, run_dir))
            finally:
                if bridge is not None:
                    bridge.close()

            log_file.flush()
            contact_sheet = write_contact_sheet(run_dir, results) if args.save_frames else None
            payload = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dry_run": False,
                "repo_root": str(ROOT),
                "model": args.model_id,
                "env_name": args.env_name,
                "eval_json": relative_path(args.eval_json),
                "unnorm_key": args.unnorm_key,
                "vehicle_name": args.vehicle_name,
                "max_steps": args.max_steps,
                "selected_samples": preflight["selected_samples"],
                "preflight": preflight,
                "airsim_settings_preparation": settings_preparation,
                "results": results,
                "final_gpu": gpu_snapshot(),
                "nvidia_smi_after": get_nvidia_smi_snapshot(),
                "pip_freeze_path": relative_path(pip_freeze_path),
                "run_command": " ".join([sys.executable, *sys.argv]),
                "warning_summary": warning_summary(log_path),
            }
            if contact_sheet is not None:
                payload["contact_sheet_path"] = relative_path(contact_sheet)
            results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            write_summary(summary_path, payload)
            print(f"[Done] results: {results_path}")
            print(f"[Done] summary: {summary_path}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
