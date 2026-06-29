import json
import sys
from pathlib import Path

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
IMAGE_PATHS = [
    ROOT / "demo_imgs" / "hist_1.png",
    ROOT / "demo_imgs" / "hist_2.png",
    ROOT / "demo_imgs" / "current.png",
]
PROMPT = "Fly forward carefully, avoid obstacles, and move towards the open area."
OUT_PATH = ROOT / "demo_results" / "openfly_own_video_result.json"


def print_gpu(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: CUDA unavailable")
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"{label}: allocated={allocated:.1f}MB "
        f"reserved={reserved:.1f}MB max_allocated={max_allocated:.1f}MB"
    )


def load_images() -> list[Image.Image]:
    missing = [str(path) for path in IMAGE_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing demo image(s): {missing}")

    return [Image.open(path).convert("RGB") for path in IMAGE_PATHS]


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; 4bit GPU inference cannot run.")

    print("[0/5] Registering OpenFly custom classes")
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

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

    print("[3/5] Loading images")
    images = load_images()

    print("[4/5] Preparing inputs")
    input_device = next(model.parameters()).device
    inputs = processor(PROMPT, images).to(input_device, dtype=torch.float16)
    print_gpu("[After input prepare]")

    print("[5/5] Running inference")
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key="vln_norm",
            do_sample=False,
        )

    action_json = to_jsonable(action)
    print("=" * 80)
    print("Prompt:", PROMPT)
    print("Predicted action:", action_json)
    print("=" * 80)
    print_gpu("[Done]")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "prompt": PROMPT,
                "image_paths": [str(path.relative_to(ROOT)) for path in IMAGE_PATHS],
                "unnorm_key": "vln_norm",
                "predicted_action": action_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
