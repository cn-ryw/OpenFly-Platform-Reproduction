import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError


ROOT = Path(__file__).resolve().parent
REPO_ID = "IPEC-COMMUNITY/OpenFly_DataGen"
DEFAULT_ENV = "env_airsim_16"
DEFAULT_OUT_DIR = ROOT / "envs" / "airsim"
MANIFEST_ROOT = ROOT / "demo_results" / "downloads"


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


def hf_cli_path() -> Path:
    standalone = Path.home() / ".local" / "bin" / "hf"
    if standalone.exists():
        return standalone
    conda_hf = Path(sys.executable).with_name("hf")
    if conda_hf.exists():
        return conda_hf
    return Path("hf")


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for member in zf.infolist():
        target = Path(member.filename)
        if target.is_absolute() or ".." in target.parts:
            raise ValueError(f"Unsafe zip member path: {member.filename}")
        members.append(member)
    return members


def inspect_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        members = safe_members(zf)
        roots = sorted({Path(member.filename).parts[0] for member in members if Path(member.filename).parts})
        total_uncompressed = sum(member.file_size for member in members)
        preview = [member.filename for member in members[:20]]
    return {
        "zip_path": str(path),
        "zip_bytes": path.stat().st_size,
        "zip_sha256": sha256_file(path),
        "roots": roots,
        "member_count": len(members),
        "total_uncompressed_bytes": total_uncompressed,
        "preview": preview,
    }


def extract_zip(path: Path, out_dir: Path, env_name: str, force: bool) -> dict[str, Any]:
    target = out_dir / env_name
    if target.exists() and not force:
        return {
            "skipped": True,
            "reason": f"{relative_path(target)} already exists; use --force-extract to overwrite.",
            "target": relative_path(target),
        }
    if target.exists() and force:
        shutil.rmtree(target)

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        members = safe_members(zf)
        zf.extractall(out_dir, members)

    start_script = target / "LinuxNoEditor" / "start.sh"
    return {
        "skipped": False,
        "target": relative_path(target),
        "target_exists": target.exists(),
        "start_script": relative_path(start_script),
        "start_script_exists": start_script.exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one OpenFly AirSim scene zip and extract it.")
    parser.add_argument("--env-name", default=DEFAULT_ENV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--zip-path", type=Path, help="Use an already downloaded local env zip instead of Hub download.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    if args.zip_path and not args.zip_path.is_absolute():
        args.zip_path = ROOT / args.zip_path

    filename = f"airsim/{args.env_name}.zip"
    manifest_dir = MANIFEST_ROOT / args.env_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    whoami = run_command([str(hf_cli_path()), "auth", "whoami"])
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "filename": filename,
        "env_name": args.env_name,
        "out_dir": relative_path(args.out_dir),
        "zip_path": str(args.zip_path) if args.zip_path else None,
        "local_files_only": args.local_files_only,
        "whoami": whoami,
    }

    try:
        if args.zip_path:
            zip_path = args.zip_path
            if not zip_path.exists():
                raise FileNotFoundError(f"Local zip does not exist: {zip_path}")
            payload["download"] = {"ok": True, "source": "local_zip", "path": str(zip_path)}
        else:
            zip_path = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    filename=filename,
                    revision=args.revision,
                    local_files_only=args.local_files_only,
                )
            )
            payload["download"] = {"ok": True, "source": "huggingface_hub", "path": str(zip_path)}
        payload["zip"] = inspect_zip(zip_path)
        if args.no_extract:
            payload["extract"] = {"skipped": True, "reason": "--no-extract"}
        else:
            payload["extract"] = extract_zip(zip_path, args.out_dir, args.env_name, args.force_extract)
    except (GatedRepoError, HfHubHTTPError) as exc:
        payload["download"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "next_step": (
                "Log in with `hf auth login` or set HF_TOKEN, and make sure the account has "
                "access to the gated dataset IPEC-COMMUNITY/OpenFly_DataGen."
            ),
        }
    except Exception as exc:
        payload["download"] = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}

    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"manifest: {manifest_path}")

    if not payload.get("download", {}).get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
