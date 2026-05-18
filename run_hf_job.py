# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth @ git+https://github.com/unslothai/unsloth.git",
#     "unsloth_zoo",
#     "trl>=0.11.0",
#     "peft",
#     "accelerate",
#     "bitsandbytes",
#     "xformers",
#     "torchvision",
#     "sentencepiece",
#     "protobuf",
#     "einops",
#     "hf_transfer",
#     "datasets",
#     "wandb",
#     "matplotlib",
# ]
# ///
"""Compatibility launcher for the final gated UndertriAI HF Job.

Prefer ``training/run_hf_job.py`` for new commands. This root entrypoint keeps
older ``run_hf_job.py`` commands on the same final-run path.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REPO_OWNER_REPO = "Faiz-1606/Undertrial"
REPO_BRANCH = os.environ.get("UNDERTRIAL_REPO_BRANCH", "main")
WORK_DIR = Path(os.environ.get("UNDERTRIAL_WORK_DIR", "/work"))


def _running_inside_repo() -> Path | None:
    root = Path(__file__).resolve().parent
    if (root / "training" / "train_grpo.py").exists():
        return root
    return None


def _download_repo() -> Path:
    url = (
        f"https://codeload.github.com/{REPO_OWNER_REPO}/tar.gz/refs/heads/"
        f"{REPO_BRANCH}"
    )
    print(f"[bootstrap] downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as staging_str:
        staging = Path(staging_str)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(path=staging)
        top = next(staging.iterdir())
        for entry in top.iterdir():
            shutil.move(str(entry), str(WORK_DIR / entry.name))
    return WORK_DIR


def _default_final_args(work_root: Path) -> list[str]:
    return [
        "--episodes_dir", str(work_root / "data" / "episodes"),
        "--output", "./output/undertrial_grpo",
        "--curriculum",
        "--profile", "prod",
        "--strict_gates",
        "--reward_mode", "offline",
        "--model_name", "unsloth/Qwen2.5-7B-Instruct",
        "--batch_size", "1",
        "--grad_accum", "8",
        "--max_completion_length", "640",
    ]


def main() -> int:
    work_root = _running_inside_repo() or _download_repo()
    train_script = work_root / "training" / "train_grpo.py"
    args = sys.argv[1:] or _default_final_args(work_root)
    if "--episodes_dir" not in args:
        args = ["--episodes_dir", str(work_root / "data" / "episodes"), *args]
    cmd = [sys.executable, str(train_script), *args]
    print(f"[bootstrap] cwd={work_root}", flush=True)
    print(f"[bootstrap] args={args}", flush=True)
    return subprocess.run(cmd, cwd=work_root).returncode


if __name__ == "__main__":
    sys.exit(main())
