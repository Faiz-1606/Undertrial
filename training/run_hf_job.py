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
"""
UndertriAI — Hugging Face Jobs bootstrap launcher.

`hf jobs uv run <script>` only uploads ONE file to the job container, but
`training/train_grpo.py` needs the rest of the repo (server/, models.py,
data/episodes/*.jsonl). This launcher:

    1. Declares every runtime dependency via PEP 723 so `uv run` installs
       them in one resolver pass (avoids the iterative ``--with`` whack-a-mole
       that hits unsloth_zoo → torchvision → sentencepiece → …).
    2. Downloads the repo as a GitHub tarball (no system `git` required).
    3. Invokes ``python training/train_grpo.py`` with the CLI args passed
       through to this script.

Canonical HF Jobs command
─────────────────────────
    hf jobs uv run --flavor a10g-large --timeout 5h --secrets HF_TOKEN \
        https://raw.githubusercontent.com/Faiz-1606/Undertrial/main/training/run_hf_job.py \
        --curriculum \
        --profile prod \
        --strict_gates \
        --reward_mode offline \
        --model_name unsloth/Qwen2.5-7B-Instruct \
        --batch_size 1 --grad_accum 8 \
        --max_completion_length 640 \
        --output ./output/undertrial_grpo \
        --hf_save_repo Draken1606/undertrial-grpo-final

Everything after the script URL is forwarded verbatim to ``train_grpo.py``.

Local use is also supported: if this file is executed from inside a clone of
the repo, it skips the download and runs the sibling ``train_grpo.py``.

Overrides (environment variables)
─────────────────────────────────
    UNDERTRIAL_REPO_URL      default: https://github.com/Faiz-1606/Undertrial
    UNDERTRIAL_REPO_BRANCH   default: main
    UNDERTRIAL_WORK_DIR      default: /work   (where the repo will live)
    UNDERTRIAL_ENV_URL       default: https://draken1606-undertrial-ai.hf.space
                              (forwarded to train_grpo.py if --env_url absent)
    UNDERTRIAL_HF_SAVE_REPO  default: Draken1606/undertrial-grpo-final
                              (forwarded as --hf_save_repo if absent so the
                              trained adapters actually survive the job)
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
from typing import Optional


# ── Config ──────────────────────────────────────────────────────────────────
REPO_OWNER_REPO = "Faiz-1606/Undertrial"
REPO_URL = os.environ.get(
    "UNDERTRIAL_REPO_URL",
    f"https://github.com/{REPO_OWNER_REPO}",
)
REPO_BRANCH = os.environ.get("UNDERTRIAL_REPO_BRANCH", "main")
WORK_DIR = Path(os.environ.get("UNDERTRIAL_WORK_DIR", "/work"))
DEFAULT_ENV_URL = os.environ.get(
    "UNDERTRIAL_ENV_URL",
    "https://draken1606-undertrial-ai.hf.space",
)
DEFAULT_HF_SAVE_REPO = os.environ.get(
    "UNDERTRIAL_HF_SAVE_REPO",
    "Draken1606/undertrial-grpo-final",
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def _running_inside_repo() -> Optional[Path]:
    """
    If this file sits inside a cloned Undertrial repo (e.g. run locally),
    return the repo root so we skip the download step.
    """
    here = Path(__file__).resolve().parent.parent
    train_script = here / "training" / "train_grpo.py"
    episodes_dir = here / "data" / "episodes"
    if train_script.exists() and episodes_dir.exists():
        return here
    return None


def _download_tarball(url: str) -> bytes:
    _log(f"downloading {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "undertrial-run-hf-job/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    _log(f"downloaded {len(data) / 1_000_000:.1f} MB")
    return data


def _extract_tarball_to(tarball_bytes: bytes, dest: Path) -> Path:
    """
    Extract a GitHub tar.gz (top-level dir like ``Undertrial-main/``) into
    ``dest``, flattening the top-level wrapper so the repo contents land
    directly inside ``dest``. Returns ``dest``.
    """
    if dest.exists():
        _log(f"removing stale {dest}")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as staging_str:
        staging = Path(staging_str)
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
            tar.extractall(path=staging)
        top_level = next(staging.iterdir())
        if not top_level.is_dir():
            raise RuntimeError(
                f"Unexpected tarball layout: {top_level} is not a directory"
            )
        for entry in top_level.iterdir():
            shutil.move(str(entry), str(dest / entry.name))

    return dest


def _ensure_repo() -> Path:
    """
    Materialise the Undertrial repo on disk and return its root path.
    """
    local_root = _running_inside_repo()
    if local_root is not None:
        _log(f"reusing local checkout at {local_root}")
        return local_root

    tarball_url = (
        f"https://codeload.github.com/{REPO_OWNER_REPO}/tar.gz/refs/heads/{REPO_BRANCH}"
    )
    tarball_bytes = _download_tarball(tarball_url)
    _extract_tarball_to(tarball_bytes, WORK_DIR)
    _log(f"repo ready at {WORK_DIR}")
    return WORK_DIR


def _forward_args(extra: list[str], work_root: Path) -> list[str]:
    """
    Inject sensible defaults into the CLI args if the user omitted them.
    Does not override any flag the user supplied.
    """
    args = list(extra)
    if not args:
        args = [
            "--curriculum",
            "--profile", "prod",
            "--strict_gates",
            "--reward_mode", "offline",
            "--model_name", "unsloth/Qwen2.5-7B-Instruct",
            "--batch_size", "1",
            "--grad_accum", "8",
            "--max_completion_length", "640",
            "--output", "./output/undertrial_grpo",
        ]
    if "--episodes_dir" not in args:
        args += ["--episodes_dir", str(work_root / "data" / "episodes")]
    reward_mode_offline = (
        "--reward_mode" in args
        and args.index("--reward_mode") + 1 < len(args)
        and args[args.index("--reward_mode") + 1] == "offline"
    )
    if "--env_url" not in args and "--offline" not in args and not reward_mode_offline:
        args += ["--env_url", DEFAULT_ENV_URL]
    # Without --hf_save_repo the trained adapters only live inside the
    # ephemeral HF Job container and disappear at teardown. Auto-inject a
    # default destination if the user didn't pick one.
    if "--hf_save_repo" not in args and DEFAULT_HF_SAVE_REPO:
        args += ["--hf_save_repo", DEFAULT_HF_SAVE_REPO]
    return args


# ── Entry point ─────────────────────────────────────────────────────────────
def main() -> int:
    work_root = _ensure_repo()
    train_script = work_root / "training" / "train_grpo.py"
    if not train_script.exists():
        _log(f"ERROR: {train_script} not found after fetching the repo")
        return 1

    forwarded = _forward_args(sys.argv[1:], work_root)
    cmd = [sys.executable, str(train_script), *forwarded]

    # Child-process environment:
    #   * PYTHONPATH includes the repo root so `from server.reward`,
    #     `from tests.test_anti_hack`, `from training.run_manifest` resolve
    #     (the preflight gates and run-manifest instrumentation depend on
    #     these imports succeeding).
    #   * PYTHONUNBUFFERED=1 so HF Jobs gets line-fresh logs instead of
    #     blocks that arrive minutes later.
    #   * HF_HUB_ENABLE_HF_TRANSFER=1 actually activates the `hf_transfer`
    #     accelerator we already pin in the UV deps header.
    child_env = os.environ.copy()
    existing_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        f"{work_root}{os.pathsep}{existing_pp}" if existing_pp else str(work_root)
    )
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    _log(f"cwd          = {work_root}")
    _log(f"python       = {sys.executable}")
    _log(f"train script = {train_script}")
    _log(f"args         = {forwarded}")
    _log(f"PYTHONPATH   = {child_env['PYTHONPATH']}")
    _log("launching train_grpo.py …")

    proc = subprocess.run(cmd, cwd=work_root, env=child_env)
    _log(f"train_grpo.py exited with code {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
