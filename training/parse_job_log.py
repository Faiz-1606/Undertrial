"""
training/parse_job_log.py

Reconstruct training artifacts from a captured Hugging Face Jobs log.

Used when training was run on ephemeral infra (`hf jobs uv run`) and the
on-disk artifacts in `./output/undertrial_grpo/` were destroyed before
being uploaded. The log itself contains every metric we need to rebuild:

    output/undertrial_grpo/curriculum_results.json
    output/undertrial_grpo/plots/reward_curve.png
    output/undertrial_grpo/plots/training_loss.png
    output/undertrial_grpo/plots/before_after_comparison.png

Capture the log with:
    hf jobs logs <job_id> > training_log.txt

Then run:
    python training/parse_job_log.py training_log.txt
    python training/parse_job_log.py training_log.txt --output ./output/undertrial_grpo
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ── Regex patterns ──────────────────────────────────────────────────────────
# Markers in `train_grpo.py` we rely on (all printed unconditionally).
STAGE_MARKER_RE = re.compile(r"Evaluating baseline on Stage\s+(\d+)")
STAGE_BASELINE_RE = re.compile(r"Stage\s+(\d+)\s+baseline:\s+(-?\d+\.\d+)")

# `hf jobs logs <id>` re-emits the full log from job start on every call. If
# the user concatenated several captures into one file, only the LAST segment
# (after the final "===== Job started at ..." banner) is the complete run.
JOB_START_BANNER_RE = re.compile(r"^=+\s*Job started at .+=+\s*$", re.MULTILINE)

# `Stage N: 0.4786 → 0.5314 (Δ = +0.0528)` — but we tolerate any non-digit
# placeholders for `→` and `Δ` so we survive Windows codepage mangling when
# the log was captured via PowerShell `>` redirection on UTF-16 boundaries.
STAGE_DELTA_RE = re.compile(
    r"Stage\s+(\d+):\s+(-?\d+\.\d+)\s+\S+\s+(-?\d+\.\d+)\s+\([^=]+=\s*([+-]?\d+\.\d+)\s*\)"
)
TRACES_RE = re.compile(r"Total traces harvested:\s+(\d+)")


# ── IO helpers ──────────────────────────────────────────────────────────────
def read_log(path: Path) -> str:
    """Read a log file, autodetecting common Windows-redirection encodings."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


# ── Parser ──────────────────────────────────────────────────────────────────
def _trim_to_last_job_segment(text: str) -> str:
    """If the log contains multiple concatenated `hf jobs logs` captures, keep
    only the final, most-complete segment (everything after the last job-start
    banner). Returns the original text unchanged if no banner is found."""
    matches = list(JOB_START_BANNER_RE.finditer(text))
    if len(matches) <= 1:
        return text
    last = matches[-1]
    return text[last.start():]


def parse_log(log_path: Path) -> Dict[str, Any]:
    text = read_log(log_path)
    text = _trim_to_last_job_segment(text)

    stage_results: Dict[int, Dict[str, float]] = {}
    log_entries: List[Dict[str, Any]] = []

    current_stage: Optional[int] = None
    cumstep_offset = 0
    last_step_in_stage = 0

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        m = STAGE_MARKER_RE.search(line)
        if m:
            new_stage = int(m.group(1))
            if current_stage is not None and new_stage != current_stage:
                cumstep_offset += last_step_in_stage
                last_step_in_stage = 0
            current_stage = new_stage
            stage_results.setdefault(current_stage, {})
            continue

        m = STAGE_BASELINE_RE.search(line)
        if m:
            s = int(m.group(1))
            stage_results.setdefault(s, {})["baseline"] = float(m.group(2))
            continue

        m = STAGE_DELTA_RE.search(line)
        if m:
            s = int(m.group(1))
            stage_results.setdefault(s, {})
            stage_results[s].setdefault("baseline", float(m.group(2)))
            stage_results[s]["post"] = float(m.group(3))
            stage_results[s]["delta"] = float(m.group(4))
            continue

        if "{" in line and "}" in line and ("'loss'" in line or "'reward'" in line):
            start = line.index("{")
            end = line.rindex("}") + 1
            try:
                d = ast.literal_eval(line[start:end])
            except (ValueError, SyntaxError):
                continue
            if not isinstance(d, dict):
                continue
            if current_stage is not None:
                d["_stage"] = current_stage
            local_step = None
            for key in ("step", "global_step"):
                if key in d:
                    try:
                        local_step = int(d[key])
                        break
                    except (TypeError, ValueError):
                        pass
            if local_step is None:
                local_step = last_step_in_stage + 5
            d["_local_step"] = local_step
            d["_cumstep"] = cumstep_offset + local_step
            last_step_in_stage = max(last_step_in_stage, local_step)
            log_entries.append(d)
            continue

    m = TRACES_RE.search(text)
    traces_harvested = int(m.group(1)) if m else 0

    return {
        "stage_results": stage_results,
        "log_entries": log_entries,
        "traces_harvested": traces_harvested,
    }


# ── Plotting ────────────────────────────────────────────────────────────────
STAGE_COLORS = {1: "#2563eb", 2: "#16a34a", 3: "#ca8a04", 4: "#dc2626"}


def _series_for_stage(entries: List[Dict[str, Any]], stage: int, key: str):
    xs, ys = [], []
    for e in entries:
        if e.get("_stage") != stage:
            continue
        if key not in e:
            continue
        xs.append(e["_cumstep"])
        ys.append(float(e[key]))
    return xs, ys


def save_reward_curve(
    log_entries: List[Dict[str, Any]],
    stage_results: Dict[int, Dict[str, float]],
    out_path: Path,
) -> None:
    if not log_entries:
        print(f"[parse] WARN: no metric entries; skipping {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    stages_present = sorted({e["_stage"] for e in log_entries if "_stage" in e})
    for stage in stages_present:
        xs, ys = _series_for_stage(log_entries, stage, "reward")
        if not xs:
            continue
        color = STAGE_COLORS.get(stage, None)
        ax.plot(xs, ys, "o-", label=f"Stage {stage} (rollout)", color=color, markersize=5, linewidth=1.8)
        if stage in stage_results and "baseline" in stage_results[stage]:
            ax.hlines(
                stage_results[stage]["baseline"],
                xmin=min(xs),
                xmax=max(xs),
                colors=color,
                linestyles="--",
                alpha=0.45,
            )
        if stage in stage_results and "post" in stage_results[stage]:
            ax.hlines(
                stage_results[stage]["post"],
                xmin=min(xs),
                xmax=max(xs),
                colors=color,
                linestyles=":",
                alpha=0.7,
            )

    boundary = 0
    for stage in stages_present[:-1]:
        xs, _ = _series_for_stage(log_entries, stage, "reward")
        if xs:
            boundary = max(xs)
            ax.axvline(boundary, color="gray", linestyle=":", alpha=0.35)

    ax.set_xlabel("Cumulative training step")
    ax.set_ylabel("Mean reward")
    ax.set_title("UndertriAI GRPO — reward curve across curriculum stages")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[parse] wrote {out_path}")


def save_loss_curve(log_entries: List[Dict[str, Any]], out_path: Path) -> None:
    xs = [e["_cumstep"] for e in log_entries if "loss" in e]
    ys = [float(e["loss"]) for e in log_entries if "loss" in e]
    if not xs:
        print(f"[parse] WARN: no loss entries; skipping {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xs, ys, "o-", color="#7c3aed", markersize=4, linewidth=1.6)
    ax.set_xlabel("Cumulative training step")
    ax.set_ylabel("Training loss")
    ax.set_title("UndertriAI GRPO — training loss")
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[parse] wrote {out_path}")


def save_comparison_plot(
    stage_results: Dict[int, Dict[str, float]], out_path: Path
) -> None:
    stages = sorted(stage_results.keys())
    if not stages:
        print(f"[parse] WARN: no stage results; skipping {out_path.name}")
        return

    baselines = [stage_results[s].get("baseline", 0.0) for s in stages]
    posts = [
        stage_results[s].get("post", stage_results[s].get("baseline", 0.0))
        for s in stages
    ]

    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baselines, width, label="Before (baseline)", color="#94a3b8")
    bars2 = ax.bar(x + width / 2, posts, width, label="After (trained)", color="#3b82f6")

    for b, v in zip(bars1, baselines):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for b, v in zip(bars2, posts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Stage {s}" for s in stages])
    ax.set_ylabel("Mean reward")
    ax.set_title("UndertriAI — baseline vs trained reward per curriculum stage")
    ax.set_ylim(0, max(max(baselines), max(posts)) * 1.18 + 0.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[parse] wrote {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct training artifacts from an HF Jobs log."
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to captured HF Jobs log (e.g. training_log.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/undertrial_grpo"),
        help="Output directory for reconstructed artifacts (default: ./output/undertrial_grpo)",
    )
    parser.add_argument(
        "--model",
        default="Qwen2.5-1.5B-Instruct",
        help="Model name to record in the JSON metadata (display only).",
    )
    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"[parse] ERROR: {args.log_path} not found", file=sys.stderr)
        return 1

    parsed = parse_log(args.log_path)
    stage_results = parsed["stage_results"]
    log_entries = parsed["log_entries"]
    traces = parsed["traces_harvested"]

    output_dir: Path = args.output
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_keys: List[str] = sorted(
        {k for e in log_entries for k in e if not k.startswith("_")}
    )
    serialisable_entries = [
        {k: v for k, v in e.items() if not k.startswith("_")} | {
            "_stage": e.get("_stage"),
            "_cumstep": e.get("_cumstep"),
        }
        for e in log_entries
    ]

    results = {
        "model": args.model,
        "mode": "curriculum",
        "source": "reconstructed_from_log",
        "log_path": str(args.log_path),
        "stage_results": {str(k): v for k, v in stage_results.items()},
        "traces_harvested": traces,
        "n_metric_entries": len(log_entries),
        "metric_keys_seen": metric_keys,
        "log_history": serialisable_entries,
    }

    json_path = output_dir / "curriculum_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[parse] wrote {json_path}")

    save_reward_curve(log_entries, stage_results, plots_dir / "reward_curve.png")
    save_loss_curve(log_entries, plots_dir / "training_loss.png")
    save_comparison_plot(stage_results, plots_dir / "before_after_comparison.png")

    print()
    print("=" * 60)
    print("Headline metrics (reconstructed from log)")
    print("=" * 60)
    for s, res in sorted(stage_results.items()):
        b = res.get("baseline", 0.0)
        p = res.get("post", b)
        d = res.get("delta", p - b)
        print(f"  Stage {s}: {b:.4f} -> {p:.4f}  (delta = {d:+.4f})")
    if stage_results:
        bs = [r.get("baseline", 0.0) for r in stage_results.values()]
        ps = [r.get("post", r.get("baseline", 0.0)) for r in stage_results.values()]
        mean_b = sum(bs) / len(bs)
        mean_p = sum(ps) / len(ps)
        print(
            f"  Mean:    {mean_b:.4f} -> {mean_p:.4f}  (delta = {mean_p - mean_b:+.4f})"
        )
    print(f"  Traces harvested: {traces}")
    print(f"  Metric entries parsed: {len(log_entries)}")
    if metric_keys:
        print(f"  Metric keys seen: {', '.join(metric_keys)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
