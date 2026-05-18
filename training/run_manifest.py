"""
UndertriAI — Run Manifest & Instrumentation
Logs git SHA, config snapshot, dataset fingerprint, and per-step metrics
for reproducible training runs.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _dataset_fingerprint(episodes_dir: str) -> str:
    """Hash all episode files for reproducibility tracking."""
    h = hashlib.sha256()
    ep_dir = Path(episodes_dir)
    if not ep_dir.exists():
        return "missing"
    for f in sorted(ep_dir.glob("*.jsonl")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def create_manifest(
    config: Dict[str, Any],
    episodes_dir: str,
    output_dir: str,
    seed: int = 42,
    reward_mode: str = "offline",
) -> Dict[str, Any]:
    """
    Create a run manifest with all configuration needed to reproduce a training run.

    Saved as run_manifest.json in the output directory.
    """
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "seed": seed,
        "reward_mode": reward_mode,
        "dataset_fingerprint": _dataset_fingerprint(episodes_dir),
        "config": config,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  [manifest] Saved to {manifest_path}")
    return manifest


class MetricsTracker:
    """
    Tracks per-step and per-level training metrics for observability.

    Metrics tracked:
    - outcome_accuracy: fraction of completions matching GT outcome direction
    - reward_source: counts of api vs local reward computation
    - template_overlap_rate: detects output collapse (consecutive similarity)
    - reward_std: reward variance (collapse = bad)
    - hack_probe_score: adversarial probe scores at eval checkpoints
    """

    def __init__(self):
        self.reward_source_counts: Dict[str, int] = {"api": 0, "local": 0}
        self.outcome_matches: List[bool] = []
        self.completions_history: List[str] = []
        self.level_metrics: Dict[str, Dict[str, Any]] = {}
        self._current_level: str = ""

    def set_level(self, level: str):
        self._current_level = level
        if level not in self.level_metrics:
            self.level_metrics[level] = {
                "reward_source": {"api": 0, "local": 0},
                "outcome_accuracy": [],
                "template_overlap_rates": [],
            }

    def log_reward_source(self, source: str):
        """Track whether reward came from API or local computation."""
        self.reward_source_counts[source] = self.reward_source_counts.get(source, 0) + 1
        if self._current_level:
            lm = self.level_metrics[self._current_level]["reward_source"]
            lm[source] = lm.get(source, 0) + 1

    def log_outcome(self, predicted: str, ground_truth: str):
        """Track outcome accuracy."""
        pred_granted = "grant" in predicted.lower()
        gt_granted = "grant" in ground_truth.lower()
        match = pred_granted == gt_granted
        self.outcome_matches.append(match)
        if self._current_level:
            self.level_metrics[self._current_level]["outcome_accuracy"].append(match)

    def log_completion(self, completion: str):
        """Track completions for template overlap detection."""
        self.completions_history.append(completion)

    def compute_template_overlap(self, window: int = 8) -> float:
        """
        Compute average word-level Jaccard similarity between recent completions.
        High overlap (>0.7) suggests policy collapse / templating.
        """
        recent = self.completions_history[-window:]
        if len(recent) < 2:
            return 0.0

        overlaps = []
        for i in range(len(recent) - 1):
            words_a = set(recent[i].lower().split())
            words_b = set(recent[i + 1].lower().split())
            if not words_a or not words_b:
                continue
            jaccard = len(words_a & words_b) / len(words_a | words_b)
            overlaps.append(jaccard)

        rate = sum(overlaps) / len(overlaps) if overlaps else 0.0
        if self._current_level:
            self.level_metrics[self._current_level]["template_overlap_rates"].append(
                round(rate, 4)
            )
        return rate

    @property
    def outcome_accuracy(self) -> float:
        if not self.outcome_matches:
            return 0.0
        return sum(self.outcome_matches) / len(self.outcome_matches)

    @property
    def api_success_rate(self) -> float:
        total = self.reward_source_counts.get("api", 0) + self.reward_source_counts.get("local", 0)
        if total == 0:
            return 1.0
        return self.reward_source_counts.get("api", 0) / total

    def summary(self) -> Dict[str, Any]:
        """Return full metrics summary for logging."""
        return {
            "overall_outcome_accuracy": round(self.outcome_accuracy, 4),
            "reward_source_counts": self.reward_source_counts,
            "api_success_rate": round(self.api_success_rate, 4),
            "template_overlap_rate": round(self.compute_template_overlap(), 4),
            "per_level": {
                level: {
                    "reward_source": data["reward_source"],
                    "outcome_accuracy": round(
                        sum(data["outcome_accuracy"]) / len(data["outcome_accuracy"]), 4
                    ) if data["outcome_accuracy"] else 0.0,
                    "template_overlap_rates": data["template_overlap_rates"][-5:],  # last 5
                }
                for level, data in self.level_metrics.items()
            },
        }

    def save(self, output_dir: str):
        """Save metrics summary to JSON."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "training_metrics.json"
        path.write_text(json.dumps(self.summary(), indent=2))
        print(f"  [metrics] Saved to {path}")
