"""
UndertriAI — Performance Tracker (Theme 4: Self-Improvement)

Tracks the agent's running performance profile across dimensions
and uses it to drive adaptive curriculum decisions.

Pure Python — no server/training/FastAPI dependencies.
"""

import warnings
from collections import deque
from typing import Any, Dict, List, Optional


class ExponentialMean:
    """Exponential moving average with configurable decay."""

    __slots__ = ("alpha", "value", "count")

    def __init__(self, alpha: float = 0.1, initial: float = 0.5):
        self.alpha = alpha
        self.value = initial
        self.count = 0

    def update(self, x: float) -> None:
        self.value = self.alpha * x + (1 - self.alpha) * self.value
        self.count += 1

    def get(self) -> float:
        return self.value


class PerformanceTracker:
    """
    Tracks agent performance across crime types, stages, and reward
    components. Drives adaptive episode selection and stage promotion.

    Thread-safe for single-session use (no locks needed).
    All public methods handle missing/malformed input gracefully.
    """

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha

        # Per-crime-type EMA of total reward
        self.per_crime_type: Dict[str, ExponentialMean] = {}

        # Per-stage EMA of total reward
        self.per_stage: Dict[int, ExponentialMean] = {
            s: ExponentialMean(alpha=alpha) for s in range(1, 5)
        }

        # Last 50 total rewards (for stage promotion smoothing)
        self.recent_rewards: deque = deque(maxlen=50)

        # Bias fire rate: 1.0 when penalty fired, 0.0 when not
        self.bias_fire_rate: ExponentialMean = ExponentialMean(alpha=alpha)

        # Tool usage counts (cumulative per session)
        self.tool_usage: Dict[str, int] = {}

        # Episode counters
        self.episodes_seen: int = 0
        self.stage_episodes: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

        # Recent case performance for failure-replay
        self._recent_case_rewards: deque = deque(maxlen=30)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        episode: Dict[str, Any],
        reward_components: Dict[str, Any],
        tools_used: Optional[List[str]] = None,
    ) -> None:
        """
        Update all internal state from a completed episode.

        Handles missing keys gracefully — never raises on malformed input.
        """
        try:
            total = float(reward_components.get("total_reward",
                          reward_components.get("total", 0.0)))
        except (TypeError, ValueError):
            total = 0.0

        # Update recent rewards
        self.recent_rewards.append(total)
        self.episodes_seen += 1

        # Per-crime-type tracking
        crime_type = ""
        try:
            crime_type = str(episode.get("crime_type", "")).strip()
        except Exception:
            pass

        if crime_type:
            if crime_type not in self.per_crime_type:
                self.per_crime_type[crime_type] = ExponentialMean(
                    alpha=self._alpha
                )
            self.per_crime_type[crime_type].update(total)

        # Per-stage tracking
        stage = 1
        try:
            stage = int(episode.get("curriculum_stage", 1))
        except (TypeError, ValueError):
            stage = 1
        if 1 <= stage <= 4:
            self.per_stage[stage].update(total)
            self.stage_episodes[stage] = self.stage_episodes.get(stage, 0) + 1

        # Bias fire rate
        try:
            bias_val = float(reward_components.get("bias_penalty", 0.0))
            self.bias_fire_rate.update(1.0 if bias_val > 0.01 else 0.0)
        except (TypeError, ValueError):
            pass

        # Tool usage
        if tools_used:
            for tool in tools_used:
                t = str(tool)
                self.tool_usage[t] = self.tool_usage.get(t, 0) + 1

        # Track case_id → reward for failure-replay
        case_id = ""
        try:
            case_id = str(episode.get("case_id", ""))
        except Exception:
            pass
        if case_id:
            self._recent_case_rewards.append((case_id, total, stage))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def weakest_domain(self) -> Optional[str]:
        """
        Returns the crime_type with the lowest EMA reward.
        Returns None if fewer than 5 episodes seen total or no crime type
        has at least 3 observations.
        """
        if self.episodes_seen < 5:
            return None

        candidates = [
            (ct, ema.get())
            for ct, ema in self.per_crime_type.items()
            if ema.count >= 3
        ]
        if not candidates:
            return None

        return min(candidates, key=lambda x: x[1])[0]

    def suggest_next_stage(self) -> int:
        """
        Returns the recommended stage (1-4) based on readiness thresholds.
        Never demotes — returns highest eligible stage.
        """
        current = 1

        # Stage 1 → 2: EMA >= 0.65 AND at least 20 episodes
        if (self.per_stage[1].get() >= 0.65
                and self.stage_episodes.get(1, 0) >= 20):
            current = 2

        # Stage 2 → 3: EMA >= 0.55 AND at least 50 episodes
        if (current >= 2
                and self.per_stage[2].get() >= 0.55
                and self.stage_episodes.get(2, 0) >= 50):
            current = 3

        # Stage 3 → 4: EMA >= 0.50 AND at least 20 episodes
        if (current >= 3
                and self.per_stage[3].get() >= 0.50
                and self.stage_episodes.get(3, 0) >= 20):
            current = 4

        return current

    def should_generate_synthetic(self, crime_type: str) -> bool:
        """
        Returns True if the agent has mastered this crime type domain
        (EMA > 0.70 with at least 10 observations).
        """
        ema = self.per_crime_type.get(crime_type)
        if ema is None:
            return False
        return ema.get() > 0.70 and ema.count >= 10

    def get_recent_failures(self, threshold: float = 0.40) -> List[str]:
        """
        Returns case_ids from recent episodes where reward was below threshold.
        Used by AdaptiveSelector for failure-replay.
        """
        return [
            case_id
            for case_id, reward, _ in self._recent_case_rewards
            if reward < threshold
        ]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_profile(self) -> Dict[str, Any]:
        """
        Returns a fully JSON-serializable profile dict.
        No class instances — all values are primitive types.
        """
        recent = list(self.recent_rewards)
        recent_mean = sum(recent) / len(recent) if recent else 0.0

        return {
            "per_crime_type": {
                ct: round(ema.get(), 4)
                for ct, ema in self.per_crime_type.items()
            },
            "per_stage": {
                str(s): round(ema.get(), 4)
                for s, ema in self.per_stage.items()
            },
            "bias_fire_rate": round(self.bias_fire_rate.get(), 4),
            "tool_usage": dict(self.tool_usage),
            "episodes_seen": self.episodes_seen,
            "stage_episodes": dict(self.stage_episodes),
            "weakest_domain": self.weakest_domain(),
            "suggested_stage": self.suggest_next_stage(),
            "recent_mean_reward": round(recent_mean, 4),
        }

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """
        Clears transient session state but preserves accumulated
        per-crime-type and per-stage learning.
        """
        self.recent_rewards.clear()
        self.tool_usage.clear()
        self._recent_case_rewards.clear()
