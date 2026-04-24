"""
UndertriAI — Adaptive Episode Selector (Theme 4: Self-Improvement)

Wraps the existing BailDataset to provide performance-aware episode
selection when adaptive mode is enabled. Falls back to uniform random
(identical to existing behavior) when adaptive=False.
"""

import random
from typing import Any, Dict, List, Optional

from .performance_tracker import PerformanceTracker


class AdaptiveSelector:
    """
    Performance-aware episode selector.

    Selection strategy (applied in order when adaptive=True):
      60%: sample from the weakest crime-type domain in current_stage
      30%: replay cases where recent performance was poor (reward < 0.40)
      10%: uniform random from current_stage (exploration)

    Always returns a valid episode dict. Never raises.
    """

    def __init__(self, dataset, tracker: PerformanceTracker):
        """
        Args:
            dataset: BailDataset instance (has _episodes, sample_episode)
            tracker: PerformanceTracker instance driving selection
        """
        self.dataset = dataset
        self.tracker = tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_episode(self, current_stage: int) -> Dict[str, Any]:
        """
        Performance-aware selection for adaptive mode.

        60% weakest domain → 30% failure replay → 10% exploration.
        Falls back to uniform on any failure.
        """
        try:
            roll = random.random()

            if roll < 0.60:
                # Try weakest domain
                ep = self._select_weakest_domain(current_stage)
                if ep is not None:
                    return ep

            if roll < 0.90:
                # Try failure replay
                ep = self._select_failure_replay(current_stage)
                if ep is not None:
                    return ep

            # 10% exploration or fallback
            return self.select_episode_uniform(current_stage)

        except Exception:
            # Absolute fallback — never crash
            return self.select_episode_uniform(current_stage)

    def select_episode_uniform(self, current_stage: int) -> Dict[str, Any]:
        """
        Pure random selection from current_stage.
        Identical to existing BailDataset.sample_episode() behavior.
        """
        return self.dataset.sample_episode(stage=current_stage)

    # ------------------------------------------------------------------
    # Internal strategies
    # ------------------------------------------------------------------

    def _select_weakest_domain(
        self, current_stage: int
    ) -> Optional[Dict[str, Any]]:
        """
        Select an episode from the weakest crime-type domain.
        Returns None if no weak domain identified or no matching episodes.
        """
        weak_domain = self.tracker.weakest_domain()
        if weak_domain is None:
            return None

        # Find episodes matching this crime type in the current stage
        episodes = self._get_stage_episodes(current_stage)
        matches = [
            ep for ep in episodes
            if str(ep.get("crime_type", "")).strip() == weak_domain
        ]

        if not matches:
            return None

        return random.choice(matches)

    def _select_failure_replay(
        self, current_stage: int
    ) -> Optional[Dict[str, Any]]:
        """
        Replay a case where the agent recently scored below 0.40.
        Returns None if no recent failures or no matching episodes.
        """
        failed_ids = self.tracker.get_recent_failures(threshold=0.40)
        if not failed_ids:
            return None

        # Find episodes matching failed case_ids in current stage
        episodes = self._get_stage_episodes(current_stage)
        matches = [
            ep for ep in episodes
            if ep.get("case_id", "") in failed_ids
        ]

        if not matches:
            return None

        return random.choice(matches)

    def _get_stage_episodes(self, stage: int) -> List[Dict[str, Any]]:
        """Get all episodes for a given stage from the dataset."""
        try:
            eps = self.dataset._episodes.get(stage, [])
            if eps:
                return eps
            # Fallback chain matching BailDataset.sample_episode
            for candidate in [stage - 1, stage + 1, 1, 2, 3, 4]:
                if 1 <= candidate <= 4:
                    eps = self.dataset._episodes.get(candidate, [])
                    if eps:
                        return eps
        except Exception:
            pass
        return []
