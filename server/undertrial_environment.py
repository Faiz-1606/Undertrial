"""
UndertriAI — Core OpenEnv Environment (Server-Side)
Implements the bail assessment RL training environment.
"""

import concurrent.futures
import uuid
from typing import Any, Dict, List, Optional

from .dataset import BailDataset
from .reward import compute_reward
from .schema_drift import maybe_apply_drift

try:
    from openenv.core import Environment  # type: ignore
except ImportError:
    try:
        from openenv_core import Environment  # type: ignore
    except ImportError:
        class Environment:  # type: ignore
            pass

try:
    from openenv.core.models import StepResult  # type: ignore
except ImportError:
    from pydantic import BaseModel
    class StepResult(BaseModel):  # type: ignore
        observation: Any
        reward: float = 0.0
        done: bool = False
        info: dict = {}

from ..models import (
    BailAction, CaseObservation, AccusedProfile,
    RequestDocumentAction, FlagInconsistencyAction,
    CrossReferencePrecedentAction, ComputeStatutoryEligibilityAction,
    AssessSuretyAction, ClassifyBailTypeAction, SubmitMemoAction,
)
from .precedent_db import PrecedentDB


class UndertriAIEnvironment(Environment):
    """
    Bail Assessment Environment — OpenEnv compliant.

    The agent reads a bail case and iteratively calls legal tools before
    submitting a structured bail recommendation memo. Reward is computed
    deterministically against the real High Court decision (ground_truth).
    """

    # Concurrent sessions are safe: each instance is independent (session_id isolation)
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 10  # Maximum tool calls before forcing memo submission

    def __init__(
        self,
        episodes_dir: Optional[str] = None,
        initial_stage: int = 1,
    ):
        super().__init__()  # Sets self.rubric = None and self.transform = None
        self.dataset     = BailDataset(episodes_dir=episodes_dir)
        self.precedents  = PrecedentDB()
        self._episode: Optional[Dict[str, Any]] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._flags: List[str] = []
        self._retrieved_precedents: List[str] = []
        self._current_stage: int = initial_stage

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        stage: Optional[int] = None,
        **kwargs,
    ) -> CaseObservation:
        """Start a new episode. Returns initial case observation."""
        s = stage or self._current_stage

        # reset() timeout guard — prevent infinite hang on slow dataset.sample_episode()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self.dataset.sample_episode, s, True, seed)
            try:
                self._episode = fut.result(timeout=5.0)
            except concurrent.futures.TimeoutError:
                raise RuntimeError("reset() timed out after 5s — check dataset loading.")

        self._episode_id    = episode_id or str(uuid.uuid4())
        self._step_count    = 0
        self._flags         = []
        self._retrieved_precedents  = []
        self._action_history: List[str] = []
        self._statutory_tool_called: bool = False  # M2: process reward tracking
        return self._make_observation(action_result=None)

    def step(
        self,
        action: BailAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> StepResult:
        """Execute one agent action. Returns StepResult with reward only when done."""
        if self._episode is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1

        # ---- Terminal action: submit memo ----
        if isinstance(action, SubmitMemoAction):
            # Penalty for skipping all tool calls
            # Increased to 0.40 so instant-submit can never be profitable by chance
            no_tool_penalty = 0.40 if self._step_count == 1 else 0.0

            reward_dict = compute_reward(
                agent_outcome     = action.recommended_outcome,
                agent_flight_risk = action.flight_risk,
                agent_eligible    = action.statutory_eligible,
                agent_computation = action.statutory_computation,
                agent_conditions  = action.recommended_conditions or [],
                episode           = self._episode,
                step_count        = self._step_count,
                max_steps         = self.MAX_STEPS,
                statutory_tool_used = self._statutory_tool_called,  # M2
            )
            # Apply skip penalty (can push total legitimately negative)
            reward_dict["total_reward"] = round(reward_dict["total_reward"] - no_tool_penalty, 4)
            reward_dict["tool_skip_penalty"] = no_tool_penalty

            obs = self._make_observation(
                action_result=self._format_memo_result(action, reward_dict),
                memo_submitted=True,
            )
            return StepResult(
                observation=obs,
                reward=reward_dict["total_reward"],
                done=True,
                info=reward_dict,
            )

        # ---- Tool actions with optional timeout enforcement ----
        if isinstance(action, ComputeStatutoryEligibilityAction):
            self._statutory_tool_called = True  # M2: track for process reward

        if timeout_s is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._dispatch_tool, action)
                try:
                    result = future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    obs = self._make_observation(
                        action_result=f"TIMEOUT: tool call exceeded {timeout_s}s limit",
                        memo_submitted=False,
                    )
                    return StepResult(
                        observation=obs,
                        reward=-0.05,
                        done=False,
                        info={"timeout": True, "tool": type(action).__name__},
                    )
        else:
            result = self._dispatch_tool(action)

        # Accumulate action history (Gap 4)
        summary = f"[Step {self._step_count}] {type(action).__name__}: {result[:120]}..."
        self._action_history.append(summary)

        # Force submit if max steps reached
        done = (self._step_count >= self.MAX_STEPS)
        reward = -0.1 if done else 0.0  # Small penalty for exhausting budget

        obs = self._make_observation(action_result=result, memo_submitted=done)
        return StepResult(observation=obs, reward=reward, done=done, info={})

    @property
    def state(self):
        """Return episode metadata (OpenEnv State interface)."""
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "stage": self._current_stage,
            "case_id": self._episode.get("case_id", "") if self._episode else "",
        }

    def set_stage(self, stage: int) -> None:
        """Advance the curriculum stage."""
        self._current_stage = stage
        self.dataset.set_stage(stage)

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, action: BailAction) -> str:
        ep = self._episode

        if isinstance(action, RequestDocumentAction):
            if action.document_type in ep.get("documents_available", []):
                return f"✓ Document retrieved: {action.document_type}. Review attached."
            return f"✗ Document '{action.document_type}' not available in this case record."

        elif isinstance(action, FlagInconsistencyAction):
            flag_msg = f"[{action.severity.upper()}] {action.inconsistency} (at: {action.location})"
            self._flags.append(flag_msg)
            return f"Inconsistency flagged ({action.severity}): {action.inconsistency}"

        elif isinstance(action, CrossReferencePrecedentAction):
            results = self.precedents.search(
                query=action.query,
                jurisdiction=action.jurisdiction,
                crime_category=action.crime_category,
            )
            self._retrieved_precedents.extend(results)
            if results:
                return "Precedents found:\n" + "\n".join(f"  • {r}" for r in results)
            return "No directly applicable precedents found in database."

        elif isinstance(action, ComputeStatutoryEligibilityAction):
            half_months = (action.max_sentence_years * 12) / 2
            eligible = action.custody_months >= half_months and not action.special_law_applicable
            pct = round((action.custody_months / (action.max_sentence_years * 12)) * 100, 1) if action.max_sentence_years else 0
            return (
                f"Statutory Eligibility Analysis:\n"
                f"  Sections: {', '.join(action.sections_invoked)}\n"
                f"  Max Sentence: {action.max_sentence_years} years ({action.max_sentence_years*12:.0f} months)\n"
                f"  Threshold (50%): {half_months:.1f} months\n"
                f"  Time Served: {action.custody_months} months ({pct}%)\n"
                f"  Special Law: {'Yes — default bail restricted' if action.special_law_applicable else 'No'}\n"
                f"  → ELIGIBLE FOR DEFAULT BAIL: {'YES ✓' if eligible else 'NO ✗'}"
            )

        elif isinstance(action, AssessSuretyAction):
            feasible = action.proposed_amount <= (action.income_estimate or 50000) * 3
            return (
                f"Surety Assessment:\n"
                f"  Proposed Amount: ₹{action.proposed_amount:,}\n"
                f"  Accused Occupation: {action.accused_occupation}\n"
                f"  Income Estimate: ₹{action.income_estimate:,}/month\n"
                f"  → {'FINANCIALLY FEASIBLE ✓' if feasible else 'AMOUNT MAY BE EXCESSIVE — consider reduction'}"
            )

        elif isinstance(action, ClassifyBailTypeAction):
            pros_count = len(action.grounds_against)
            def_count  = len(action.grounds_for)
            if def_count > pros_count:
                suggestion = "Conditional Bail (grounds for bail outweigh grounds against)"
            elif pros_count > def_count:
                suggestion = "Bail Denial (grounds against outweigh grounds for bail)"
            else:
                suggestion = "Contested — full assessment required"
            return (
                f"Bail Type Classification:\n"
                f"  Grounds FOR bail ({def_count}): {'; '.join(action.grounds_for[:3])}\n"
                f"  Grounds AGAINST bail ({pros_count}): {'; '.join(action.grounds_against[:3])}\n"
                f"  → Preliminary classification: {suggestion}"
            )

        return f"Unknown action type: {type(action).__name__}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        action_result: Optional[str] = None,
        memo_submitted: bool = False,
    ) -> CaseObservation:
        ep = self._episode
        profile_data = ep.get("accused_profile", {})
        profile = AccusedProfile(
            name    = profile_data.get("name", "Unknown"),
            gender  = profile_data.get("gender", "Unknown"),
            occupation = profile_data.get("occupation"),
            region  = profile_data.get("region"),
            prior_cases = profile_data.get("prior_cases"),
            bail_type = profile_data.get("bail_type"),
        )
        init_precedents = self.precedents.get_initial_precedents(ep)

        return CaseObservation(
            case_id   = ep.get("case_id", ""),
            case_title = ep.get("case_title", ""),
            charge_sheet = ep.get("charge_sheet", ""),
            ipc_sections = ep.get("ipc_sections", []),
            crime_type = ep.get("crime_type", ""),
            court     = ep.get("court", ""),
            date      = ep.get("date", ""),
            accused_profile     = profile,
            prosecution_arguments = ep.get("prosecution_arguments", []),
            defence_arguments   = ep.get("defence_arguments", []),
            legal_issues        = ep.get("legal_principles", []),
            cited_precedents    = init_precedents + self._retrieved_precedents,
            documents_available = ep.get("documents_available", []),
            action_result       = action_result,
            action_history      = list(self._action_history),  # Gap 4
            flags_raised        = list(self._flags),
            precedents_retrieved = list(self._retrieved_precedents),
            memo_submitted      = memo_submitted,
            step_count          = self._step_count,
            schema_variant      = ep.get("schema_variant", "standard"),
        )

    def _format_memo_result(self, memo: SubmitMemoAction, reward: Dict[str, Any]) -> str:
        lines = [
            "═══ BAIL ASSESSMENT MEMO SUBMITTED ═══",
            f"Recommended Outcome:  {memo.recommended_outcome}",
            f"Flight Risk:          {memo.flight_risk}",
            f"Statutory Eligible:   {'Yes' if memo.statutory_eligible else 'No'}",
            f"Confidence:           {memo.confidence}",
            "",
            "── Reward Breakdown ──",
            f"  Outcome Match:        {reward['outcome_match']:.2f} × 0.40",
            f"  Flight Risk Accuracy: {reward['flight_risk_accuracy']:.2f} × 0.20",
            f"  Statutory Accuracy:   {reward['statutory_accuracy']:.2f} × 0.20",
            f"  Condition Score:      {reward['condition_appropriateness']:.2f} × 0.20",
            f"  Bias Penalty:       − {reward['bias_penalty']:.2f} × 0.30",
            f"  ─────────────────────────────────",
            f"  TOTAL REWARD:         {reward['total_reward']:.4f}",
            "",
            f"Ground Truth: {reward['ground_truth_outcome']}",
        ]
        return "\n".join(lines)
