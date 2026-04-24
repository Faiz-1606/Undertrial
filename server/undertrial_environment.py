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
    AccusedProfile, CaseObservation,
    BailAction,
    RequestDocumentAction, FlagInconsistencyAction, CrossReferencePrecedentAction,
    ComputeStatutoryEligibilityAction, AssessSuretyAction, ClassifyBailTypeAction,
    ReadSubmissionsAction, AssessFlightRiskAction, CheckCaseFactorsAction,
    ApplyProportionalityAction,
    PullCriminalHistoryAction,
    SubmitMemoAction,
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

        # A8 fix: if episode_id is given, look up that specific case by case_id.
        # Previously episode_id was stored but episode was always sampled randomly.
        found_episode = None
        if episode_id is not None:
            for stage_eps in self.dataset._episodes.values():
                for ep in stage_eps:
                    if ep.get("case_id") == episode_id:
                        found_episode = ep
                        break
                if found_episode:
                    break

        if found_episode:
            self._episode = found_episode
        else:
            # Timeout guard — prevent infinite hang on slow dataset.sample_episode()
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
        self._statutory_tool_called: bool = False
        self._tools_called: set = set()
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
            # 4.5 Hard minimum: agent must have called at least 1 distinct tool before submitting.
            # This is a structural gate — even a skip-penalty can't compensate for zero information.
            if len(self._tools_called) == 0:
                obs = self._make_observation(
                    action_result=(
                        "[BLOCKED] You must call at least one legal tool before submitting a memo. "
                        "Use tools such as compute_statutory_eligibility, assess_flight_risk, "
                        "read_submissions, or check_case_factors first."
                    ),
                    memo_submitted=False,
                )
                return StepResult(
                    observation=obs,
                    reward=-0.15,  # Stronger signal than just a penalty post-submission
                    done=False,
                    info={"blocked": "minimum_tools_not_met", "tools_called": 0},
                )

            # Skip penalty only if submitted on step 1 despite having called a tool
            # (edge case where first action is somehow both a tool and submit)
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
                statutory_tool_used              = self._statutory_tool_called,
                agent_flight_risk_justification  = action.flight_risk_justification,
                agent_grounds_for                = action.grounds_for_bail,
                agent_grounds_against            = action.grounds_against_bail,
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

        # ---- Repeat-action deduplication (5B.2) ----
        tool_key = type(action).__name__
        if tool_key in self._tools_called:
            # Return cached note — no re-execution, no reward gaming
            obs = self._make_observation(
                action_result=(
                    f"[CACHED] {tool_key} was already called this episode. "
                    "The result is already in your action history above. "
                    "Use a different tool or submit your memo."
                ),
                memo_submitted=False,
            )
            return StepResult(observation=obs, reward=-0.05, done=False,
                              info={"cached": True, "tool": tool_key})

        self._tools_called.add(tool_key)

        # ---- Tool actions with optional timeout enforcement ----
        if isinstance(action, ComputeStatutoryEligibilityAction):
            self._statutory_tool_called = True  # track for process reward

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

        elif isinstance(action, ReadSubmissionsAction):
            ep = self._episode
            lines = []
            if action.party in ("prosecution", "both"):
                pros = ep.get("prosecution_arguments", [])
                lines.append("── Prosecution Submissions ──")
                lines += [f"  {i+1}. {a}" for i, a in enumerate(pros)] or ["  (none on record)"]
            if action.party in ("defence", "both"):
                defence = ep.get("defence_arguments", [])
                lines.append("── Defence Submissions ──")
                lines += [f"  {i+1}. {a}" for i, a in enumerate(defence)] or ["  (none on record)"]
            if action.focus:
                lines.append(f"\n[Focus filter: '{action.focus}' — review above for relevance]")
            return "\n".join(lines)

        elif isinstance(action, AssessFlightRiskAction):
            score = 0
            reasons = []
            severity_map = {"minor": 0, "moderate": 1, "serious": 2, "heinous": 3}
            score += severity_map.get(action.severity_of_offence, 1)
            reasons.append(f"Offence severity ({action.severity_of_offence}): +{severity_map.get(action.severity_of_offence, 1)}")
            if action.prior_absconding:
                score += 3; reasons.append("Prior absconding on record: +3")
            if action.passport_status and action.passport_status not in ("surrendered", "impounded"):
                score += 2; reasons.append(f"Passport status ({action.passport_status}): +2")
            if action.roots_in_community:
                score -= 1; reasons.append("Community roots present: −1")
            if score <= 1:   verdict = "Low"
            elif score <= 3: verdict = "Medium"
            else:            verdict = "High"
            return (
                "Flight Risk Assessment:\n"
                + "\n".join(f"  {r}" for r in reasons)
                + f"\n  Total score: {score}"
                + f"\n  → FLIGHT RISK: {verdict}"
            )

        elif isinstance(action, CheckCaseFactorsAction):
            ep = self._episode
            gt = ep.get("ground_truth", {})
            results = []
            for factor in action.factors_to_check:
                f = factor.lower()
                if "offence" in f or "nature" in f:
                    results.append(f"nature_of_offence: {ep.get('crime_type', 'unknown')} — sections: {', '.join(ep.get('ipc_sections', ['n/a']))}")
                elif "prior" in f or "history" in f or "criminal" in f:
                    results.append(f"criminal_history: {ep.get('accused_profile', {}).get('prior_cases', 'no prior cases on record')}")
                elif "co_accused" in f or "parity" in f:
                    parity = gt.get("parity_argument_used", False)
                    results.append(f"co_accused_parity_argument: {'YES — HC relied on parity reasoning' if parity else 'Not applicable in this case'}")
                elif "evidence" in f or "tampering" in f:
                    results.append(f"evidence_tampering_risk: {'flagged' if self._flags else 'no inconsistencies flagged yet — run flag_inconsistency if needed'}")
                elif "victim" in f or "vulnerability" in f:
                    results.append(f"victim_vulnerability: assess from charge sheet — crime_type is {ep.get('crime_type', 'unknown')}")
                else:
                    results.append(f"{factor}: case record does not have a structured field for this — review charge sheet.")
            return "Case Factors Examined:\n" + "\n".join(f"  • {r}" for r in results)

        elif isinstance(action, ApplyProportionalityAction):
            max_months = action.max_sentence_years * 12
            pct = round((action.custody_months / max_months) * 100, 1) if max_months else 0
            bnss_threshold = max_months / 2
            over_threshold = action.custody_months >= bnss_threshold
            lines = [
                "Proportionality Analysis (BNSS 479 / former CrPC 436A):",
                f"  Custody to date:    {action.custody_months:.1f} months",
                f"  Max sentence:       {action.max_sentence_years} years ({max_months:.0f} months)",
                f"  BNSS 479 threshold: {bnss_threshold:.1f} months (50%)",
                f"  Time served:        {pct}% of maximum sentence",
                f"  → Threshold crossed: {'YES — default bail right accrued' if over_threshold else 'NO — threshold not yet met'}",
            ]
            if action.expected_trial_months:
                remaining = action.expected_trial_months
                lines.append(f"  Estimated trial completion: {remaining:.0f} more months")
                if remaining > (max_months - action.custody_months):
                    lines.append("  ⚠️  Projected total custody exceeds maximum sentence — strong proportionality argument for bail")
            return "\n".join(lines)

        elif isinstance(action, PullCriminalHistoryAction):
            ep      = self._episode
            profile = ep.get("accused_profile", {})
            prior   = profile.get("prior_cases", "No prior criminal record on file")
            bail_type = profile.get("bail_type", "Unknown")
            lines = [
                "Criminal History Report:",
                f"  Prior cases: {prior}",
                f"  Bail type context: {bail_type}",
            ]
            if action.include_bail_history:
                # Infer from parity flag and stage whether HC has dealt with bail before
                parity = ep.get("ground_truth", {}).get("parity_argument_used", False)
                lines.append(
                    f"  Prior bail history: {'Co-accused parity argument on record — HC previously granted bail to similarly placed accused' if parity else 'No co-accused parity argument on record'}"
                )
            first_time = prior in ("None", "nil", "no prior", "No prior criminal record on file", None, "")
            lines.append(f"  → Classification: {'FIRST-TIME OFFENDER ✓' if first_time else 'HAS PRIOR RECORD — review above'}")
            return "\n".join(lines)

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
