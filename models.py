"""
UndertriAI — Pydantic Models
Defines all Action and Observation types for the bail assessment environment.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OpenEnv base stubs (avoids hard import dependency when running locally)
# ---------------------------------------------------------------------------
try:
    from openenv.core.models import Action, Observation, State, StepResult  # type: ignore
except ImportError:
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        pass

    class State(BaseModel):
        episode_id: str = ""
        step_count: int = 0

    class StepResult(BaseModel):
        observation: Any
        reward: float = 0.0
        done: bool = False
        info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ACTIONS — tool calls the agent can make before submitting the final memo
# ---------------------------------------------------------------------------

class RequestDocumentAction(Action):
    """Request a missing document (surety affidavit, prior judgment, etc.)"""
    tool_name: Literal["request_document"] = "request_document"
    document_type: str = Field(
        ...,
        description="Type of document to request: surety_affidavit | prior_judgment | fir_copy | medical_report | employment_proof",
    )
    justification: str = Field(..., description="Why this document is needed")


class FlagInconsistencyAction(Action):
    """Flag a legal inconsistency in the charge or prosecution argument."""
    tool_name: Literal["flag_inconsistency"] = "flag_inconsistency"
    inconsistency: str = Field(..., description="Description of the inconsistency found")
    severity: Literal["minor", "major", "fatal"] = Field(
        ..., description="Severity: minor=procedural, major=affects merits, fatal=vitiates proceedings"
    )
    location: str = Field(..., description="Where in the record the inconsistency appears")


class CrossReferencePrecedentAction(Action):
    """Retrieve and cite a relevant precedent from the case database."""
    tool_name: Literal["cross_reference_precedent"] = "cross_reference_precedent"
    query: str = Field(..., description="Legal principle or scenario to search for")
    jurisdiction: Optional[str] = Field(None, description="Preferred jurisdiction (e.g., 'Supreme Court', 'Delhi HC')")
    crime_category: Optional[str] = Field(None, description="Narrow search by crime category")


class ComputeStatutoryEligibilityAction(Action):
    """Check if accused has served half the maximum sentence (default bail eligibility)."""
    tool_name: Literal["compute_statutory_eligibility"] = "compute_statutory_eligibility"
    sections_invoked: List[str] = Field(..., description="IPC/BNSS sections charged under")
    max_sentence_years: float = Field(..., description="Maximum sentence for the most serious charge in years")
    custody_months: float = Field(..., description="Months in custody to date")
    special_law_applicable: bool = Field(False, description="Whether NDPS/UAPA/PMLA or similar special law applies")


class AssessSuretyAction(Action):
    """Evaluate financial viability of the proposed surety."""
    tool_name: Literal["assess_surety"] = "assess_surety"
    proposed_amount: int = Field(..., description="Proposed surety amount in INR")
    accused_occupation: str = Field(..., description="Occupation of accused")
    income_estimate: Optional[int] = Field(None, description="Estimated monthly income in INR")
    surety_relation: Optional[str] = Field(None, description="Relation of surety to accused")


class ClassifyBailTypeAction(Action):
    """Determine whether grounds support conditional bail, absolute bail, or denial."""
    tool_name: Literal["classify_bail_type"] = "classify_bail_type"
    grounds_for: List[str] = Field(..., description="List of grounds supporting bail")
    grounds_against: List[str] = Field(..., description="List of grounds opposing bail")
    accused_category: Optional[str] = Field(None, description="First-time offender | repeat | undertrial | convict")


class ReadSubmissionsAction(Action):
    """Read and summarise prosecution or defence submissions on record."""
    tool_name: Literal["read_submissions"] = "read_submissions"
    party: Literal["prosecution", "defence", "both"] = Field(
        ..., description="Which party's submissions to read"
    )
    focus: Optional[str] = Field(
        None, description="Specific legal issue to focus on (e.g. 'flight risk', 'BNSS 479')"
    )


class AssessFlightRiskAction(Action):
    """Systematically assess the accused's flight risk based on case factors."""
    tool_name: Literal["assess_flight_risk"] = "assess_flight_risk"
    roots_in_community: Optional[str] = Field(
        None, description="Evidence of local ties: family, employment, property"
    )
    prior_absconding: bool = Field(False, description="Has the accused ever absconded before?")
    passport_status: Optional[str] = Field(
        None, description="surrendered | impounded | at-large | unknown"
    )
    severity_of_offence: Literal["minor", "moderate", "serious", "heinous"] = Field(
        ..., description="Gravity of the offence (determines flight incentive)"
    )


class CheckCaseFactorsAction(Action):
    """Examine specific case factors relevant to bail determination."""
    tool_name: Literal["check_case_factors"] = "check_case_factors"
    factors_to_check: List[str] = Field(
        ...,
        description="Factors to examine, e.g.: 'nature_of_offence', 'victim_vulnerability', "
                    "'evidence_tampering_risk', 'co_accused_bail_status', 'recovery_of_property'"
    )


class ApplyProportionalityAction(Action):
    """Apply proportionality principle: custody duration vs. maximum sentence vs. trial timeline."""
    tool_name: Literal["apply_proportionality"] = "apply_proportionality"
    custody_months: float = Field(..., description="Months in custody to date")
    max_sentence_years: float = Field(..., description="Maximum sentence for the most serious charge")
    expected_trial_months: Optional[float] = Field(
        None, description="Estimated months until trial completion (if known)"
    )


class PullCriminalHistoryAction(Action):
    """Pull the accused's prior criminal record, bail history, and conviction status."""
    tool_name: Literal["pull_criminal_history"] = "pull_criminal_history"
    include_bail_history: bool = Field(
        default=True, description="Whether to include prior bail applications and outcomes"
    )

class IssueOrderAction(Action):
    """
    TERMINAL ACTION — Issue a bail order (Block 4.3 spec alias for submit_memo).

    Maps order_type to recommended_outcome:
      grant       → Bail Granted
      deny        → Bail Denied
      conditional → Bail Granted (conditions must be provided)

    This is the short-form action compatible with the OpenEnv compliance
    checklist spec (`issue_order(grant | deny | conditional)`). Use
    submit_memo for the full structured memo form.
    """
    tool_name: Literal["issue_order"] = "issue_order"
    order_type: Literal["grant", "deny", "conditional"] = Field(
        ..., description="Type of bail order: grant | deny | conditional"
    )
    flight_risk: Literal["Low", "Medium", "High"] = Field(
        ..., description="Flight risk classification"
    )
    flight_risk_justification: str = Field(
        ..., description="Justification for flight risk assessment referencing case facts"
    )
    statutory_eligible: bool = Field(
        ..., description="Whether accused qualifies for default bail under statute"
    )
    statutory_computation: str = Field(
        ..., description="Computation: section → max sentence → threshold → custody served"
    )
    grounds_for_bail: List[str] = Field(..., description="Grounds supporting bail")
    grounds_against_bail: List[str] = Field(..., description="Grounds opposing bail")
    recommended_conditions: Optional[List[str]] = Field(
        None, description="Bail conditions (required when order_type='conditional')"
    )
    confidence: Literal["High", "Medium", "Low"] = "Medium"


class SubmitMemoAction(Action):
    """
    TERMINAL ACTION — Submit the structured bail assessment memo.
    This triggers reward computation against the ground truth.
    """
    tool_name: Literal["submit_memo"] = "submit_memo"

    # Flight risk assessment
    flight_risk: Literal["Low", "Medium", "High"] = Field(
        ..., description="Flight risk classification"
    )
    flight_risk_justification: str = Field(
        ..., description="Justification for flight risk score with reference to case facts"
    )

    # Statutory eligibility
    statutory_eligible: bool = Field(
        ..., description="Whether accused is eligible for bail under the statute"
    )
    statutory_computation: str = Field(
        ..., description="Show the computation: sections, max sentence, time served, threshold"
    )

    # Balanced assessment
    grounds_for_bail: List[str] = Field(
        ..., description="Specific grounds from case facts supporting bail"
    )
    grounds_against_bail: List[str] = Field(
        ..., description="Specific grounds from prosecution / case facts opposing bail"
    )

    # Recommendation
    recommended_outcome: Literal["Bail Granted", "Bail Denied"] = Field(
        ..., description="Final recommendation: Bail Granted | Bail Denied"
    )
    recommended_conditions: Optional[List[str]] = Field(
        None,
        description="Conditions if bail granted: surety amount, travel restrictions, reporting, etc."
    )

    # Confidence
    confidence: Literal["High", "Medium", "Low"] = Field(
        "Medium", description="Confidence in the recommendation"
    )


# Union of all valid agent actions
BailAction = Union[
    RequestDocumentAction,
    FlagInconsistencyAction,
    CrossReferencePrecedentAction,
    ComputeStatutoryEligibilityAction,
    AssessSuretyAction,
    ClassifyBailTypeAction,
    ReadSubmissionsAction,
    AssessFlightRiskAction,
    CheckCaseFactorsAction,
    ApplyProportionalityAction,
    PullCriminalHistoryAction,
    IssueOrderAction,   # Block 4.3: spec-compliant alias for submit_memo
    SubmitMemoAction,
]


# ---------------------------------------------------------------------------
# OBSERVATION — what the agent sees at each step
# ---------------------------------------------------------------------------

class AccusedProfile(BaseModel):
    name: str
    gender: str
    occupation: Optional[str] = None
    region: Optional[str] = None
    prior_cases: Optional[str] = None
    bail_type: Optional[str] = None


class CaseObservation(Observation):
    """Full state the agent observes at each step of an episode."""
    case_id: str
    case_title: str

    # Case materials
    charge_sheet: str = Field(..., description="Facts and FIR summary")
    ipc_sections: List[str] = Field(..., description="Sections invoked (IPC or BNSS)")
    crime_type: str
    court: str
    date: str

    # Accused
    accused_profile: AccusedProfile

    # Arguments
    prosecution_arguments: List[str]
    defence_arguments: List[str]
    legal_issues: List[str]

    # Context
    cited_precedents: List[str] = Field(default_factory=list)
    documents_available: List[str] = Field(default_factory=list)

    # Episode state
    action_result: Optional[str] = None
    action_history: List[str] = Field(
        default_factory=list,
        description="Ordered log of all tool results seen so far this episode",
    )
    flags_raised: List[str] = Field(default_factory=list)
    precedents_retrieved: List[str] = Field(default_factory=list)
    memo_submitted: bool = False
    step_count: int = 0

    # Schema drift indicator (Patronus AI bonus track)
    schema_variant: str = "standard"  # "standard" | "bnss" | "regional_<state>"


# ---------------------------------------------------------------------------
# REWARD BREAKDOWN — returned in StepResult.info when memo is submitted
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    outcome_match: float          # 0.0 – 1.0
    flight_risk_accuracy: float   # 0.0 – 1.0
    statutory_accuracy: float     # 0.0 – 1.0
    condition_appropriateness: float  # 0.0 – 1.0
    bias_penalty: float           # 0.0 – 1.0 (subtracted)
    total_reward: float           # final R

    ground_truth_outcome: str
    agent_outcome: str
    explanation: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Base types
    "Action", "Observation", "State", "StepResult",
    # Actions (12 tool types + 1 terminal alias)
    "RequestDocumentAction",
    "FlagInconsistencyAction",
    "CrossReferencePrecedentAction",
    "ComputeStatutoryEligibilityAction",
    "AssessSuretyAction",
    "ClassifyBailTypeAction",
    "ReadSubmissionsAction",
    "AssessFlightRiskAction",
    "CheckCaseFactorsAction",
    "ApplyProportionalityAction",
    "PullCriminalHistoryAction",
    "IssueOrderAction",
    "SubmitMemoAction",
    # Union type
    "BailAction",
    # Observation / state
    "AccusedProfile",
    "CaseObservation",
    "RewardBreakdown",
]
