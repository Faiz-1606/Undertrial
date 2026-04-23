"""
UndertriAI — OpenEnv Client
Use this to connect to a running UndertriAI environment server.
"""

import json
from typing import Any, Dict, Optional

try:
    import httpx  # type: ignore
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from .models import (
    BailAction, CaseObservation, AccusedProfile,
    RequestDocumentAction, FlagInconsistencyAction,
    CrossReferencePrecedentAction, ComputeStatutoryEligibilityAction,
    AssessSuretyAction, ClassifyBailTypeAction,
    ReadSubmissionsAction, AssessFlightRiskAction,
    CheckCaseFactorsAction, ApplyProportionalityAction,
    SubmitMemoAction,
    StepResult,
)

try:
    from openenv.core.env_client import EnvClient  # type: ignore
except ImportError:
    class EnvClient:
        """Stub when openenv-core is not installed."""
        def __init__(self, base_url: str):
            self.base_url = base_url.rstrip("/")


class UndertriAIEnv(EnvClient):
    """
    HTTP client for the UndertriAI bail assessment environment.

    Connects to a running UndertriAI FastAPI server (local or HF Spaces).

    Usage (sync):
        env = UndertriAIEnv(base_url="https://draken1606-undertrial-ai.hf.space")
        obs_data = env.reset(stage=1)
        result = env.step(ComputeStatutoryEligibilityAction(
            sections_invoked=["420"],
            max_sentence_years=7.0,
            custody_months=8.0,
            special_law_applicable=False,
        ))
        result = env.step(SubmitMemoAction(
            flight_risk="Low",
            flight_risk_justification="No prior record, permanent resident.",
            statutory_eligible=True,
            statutory_computation="IPC 420 → max 7 yrs → 42 months threshold → served 8 months",
            grounds_for_bail=["No flight risk", "Family ties"],
            grounds_against_bail=["Investigation pending"],
            recommended_outcome="Bail Granted",
            recommended_conditions=["Surety ₹25,000", "Weekly reporting"],
        ))
        print(result["reward"])   # e.g. 0.78
        print(result["info"])     # Full reward breakdown
    """

    def __init__(self, base_url: str = "https://draken1606-undertrial-ai.hf.space"):
        super().__init__(base_url)
        self.base_url = base_url.rstrip("/")
        self._session_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, stage: int = 1) -> Dict[str, Any]:
        """Start a new episode. Returns the initial case observation as a dict."""
        resp = self._post("/reset", params={"stage": stage})
        self._session_id = resp.get("session_id")
        return resp

    def step(self, action: BailAction) -> Dict[str, Any]:
        """
        Execute one action. Returns dict with keys:
            observation, reward, done, info, session_id
        """
        if self._session_id is None:
            raise RuntimeError("Call reset() before step().")
        payload = {
            "session_id": self._session_id,
            "action": action.model_dump(),
        }
        return self._post("/step", json=payload)

    def state(self) -> Dict[str, Any]:
        """Return current episode metadata."""
        if self._session_id is None:
            raise RuntimeError("Call reset() before state().")
        return self._get("/state", params={"session_id": self._session_id})

    def health(self) -> Dict[str, Any]:
        """Check if the server is live."""
        return self._get("/health")

    def tools(self) -> Dict[str, Any]:
        """List available tool signatures."""
        return self._get("/tools")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Dict = None) -> Dict[str, Any]:
        if not _HTTPX_AVAILABLE:
            raise ImportError("Install httpx: pip install httpx")
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{self.base_url}{path}", params=params)
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, params: Dict = None, json: Dict = None) -> Dict[str, Any]:
        if not _HTTPX_AVAILABLE:
            raise ImportError("Install httpx: pip install httpx")
        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{self.base_url}{path}", params=params, json=json)
            resp.raise_for_status()
            return resp.json()


# ---------------------------------------------------------------------------
# Convenience re-exports so users only need to import from undertrial_ai
# ---------------------------------------------------------------------------
__all__ = [
    "UndertriAIEnv",
    "BailAction",
    "CaseObservation",
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
    "SubmitMemoAction",
]
