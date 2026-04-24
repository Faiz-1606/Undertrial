"""
UndertriAI — FastAPI Server App
Wraps UndertriAIEnvironment as an OpenEnv-compatible HTTP + WebSocket server.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import json
import uuid
from typing import List, Optional

logger = logging.getLogger("undertrial")

from .undertrial_environment import UndertriAIEnvironment
from .performance_tracker import PerformanceTracker
from .adaptive_selector import AdaptiveSelector
from .case_generator import generate_variants


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

@dataclass
class SessionState:
    """Per-session state wrapping the environment + Theme 4 components."""
    env: UndertriAIEnvironment
    tracker: PerformanceTracker = field(default_factory=PerformanceTracker)
    adaptive: bool = False
    selector: Optional[AdaptiveSelector] = None
    tools_used: List[str] = field(default_factory=list)
    synthetic_cases_generated: int = 0

    def __post_init__(self):
        if self.selector is None:
            self.selector = AdaptiveSelector(self.env.dataset, self.tracker)


# Session store: session_id → SessionState
_sessions: dict = {}

app = FastAPI(
    title="UndertriAI — Bail Assessment Environment",
    description="OpenEnv-compatible RL training environment for Indian bail decision support.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EPISODES_DIR = os.environ.get("UNDERTRIAL_EPISODES_DIR", None)


def get_or_create_session(session_id: str) -> SessionState:
    """Get existing session or create new one with all Theme 4 components."""
    if session_id not in _sessions:
        env = UndertriAIEnvironment(episodes_dir=EPISODES_DIR)
        _sessions[session_id] = SessionState(env=env)
    return _sessions[session_id]


# ------------------------------------------------------------------
# REST endpoints (existing — preserved exactly)
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the interactive demo UI."""
    html_path = Path(__file__).parent.parent / "demo" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    # Fallback if demo file not found
    return HTMLResponse(content="""
    <html><body style="font-family:monospace;background:#0a0d1a;color:#e2e8f0;padding:40px">
    <h1>UndertriAI ⚖️</h1>
    <p>OpenEnv bail assessment environment is running.</p>
    <ul>
      <li><a href="/docs" style="color:#6366f1">Swagger Docs</a></li>
      <li><a href="/health" style="color:#6366f1">Health Check</a></li>
      <li><a href="/tools" style="color:#6366f1">Available Tools</a></li>
    </ul>
    </body></html>
    """)

@app.get("/health")
def health():
    return {"status": "ok", "env": "UndertriAI", "version": "1.0.0"}



@app.post("/reset")
def reset(
    stage: int = 1,
    session_id: str = None,
    seed: int = None,
    episode_id: str = None,
    adaptive: bool = False,
    auto_stage: bool = False,
):
    if session_id is None:
        session_id = str(uuid.uuid4())

    session = get_or_create_session(session_id)
    env = session.env
    session.adaptive = adaptive
    session.tools_used = []  # Reset tools tracking

    # Auto-stage: use tracker's suggestion
    effective_stage = stage
    if auto_stage:
        effective_stage = session.tracker.suggest_next_stage()

    env.set_stage(effective_stage)

    # Adaptive episode selection
    if adaptive and episode_id is None and seed is None:
        # Use adaptive selector instead of uniform random
        selected_ep = session.selector.select_episode(effective_stage)
        # Inject the selected episode directly into the environment
        env._episode = selected_ep
        env._episode_id = str(uuid.uuid4())
        env._step_count = 0
        env._flags = []
        env._retrieved_precedents = []
        env._action_history = []
        env._statutory_tool_called = False
        env._tools_called = set()
        obs = env._make_observation(action_result=None)
    else:
        obs = env.reset(stage=effective_stage, seed=seed, episode_id=episode_id)

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {},
    }


@app.post("/step")
def step(payload: dict):
    session_id = payload.get("session_id")
    action_data = payload.get("action", {})
    if not session_id or session_id not in _sessions:
        return JSONResponse(status_code=400, content={"error": "Invalid session_id. Call /reset first."})

    session = _sessions[session_id]
    env = session.env

    # Deserialize action by tool_name
    tool_name = action_data.get("tool_name", "")
    from ..models import (
        RequestDocumentAction, FlagInconsistencyAction,
        CrossReferencePrecedentAction, ComputeStatutoryEligibilityAction,
        AssessSuretyAction, ClassifyBailTypeAction,
        ReadSubmissionsAction, AssessFlightRiskAction,
        CheckCaseFactorsAction, ApplyProportionalityAction,
        PullCriminalHistoryAction, SubmitMemoAction,
    )
    ACTION_MAP = {
        "request_document":             RequestDocumentAction,
        "flag_inconsistency":           FlagInconsistencyAction,
        "cross_reference_precedent":    CrossReferencePrecedentAction,
        "compute_statutory_eligibility":ComputeStatutoryEligibilityAction,
        "assess_surety":                AssessSuretyAction,
        "classify_bail_type":           ClassifyBailTypeAction,
        "read_submissions":             ReadSubmissionsAction,
        "assess_flight_risk":           AssessFlightRiskAction,
        "check_case_factors":           CheckCaseFactorsAction,
        "apply_proportionality":        ApplyProportionalityAction,
        "pull_criminal_history":        PullCriminalHistoryAction,
        "submit_memo":                  SubmitMemoAction,
    }
    action_cls = ACTION_MAP.get(tool_name)
    if not action_cls:
        return JSONResponse(status_code=400, content={"error": f"Unknown tool: {tool_name}"})

    try:
        action = action_cls(**action_data)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    # Track tool usage for this session
    if tool_name != "submit_memo":
        session.tools_used.append(tool_name)

    result = env.step(action)

    # Theme 4: Update tracker after terminal action (reward available)
    if result.done and hasattr(result, "info") and isinstance(result.info, dict):
        reward_components = result.info
        episode = env._episode or {}
        session.tracker.update(
            episode=episode,
            reward_components=reward_components,
            tools_used=list(session.tools_used),
        )

        # Generate synthetic cases if agent mastered this domain
        if session.adaptive:
            crime_type = episode.get("crime_type", "")
            if crime_type and session.tracker.should_generate_synthetic(crime_type):
                variants = generate_variants(episode, n=3)
                if variants:
                    # Inject synthetic cases into the dataset
                    stage = episode.get("curriculum_stage", 1)
                    for v in variants:
                        v["curriculum_stage"] = stage
                        env.dataset._episodes.setdefault(stage, []).append(v)
                    session.synthetic_cases_generated += len(variants)
                    for v in variants:
                        logger.info(
                            f"Synthetic case generated: {v['case_id']} "
                            f"({v.get('perturbation_type', 'unknown')})"
                        )

    return {
        "session_id": session_id,
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info if hasattr(result, "info") else {},
    }


@app.get("/state")
def state(session_id: str):
    if session_id not in _sessions:
        return JSONResponse(status_code=400, content={"error": "Invalid session_id."})
    return _sessions[session_id].env.state


@app.get("/observation")
def observation(session_id: str):
    """OpenEnv spec alias for /state — returns current episode observation."""
    if session_id not in _sessions:
        return JSONResponse(status_code=400, content={"error": "Invalid session_id."})
    return _sessions[session_id].env.state


@app.get("/tools")
def list_tools():
    """Return available tool signatures (RFC 002 — tool discoverability)."""
    return {
        "tools": [
            {"name": "request_document",             "description": "Request a missing document (FIR, charge sheet, prior judgment)"},
            {"name": "flag_inconsistency",           "description": "Flag a legal inconsistency in the record"},
            {"name": "cross_reference_precedent",    "description": "Retrieve relevant SC/HC precedent"},
            {"name": "compute_statutory_eligibility","description": "Check BNSS 479 default bail eligibility"},
            {"name": "assess_surety",                "description": "Evaluate financial viability of proposed surety"},
            {"name": "classify_bail_type",           "description": "Classify bail type from grounds for/against"},
            {"name": "read_submissions",             "description": "Read and summarise prosecution or defence submissions"},
            {"name": "assess_flight_risk",           "description": "Systematic flight risk assessment with scoring matrix"},
            {"name": "check_case_factors",           "description": "Examine specific case factors (parity, evidence tampering, victim vulnerability)"},
            {"name": "apply_proportionality",        "description": "Apply BNSS 479 proportionality: custody vs. max sentence vs. trial timeline"},
            {"name": "pull_criminal_history",        "description": "Pull accused's prior criminal record, bail history, and conviction status"},
            {"name": "submit_memo",                  "description": "TERMINAL — Submit structured bail assessment memo"},
        ]
    }


# ------------------------------------------------------------------
# Theme 4: New API endpoints (additive — do not replace existing)
# ------------------------------------------------------------------

@app.get("/profile")
def get_profile(session_id: str):
    """Returns the current PerformanceTracker profile for the session."""
    if session_id not in _sessions:
        return JSONResponse(
            status_code=404,
            content={"error": f"Session '{session_id}' not found. Call /reset first."},
        )
    session = _sessions[session_id]
    return {
        "session_id": session_id,
        "profile": session.tracker.get_profile(),
        "adaptive_mode": session.adaptive,
        "synthetic_cases_generated": session.synthetic_cases_generated,
    }


@app.get("/adaptive_status")
def adaptive_status():
    """Returns global adaptive mode capabilities (not session-specific)."""
    return {
        "adaptive_available": True,
        "description": "Performance-aware episode selection and synthetic case generation",
        "promotion_thresholds": {
            "stage_1_to_2": {"min_reward": 0.65, "min_episodes": 20},
            "stage_2_to_3": {"min_reward": 0.55, "min_episodes": 50},
            "stage_3_to_4": {"min_reward": 0.50, "min_episodes": 20},
        },
        "perturbation_types": [
            "custody_escalation",
            "co_accused_conflict",
            "section_ambiguity",
            "evidence_reversal",
            "surety_complexity",
        ],
    }


# ------------------------------------------------------------------
# WebSocket endpoint (OpenEnv standard)
# ------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = get_or_create_session(session_id)
    env = session.env
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("command", "")

            if cmd == "reset":
                stage = msg.get("stage", 1)
                env.set_stage(stage)
                obs = env.reset(stage=stage)
                await websocket.send_text(json.dumps({
                    "type": "reset",
                    "observation": obs.model_dump(),
                    "reward": 0.0,
                    "done": False,
                }))

            elif cmd == "step":
                action_data = msg.get("action", {})
                tool_name = action_data.get("tool_name", "")
                from ..models import (
                    RequestDocumentAction, FlagInconsistencyAction,
                    CrossReferencePrecedentAction, ComputeStatutoryEligibilityAction,
                    AssessSuretyAction, ClassifyBailTypeAction,
                    ReadSubmissionsAction, AssessFlightRiskAction,
                    CheckCaseFactorsAction, ApplyProportionalityAction,
                    PullCriminalHistoryAction, SubmitMemoAction,
                )
                ACTION_MAP = {
                    "request_document":              RequestDocumentAction,
                    "flag_inconsistency":            FlagInconsistencyAction,
                    "cross_reference_precedent":     CrossReferencePrecedentAction,
                    "compute_statutory_eligibility": ComputeStatutoryEligibilityAction,
                    "assess_surety":                 AssessSuretyAction,
                    "classify_bail_type":            ClassifyBailTypeAction,
                    "read_submissions":              ReadSubmissionsAction,
                    "assess_flight_risk":            AssessFlightRiskAction,
                    "check_case_factors":            CheckCaseFactorsAction,
                    "apply_proportionality":         ApplyProportionalityAction,
                    "pull_criminal_history":         PullCriminalHistoryAction,
                    "submit_memo":                   SubmitMemoAction,
                }
                action_cls = ACTION_MAP.get(tool_name)
                if action_cls:
                    action = action_cls(**action_data)
                    result = env.step(action)
                    await websocket.send_text(json.dumps({
                        "type": "step",
                        "observation": result.observation.model_dump(),
                        "reward": result.reward,
                        "done": result.done,
                        "info": result.info if hasattr(result, "info") else {},
                    }))

            elif cmd == "state":
                await websocket.send_text(json.dumps({
                    "type": "state",
                    "state": env.state,
                }))

    except WebSocketDisconnect:
        if session_id in _sessions:
            del _sessions[session_id]


# ------------------------------------------------------------------
# Entry point for local dev
# ------------------------------------------------------------------

def main():
    """Main entry point for the UndertriAI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
