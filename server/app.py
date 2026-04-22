"""
UndertriAI — FastAPI Server App
Wraps UndertriAIEnvironment as an OpenEnv-compatible HTTP + WebSocket server.
"""

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import uuid

from .undertrial_environment import UndertriAIEnvironment

# Session store: episode_id → environment instance
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


def get_or_create_env(session_id: str) -> UndertriAIEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = UndertriAIEnvironment(episodes_dir=EPISODES_DIR)
    return _sessions[session_id]


# ------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "UndertriAI ⚖️",
        "description": "OpenEnv-compliant RL environment for Indian bail decision support",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "tools": "/tools",
        "usage": {
            "reset": "POST /reset?stage=1",
            "step":  "POST /step  {session_id, action}",
            "state": "GET  /state?session_id=...",
        },
        "github": "https://github.com/Faiz-1606/Undertrial",
        "space":  "https://huggingface.co/spaces/Draken1606/undertrial-ai",
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": "UndertriAI", "version": "1.0.0"}



@app.post("/reset")
def reset(stage: int = 1, session_id: str = None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    env = get_or_create_env(session_id)
    env.set_stage(stage)
    obs = env.reset(stage=stage)
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

    env = _sessions[session_id]

    # Deserialize action by tool_name
    tool_name = action_data.get("tool_name", "")
    from ..models import (
        RequestDocumentAction, FlagInconsistencyAction,
        CrossReferencePrecedentAction, ComputeStatutoryEligibilityAction,
        AssessSuretyAction, ClassifyBailTypeAction, SubmitMemoAction,
    )
    ACTION_MAP = {
        "request_document":          RequestDocumentAction,
        "flag_inconsistency":        FlagInconsistencyAction,
        "cross_reference_precedent": CrossReferencePrecedentAction,
        "compute_statutory_eligibility": ComputeStatutoryEligibilityAction,
        "assess_surety":             AssessSuretyAction,
        "classify_bail_type":        ClassifyBailTypeAction,
        "submit_memo":               SubmitMemoAction,
    }
    action_cls = ACTION_MAP.get(tool_name)
    if not action_cls:
        return JSONResponse(status_code=400, content={"error": f"Unknown tool: {tool_name}"})

    try:
        action = action_cls(**action_data)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    result = env.step(action)
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
    return _sessions[session_id].state()


@app.get("/tools")
def list_tools():
    """Return available tool signatures (RFC 002 — tool discoverability)."""
    return {
        "tools": [
            {"name": "request_document",           "description": "Request a missing document"},
            {"name": "flag_inconsistency",         "description": "Flag legal inconsistency"},
            {"name": "cross_reference_precedent",  "description": "Retrieve relevant precedent"},
            {"name": "compute_statutory_eligibility", "description": "Check default bail eligibility"},
            {"name": "assess_surety",              "description": "Evaluate surety viability"},
            {"name": "classify_bail_type",         "description": "Classify bail type from grounds"},
            {"name": "submit_memo",                "description": "TERMINAL — Submit assessment memo"},
        ]
    }


# ------------------------------------------------------------------
# WebSocket endpoint (OpenEnv standard)
# ------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    env = get_or_create_env(session_id)
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
                    AssessSuretyAction, ClassifyBailTypeAction, SubmitMemoAction,
                )
                ACTION_MAP = {
                    "request_document": RequestDocumentAction,
                    "flag_inconsistency": FlagInconsistencyAction,
                    "cross_reference_precedent": CrossReferencePrecedentAction,
                    "compute_statutory_eligibility": ComputeStatutoryEligibilityAction,
                    "assess_surety": AssessSuretyAction,
                    "classify_bail_type": ClassifyBailTypeAction,
                    "submit_memo": SubmitMemoAction,
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
                    "state": env.state(),
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
