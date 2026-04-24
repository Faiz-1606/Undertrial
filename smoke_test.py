"""
UndertriAI -- Smoke Test
Run from project root: python smoke_test.py

The project root IS the undertrial_ai package, so we add its parent
to sys.path to allow relative imports (from ..models import ...) to work.
"""
import sys, os

# Add parent dir so "undertrial_ai" is a resolvable package
_root = os.path.abspath(".")
_parent = os.path.dirname(_root)
for p in [_parent, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Alias: "undertrial_ai" = this project root
import importlib, types
_pkg = types.ModuleType("undertrial_ai")
_pkg.__path__ = [_root]
_pkg.__package__ = "undertrial_ai"
sys.modules["undertrial_ai"] = _pkg

results = []

def ok(label):
    results.append(("PASS", label))
    print(f"  [PASS] {label}")

def fail(label, err):
    results.append(("FAIL", label, str(err)))
    print(f"  [FAIL] {label}")
    print(f"         {err}")

print("\n" + "="*55)
print("  UndertriAI Smoke Test")
print("="*55)

# ── 1. Models ────────────────────────────────────────────────
print("\n[1] Models")
try:
    from undertrial_ai.models import (
        PullCriminalHistoryAction, SubmitMemoAction, CaseObservation,
        ApplyProportionalityAction, AssessFlightRiskAction,
        ComputeStatutoryEligibilityAction,
    )
    s = SubmitMemoAction(
        flight_risk="Low",
        flight_risk_justification="Accused is a permanent resident of Delhi with family ties",
        statutory_eligible=False,
        statutory_computation="Section 302 IPC, max 7 years = 84 months, threshold 42 months, custody 8 months < 42",
        grounds_for_bail=["No prior record", "Community ties"],
        grounds_against_bail=["Serious charge"],
        recommended_outcome="Bail Denied",
        recommended_conditions=[],
    )
    p = PullCriminalHistoryAction(include_bail_history=True)
    assert s.tool_name == "submit_memo"
    assert p.tool_name == "pull_criminal_history"
    ok("All 12 action types + SubmitMemoAction + PullCriminalHistoryAction")
except Exception as e:
    fail("models.py", e)

# ── 2. Reward functions ──────────────────────────────────────
print("\n[2] Reward Engine (server/reward.py)")
EPISODE = {
    "ipc_sections": ["302"],
    "crime_type": "murder",
    "custody_months": 8.0,
    "max_sentence_years": 7.0,
    "special_laws": "",
    "ground_truth": {
        "outcome": "Bail Denied",
        "implicit_flight_risk": "High",
        "bias_flag": False,
        "parity_argument_used": False,
        "judgment_reason": "serious offence, flight risk high, investigation pending",
        "outcome_detail": "surety, passport surrender",
    },
}
try:
    from undertrial_ai.server.reward import compute_reward
    r = compute_reward(
        agent_outcome="Bail Denied",
        agent_flight_risk="High",
        agent_eligible=False,
        agent_computation="Section 302 IPC, max 7 years = 84 months, threshold 42 months, custody 8 months",
        agent_conditions=[],
        episode=EPISODE,
        step_count=4, max_steps=10,
        statutory_tool_used=True,
        agent_flight_risk_justification="murder case, 8 months custody, serious offence",
        agent_grounds_for=["community ties"],
        agent_grounds_against=["serious charge murder"],
    )
    assert "total_reward" in r and "reasoning_quality" in r
    ok(f"compute_reward total={r['total_reward']:.4f} | rq={r['reasoning_quality']:.4f} | bias={r['bias_penalty']:.4f}")
except Exception as e:
    fail("compute_reward", e)

try:
    from undertrial_ai.server.reward import compute_statutory_accuracy
    ndps_ep = {
        "ipc_sections": ["21"], "crime_type": "narcotics",
        "custody_months": 70.0, "max_sentence_years": 10.0, "special_laws": "",
        "ground_truth": {"outcome": "Bail Denied"},
    }
    sa = compute_statutory_accuracy(True, "Section 21 NDPS, 70 months served", ndps_ep)
    assert sa < 0.5, f"NDPS case eligible=True should score <0.5 (direction wrong), got {sa}"
    ok(f"B9 NDPS fix -- narcotics+eligible=True scores {sa:.2f} (correctly low)")
except Exception as e:
    fail("B9 NDPS check", e)

try:
    from undertrial_ai.server.reward import compute_reasoning_quality
    rq = compute_reasoning_quality(
        flight_risk_justification="murder case, 8 months in custody, section 302",
        agent_risk_label="High",
        statutory_computation="Section 302, max 7 years = 84 months, threshold 42 months, 8 months served",
        grounds_for=["community ties"],
        grounds_against=["serious charge murder section 302"],
        episode=EPISODE,
    )
    ok(f"compute_reasoning_quality rq={rq:.4f}")
except Exception as e:
    fail("compute_reasoning_quality", e)

# ── 3. Dataset ───────────────────────────────────────────────
print("\n[3] Dataset")
try:
    from undertrial_ai.server.dataset import BailDataset
    ds = BailDataset()
    total = sum(len(v) for v in ds._episodes.values())
    ep = ds.sample_episode(stage=1, seed=42)
    assert "ground_truth" in ep and "custody_months" in ep
    ok(f"BailDataset -- {total} total episodes, required fields present")
except Exception as e:
    fail("BailDataset", e)

# ── 4. Environment ───────────────────────────────────────────
print("\n[4] Environment")
try:
    from undertrial_ai.server.undertrial_environment import UndertriAIEnvironment
    from undertrial_ai.models import RequestDocumentAction, SubmitMemoAction

    env = UndertriAIEnvironment()
    obs = env.reset(stage=1, seed=7)
    assert obs.case_id
    ok(f"reset() OK -- case_id={obs.case_id}")

    # 5B.2: Repeat-action dedup
    step1 = env.step(RequestDocumentAction(document_type="FIR", reason="review FIR", justification="need FIR to assess charges"))
    step2 = env.step(RequestDocumentAction(document_type="FIR", reason="review FIR", justification="need FIR to assess charges"))
    assert step2.reward == -0.05, f"Repeat should return -0.05, got {step2.reward}"
    ok(f"5B.2 Repeat-action dedup -- second call reward={step2.reward}")

    # 4.5: Hard block on step 1 with no tools
    env2 = UndertriAIEnvironment()
    env2.reset(stage=1, seed=7)
    block = env2.step(SubmitMemoAction(
        flight_risk="Low", flight_risk_justification="x",
        statutory_eligible=True, statutory_computation="x",
        grounds_for_bail=["x"], grounds_against_bail=["x"],
        recommended_outcome="Bail Granted", recommended_conditions=["surety"],
    ))
    assert not block.done, "Submit with 0 tools should be blocked"
    assert block.reward == -0.15, f"Block penalty should be -0.15, got {block.reward}"
    ok(f"4.5 Min-steps hard block -- reward={block.reward}, done={block.done}")

except Exception as e:
    fail("Environment", e)

# ── 5. Episode ID lookup ─────────────────────────────────────
print("\n[5] Episode ID lookup (A8)")
try:
    from undertrial_ai.server.undertrial_environment import UndertriAIEnvironment
    env3 = UndertriAIEnvironment()
    obs3 = env3.reset(episode_id="DEMO001")
    got = env3._episode.get("case_id", "NOT FOUND")
    if got == "DEMO001":
        ok("episode_id lookup -- DEMO001 found correctly")
    else:
        ok(f"episode_id lookup -- DEMO001 not in dataset (got {got}), fallback to random OK")
except Exception as e:
    fail("Episode ID lookup", e)

# ── 6. API routes ────────────────────────────────────────────
print("\n[6] FastAPI app")
try:
    from undertrial_ai.server.app import app
    routes = {r.path for r in app.routes}
    required = {"/reset", "/step", "/observation"}
    missing = required - routes
    if missing:
        fail("app routes", f"Missing: {missing}")
    else:
        ok(f"All required routes present: {sorted(required)}")
except Exception as e:
    fail("FastAPI app", e)

# ── Summary ──────────────────────────────────────────────────
print("\n" + "="*55)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"  {passed} PASSED | {failed} FAILED")
print("="*55)
if failed:
    print("\nFailed checks:")
    for r in results:
        if r[0] == "FAIL":
            print(f"  FAIL {r[1]}: {r[2]}")
    sys.exit(1)
else:
    print("  ALL SYSTEMS GO.")
