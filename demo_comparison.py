"""
UndertriAI — Before/After Demo Comparison Script

Demonstrates the environment using DEMO001 (Ramesh Kumar — IPC 420 cheating case).
Shows two simulated agent trajectories on the SAME case:
  1. Naive agent: skips tools, guesses wrong
  2. Skilled agent: uses tools properly, reaches correct conclusion

This script does NOT require a trained model — it simulates both agent
behaviors programmatically to show the reward difference.

Usage:
    python demo_comparison.py
"""

import sys
import os
import json

# Add parent of project root so relative imports within the package work
_project_root = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_project_root)
_pkg_name = os.path.basename(_project_root)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Import via package name (needed for relative imports in server/)
_env_mod = __import__(f"{_pkg_name}.server.undertrial_environment", fromlist=["UndertriAIEnvironment"])
UndertriAIEnvironment = _env_mod.UndertriAIEnvironment

_models_mod = __import__(f"{_pkg_name}.models", fromlist=[
    "ComputeStatutoryEligibilityAction", "AssessFlightRiskAction",
    "ReadSubmissionsAction", "CheckCaseFactorsAction", "SubmitMemoAction",
])
ComputeStatutoryEligibilityAction = _models_mod.ComputeStatutoryEligibilityAction
AssessFlightRiskAction = _models_mod.AssessFlightRiskAction
ReadSubmissionsAction = _models_mod.ReadSubmissionsAction
CheckCaseFactorsAction = _models_mod.CheckCaseFactorsAction
SubmitMemoAction = _models_mod.SubmitMemoAction


def run_demo():
    """Run before/after comparison on DEMO001."""
    print("=" * 65)
    print("  UndertriAI — Before vs After Training Demo")
    print("  Case: DEMO001 — Ramesh Kumar vs State of Delhi (IPC 420)")
    print("=" * 65)

    env = UndertriAIEnvironment()

    # ================================================================
    # NAIVE AGENT (simulates untrained model behavior)
    # ================================================================
    print("\n" + "─" * 65)
    print("  NAIVE AGENT (before training)")
    print("─" * 65)

    obs = env.reset(stage=1, seed=0)
    print(f"  Case: {obs.case_title}")
    print(f"  Crime: {obs.crime_type} | Sections: {obs.ipc_sections}")
    print(f"  Custody: {env._episode.get('custody_months')} months")

    # Naive agent: calls one tool minimally, then submits wrong answer
    print("\n  Step 1: Read submissions (both)")
    result = env.step(ReadSubmissionsAction(
        party="both",
    ))
    print(f"    → {result.observation.action_result[:80]}...")

    # Naive agent gets the outcome WRONG (denies bail when it should be granted)
    print("\n  Step 2: Submit memo (WRONG — denies bail)")
    result = env.step(SubmitMemoAction(
        flight_risk="High",
        flight_risk_justification="Accused may flee",
        statutory_eligible=False,
        statutory_computation="Unknown sections, cannot determine",
        grounds_for_bail=["None identified"],
        grounds_against_bail=["Serious charge"],
        recommended_outcome="Bail Denied",
        recommended_conditions=[],
    ))
    naive_reward = result.reward
    naive_info = result.info
    print(f"\n  NAIVE REWARD: {naive_reward:.4f}")
    print(f"    Outcome match:     {naive_info.get('outcome_match', 'N/A')}")
    print(f"    Flight risk acc:   {naive_info.get('flight_risk_accuracy', 'N/A')}")
    print(f"    Statutory acc:     {naive_info.get('statutory_accuracy', 'N/A')}")
    print(f"    Condition score:   {naive_info.get('condition_appropriateness', 'N/A')}")
    print(f"    Bias penalty:      {naive_info.get('bias_penalty', 'N/A')}")
    print(f"    Ground truth:      {naive_info.get('ground_truth_outcome', 'N/A')}")

    # ================================================================
    # SKILLED AGENT (simulates trained model behavior)
    # ================================================================
    print("\n" + "─" * 65)
    print("  SKILLED AGENT (after training)")
    print("─" * 65)

    obs = env.reset(stage=1, seed=0)  # Same case
    print(f"  Case: {obs.case_title}")

    # Skilled agent: uses multiple relevant tools
    print("\n  Step 1: Read submissions (both)")
    result = env.step(ReadSubmissionsAction(party="both"))
    print(f"    → {result.observation.action_result[:80]}...")

    print("\n  Step 2: Compute statutory eligibility")
    result = env.step(ComputeStatutoryEligibilityAction(
        sections_invoked=["420"],
        max_sentence_years=7.0,
        custody_months=8.0,
        special_law_applicable=False,
    ))
    print(f"    → {result.observation.action_result[:100]}...")

    print("\n  Step 3: Assess flight risk")
    result = env.step(AssessFlightRiskAction(
        severity_of_offence="moderate",
        roots_in_community="Permanent resident of Delhi, family with minor children",
        prior_absconding=False,
        passport_status="unknown",
    ))
    print(f"    → {result.observation.action_result[:100]}...")

    print("\n  Step 4: Check case factors")
    result = env.step(CheckCaseFactorsAction(
        factors_to_check=["nature_of_offence", "criminal_history", "evidence_tampering"],
    ))
    print(f"    → {result.observation.action_result[:100]}...")

    # Skilled agent: correct outcome with proper reasoning
    print("\n  Step 5: Submit memo (CORRECT — grants bail with conditions)")
    result = env.step(SubmitMemoAction(
        flight_risk="Low",
        flight_risk_justification=(
            "Accused is a permanent resident of Delhi with family ties including "
            "two minor children. No prior criminal record. IPC 420 is a moderate "
            "offence. No evidence of prior absconding. Prosecution has not cited "
            "any flight risk. Community roots are strong."
        ),
        statutory_eligible=False,
        statutory_computation=(
            "IPC Section 420: max sentence 7 years (84 months). "
            "BNSS 479 threshold = 42 months (50%). "
            "Time served = 8 months (9.5%). "
            "Threshold NOT yet met — not eligible for default bail. "
            "However, bail sought on merits, not statutory default."
        ),
        grounds_for_bail=[
            "No prior criminal record — first-time offender",
            "Permanent resident of Delhi with strong family ties",
            "Two minor children dependent on accused",
            "No flight risk identified by prosecution",
            "Offence is non-violent (cheating, not bodily harm)",
        ],
        grounds_against_bail=[
            "Investigation still pending per prosecution",
            "Alleged fraud of Rs. 50,000",
        ],
        recommended_outcome="Bail Granted",
        recommended_conditions=[
            "Personal bond of Rs. 25,000 with one local surety",
            "Weekly reporting to the concerned police station",
            "Surrender passport if held",
            "Not to leave Delhi without court permission",
            "Cooperate with ongoing investigation",
        ],
    ))
    skilled_reward = result.reward
    skilled_info = result.info
    print(f"\n  SKILLED REWARD: {skilled_reward:.4f}")
    print(f"    Outcome match:     {skilled_info.get('outcome_match', 'N/A')}")
    print(f"    Flight risk acc:   {skilled_info.get('flight_risk_accuracy', 'N/A')}")
    print(f"    Statutory acc:     {skilled_info.get('statutory_accuracy', 'N/A')}")
    print(f"    Condition score:   {skilled_info.get('condition_appropriateness', 'N/A')}")
    print(f"    Bias penalty:      {skilled_info.get('bias_penalty', 'N/A')}")
    print(f"    Ground truth:      {skilled_info.get('ground_truth_outcome', 'N/A')}")

    # ================================================================
    # COMPARISON
    # ================================================================
    print("\n" + "═" * 65)
    print("  COMPARISON SUMMARY")
    print("═" * 65)
    delta = skilled_reward - naive_reward
    print(f"  Naive agent reward:   {naive_reward:.4f}")
    print(f"  Skilled agent reward: {skilled_reward:.4f}")
    print(f"  Improvement:          {delta:+.4f} ({delta/max(0.01, abs(naive_reward))*100:+.0f}%)")
    print()

    # Component-by-component comparison
    components = [
        ("Outcome Match", "outcome_match"),
        ("Flight Risk",   "flight_risk_accuracy"),
        ("Statutory",     "statutory_accuracy"),
        ("Conditions",    "condition_appropriateness"),
        ("Bias Penalty",  "bias_penalty"),
    ]
    print(f"  {'Component':<20} {'Naive':>8} {'Skilled':>8} {'Delta':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    for name, key in components:
        n = naive_info.get(key, 0)
        s = skilled_info.get(key, 0)
        d = s - n
        sign = "+" if d >= 0 else ""
        print(f"  {name:<20} {n:>8.3f} {s:>8.3f} {sign}{d:>7.3f}")

    print()
    print(f"  Ground truth: {skilled_info.get('ground_truth_outcome', '?')}")
    print(f"  Naive agent:  Bail Denied (WRONG)")
    print(f"  Skilled agent: Bail Granted (CORRECT)")
    print("═" * 65)

    return {
        "naive_reward": naive_reward,
        "skilled_reward": skilled_reward,
        "delta": delta,
    }


if __name__ == "__main__":
    results = run_demo()
