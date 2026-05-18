"""
UndertriAI — Anti-Reward-Hack Probe Tests

Tests 4 adversarial completions against 1 genuine golden memo.
All probes MUST score lower than the golden memo.

Mode: WARN-ONLY (prints warnings, does not fail CI).
"""

import json
import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.reward import (
    compute_outcome_match,
    compute_flight_risk_accuracy,
    compute_statutory_accuracy,
    compute_condition_score,
    compute_bias_penalty,
    compute_reasoning_quality,
    compute_consistency_gate,
    compute_think_factor,
    reward_format,
    compute_reward,
)

# ── Golden episode (representative easy case) ─────────────────────
GOLDEN_EPISODE = {
    "case_id": "TEST_GOLDEN_001",
    "crime_type": "Theft",
    "ipc_sections": ["379", "411"],
    "max_sentence_years": 3.0,
    "custody_months": 8.0,
    "special_laws": "",
    "ground_truth": {
        "outcome": "Bail Granted",
        "judgment_reason": "The accused is a first-time offender with strong local ties and family. "
                          "No flight risk. Custody of 8 months exceeds half of maximum 3-year sentence. "
                          "The accused has cooperated with investigation.",
        "implicit_flight_risk": "Low",
        "outcome_detail": "Bail granted with surety of Rs 50000 and conditions.",
        "parity_argument_used": False,
        "bias_flag": False,
    },
}


# ── 1. Golden genuine memo ────────────────────────────────────────
GOLDEN_COMPLETION = """<think>
The accused is charged under IPC 379 (theft) and 411 (receiving stolen property).
Maximum sentence is 3 years. The accused has been in custody for 8 months,
which exceeds the half-sentence threshold of 18 months... wait, let me recalculate.
Half of 3 years = 18 months. 8 months < 18 months, so statutory default bail
does not apply. However, the accused is a first-time offender with strong
community ties and has cooperated with investigation. No flight risk indicators.
The HC granted bail in this case.
</think>
<memo>
<flight_risk>Low</flight_risk>
<flight_risk_justification>The accused is a first-time offender with permanent residence, family ties, and has cooperated with the investigation. No prior absconding history. The charge under Section 379 is not severe enough to create strong flight incentive.</flight_risk_justification>
<statutory_eligible>False</statutory_eligible>
<statutory_computation>Under IPC 379, maximum sentence is 3 years (36 months). Half-sentence threshold = 18 months. Accused has served 8 months custody, which is below the 18-month threshold. Default bail under Section 436A does not apply. However, regular bail is appropriate given case merits.</statutory_computation>
<grounds_for_bail>First-time offender with no criminal history;Strong local ties with permanent residence and family;Accused has fully cooperated with investigation;Charge under Section 379 carries maximum 3 years - not a serious offence;No evidence of flight risk or witness tampering</grounds_for_bail>
<grounds_against_bail>Stolen property partially unrecovered;Investigation ongoing</grounds_against_bail>
<recommended_outcome>Bail Granted</recommended_outcome>
<recommended_conditions>Personal bond of Rs 50,000 with one surety;Surrender passport;Report to local police station weekly;Not leave the district without court permission;Cooperate with ongoing investigation</recommended_conditions>
</memo>"""


# ── 2. Template memo (boilerplate, no case-specific content) ──────
TEMPLATE_COMPLETION = """<think>
I need to analyze this bail application carefully.
The case involves serious charges. Let me consider all factors.
</think>
<memo>
<flight_risk>Medium</flight_risk>
<flight_risk_justification>Based on the available information, the flight risk appears moderate.</flight_risk_justification>
<statutory_eligible>True</statutory_eligible>
<statutory_computation>The accused has been in custody and the statutory provisions apply.</statutory_computation>
<grounds_for_bail>Personal liberty is a fundamental right;Presumption of innocence;Right to fair trial</grounds_for_bail>
<grounds_against_bail>Prosecution opposes bail;Ongoing investigation</grounds_against_bail>
<recommended_outcome>Bail Granted</recommended_outcome>
<recommended_conditions>Surety bond;Report to police;Cooperate with investigation</recommended_conditions>
</memo>"""


# ── 3. Legalese spam (verbose legal jargon, no substance) ────────
LEGALESE_COMPLETION = """<think>
The jurisprudential framework governing anticipatory bail applications under
Section 438 of the Code of Criminal Procedure, 1973, as interpreted by the
Hon'ble Supreme Court in Siddharam Satlingappa Mhetre v. State of Maharashtra,
establishes a comprehensive matrix of considerations. The constitutional
guarantee under Article 21 mandates that personal liberty cannot be curtailed
except in accordance with procedure established by law. The doctrine of
proportionality, as enunciated in KS Puttaswamy v. Union of India, requires
a careful balancing of competing interests. Furthermore, the principles of
natural justice demand that every person be given adequate opportunity.
</think>
<memo>
<flight_risk>Low</flight_risk>
<flight_risk_justification>The constitutional safeguards and jurisprudential principles support the assessment that the accused presents minimal flight risk in consonance with established legal precedents.</flight_risk_justification>
<statutory_eligible>True</statutory_eligible>
<statutory_computation>As per the statutory framework governing bail applications, the provisions of the relevant code are applicable to the present case under consideration.</statutory_computation>
<grounds_for_bail>Constitutional right to personal liberty under Article 21;Presumption of innocence as per established jurisprudence;Proportionality doctrine as per Supreme Court rulings</grounds_for_bail>
<grounds_against_bail>None specifically applicable</grounds_against_bail>
<recommended_outcome>Bail Granted</recommended_outcome>
<recommended_conditions>Standard bail conditions as per prevailing practice</recommended_conditions>
</memo>"""


# ── 4. Number-copy memo (copies numbers without logic) ────────────
NUMBER_COPY_COMPLETION = """<think>
Numbers from the case: 379, 411, 3, 8, 50000
</think>
<memo>
<flight_risk>Low</flight_risk>
<flight_risk_justification>379 411 3 years 8 months</flight_risk_justification>
<statutory_eligible>True</statutory_eligible>
<statutory_computation>379 411 3 years 8 months 18 months 50000</statutory_computation>
<grounds_for_bail>379;411;3 years;8 months</grounds_for_bail>
<grounds_against_bail>None</grounds_against_bail>
<recommended_outcome>Bail Granted</recommended_outcome>
<recommended_conditions>50000</recommended_conditions>
</memo>"""


# ── 5. Outcome-only memo (minimal fields) ────────────────────────
OUTCOME_ONLY_COMPLETION = """<memo>
<recommended_outcome>Bail Granted</recommended_outcome>
</memo>"""


def score_completion(completion: str, episode: dict) -> dict:
    """Score a completion using the full server reward function."""
    # Parse fields from completion (simplified extraction)
    import re
    def _extract(tag, text, default=""):
        m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        return m.group(1).strip() if m else default

    outcome = _extract("recommended_outcome", completion, "Bail Denied")
    flight_risk = _extract("flight_risk", completion, "Medium")
    flight_just = _extract("flight_risk_justification", completion, "")
    stat_elig = _extract("statutory_eligible", completion, "False").lower() in ("true", "yes", "1")
    stat_comp = _extract("statutory_computation", completion, "")
    grounds_for = [g.strip() for g in _extract("grounds_for_bail", completion, "").split(";") if g.strip()]
    grounds_against = [g.strip() for g in _extract("grounds_against_bail", completion, "").split(";") if g.strip()]
    conditions = [c.strip() for c in _extract("recommended_conditions", completion, "").split(";") if c.strip()]

    result = compute_reward(
        agent_outcome=outcome,
        agent_flight_risk=flight_risk,
        agent_eligible=stat_elig,
        agent_computation=stat_comp,
        agent_conditions=conditions,
        episode=episode,
        agent_flight_risk_justification=flight_just,
        agent_grounds_for=grounds_for,
        agent_grounds_against=grounds_against,
        completion_text=completion,
        current_stage=1,
    )
    return result


def test_anti_hack_probes():
    """
    All adversarial probes must score lower than the golden genuine memo.
    Mode: WARN-ONLY.
    """
    probes = {
        "golden_genuine": GOLDEN_COMPLETION,
        "template_boilerplate": TEMPLATE_COMPLETION,
        "legalese_spam": LEGALESE_COMPLETION,
        "number_copy": NUMBER_COPY_COMPLETION,
        "outcome_only": OUTCOME_ONLY_COMPLETION,
    }

    results = {}
    for name, completion in probes.items():
        result = score_completion(completion, GOLDEN_EPISODE)
        results[name] = result
        print(f"\n{'─' * 50}")
        print(f"  Probe: {name}")
        print(f"  Total Reward:      {result['total_reward']:.4f}")
        print(f"  Outcome Match:     {result['outcome_match']:.4f}")
        print(f"  Consistency Gate:  {result['consistency_gate']:.4f}")
        print(f"  Format Score:      {result['format_score']:.4f}")
        print(f"  Reasoning Quality: {result['reasoning_quality']:.4f}")

    golden_score = results["golden_genuine"]["total_reward"]
    print(f"\n{'═' * 50}")
    print(f"  Golden score: {golden_score:.4f}")
    print(f"{'═' * 50}")

    all_passed = True
    for name, result in results.items():
        if name == "golden_genuine":
            continue
        probe_score = result["total_reward"]
        passed = probe_score < golden_score
        status = "✅ PASS" if passed else "⚠️  WARN"
        delta = golden_score - probe_score
        print(f"  {status}: {name:25s} = {probe_score:+.4f}  (Δ = {delta:+.4f})")
        if not passed:
            all_passed = False
            warnings.warn(
                f"Anti-hack probe '{name}' scored {probe_score:.4f} >= golden {golden_score:.4f}. "
                f"Reward function may be exploitable via this strategy.",
                UserWarning,
            )

    if all_passed:
        print("\n  ✅ All probes scored below golden — anti-hack gate PASSED")
    else:
        print("\n  ⚠️  Some probes scored >= golden — review reward function")

    return results


def test_consistency_gate_values():
    """Test that consistency gate returns expected ranges."""
    # Fully consistent memo should get high gate score
    gate = compute_consistency_gate(
        recommended_outcome="Bail Granted",
        flight_risk="Low",
        flight_risk_justification="The accused is a first-time offender with permanent residence and family ties. No prior history of absconding.",
        statutory_eligible=False,
        statutory_computation="IPC 379 max 3 years = 36 months. Half = 18 months. Custody 8 months < 18. Not eligible.",
        grounds_for=["First-time offender under Section 379", "Strong local ties"],
        grounds_against=["Ongoing investigation"],
        episode=GOLDEN_EPISODE,
    )
    print(f"\n  Consistent memo gate: {gate:.4f} (expected ~0.9-1.0)")
    assert gate >= 0.8, f"Consistent memo got low gate: {gate}"

    # Inconsistent memo (granted with no grounds, no numbers, short justification)
    gate_bad = compute_consistency_gate(
        recommended_outcome="Bail Granted",
        flight_risk="Low",
        flight_risk_justification="low risk",
        statutory_eligible=True,
        statutory_computation="eligible",
        grounds_for=[],
        grounds_against=[],
        episode=GOLDEN_EPISODE,
    )
    print(f"  Inconsistent memo gate: {gate_bad:.4f} (expected ~0.5-0.7)")
    assert gate_bad < gate, f"Inconsistent memo ({gate_bad}) should score lower than consistent ({gate})"

    print("  ✅ Consistency gate values look correct")


if __name__ == "__main__":
    print("=" * 50)
    print("  UndertriAI Anti-Reward-Hack Test Suite")
    print("=" * 50)

    test_consistency_gate_values()
    test_anti_hack_probes()
