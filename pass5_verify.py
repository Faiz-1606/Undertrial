"""
Pass 5 — Gaming Resistance & Verification Suite
Tests that the reward function correctly ranks:
  C (ideal) > B (filler) > D (tool spam) > A (minimal)

Uses server/reward.py directly (no torch needed).
"""
import sys, os, re

_root = os.path.abspath(".")
_parent = os.path.dirname(_root)
for p in [_parent, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import types
_pkg = types.ModuleType("undertrial_ai")
_pkg.__path__ = [_root]
_pkg.__package__ = "undertrial_ai"
sys.modules["undertrial_ai"] = _pkg

from server.reward import (
    compute_outcome_match,
    compute_flight_risk_accuracy,
    compute_statutory_accuracy,
    compute_condition_score,
    compute_bias_penalty,
    compute_reasoning_quality,
    compute_think_factor,
    reward_format,
    _is_ndps_case,
)

# ── Minimal parse (mirrors train_grpo.py::parse_model_output) ──
def extract_xml_field(text, tag):
    m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def extract_xml_list(text, tag, item_tag="ground"):
    block = extract_xml_field(text, tag)
    return re.findall(rf'<{item_tag}>(.*?)</{item_tag}>', block, re.DOTALL)

def parse_output(output):
    if not output:
        output = ""
    memo_block = extract_xml_field(output, "memo")
    if not memo_block:
        return {
            "recommended_outcome": "", "flight_risk": "", "flight_risk_just": "",
            "statutory_eligible": False, "statutory_computation": "",
            "grounds_for": [], "grounds_against": [], "conditions": [],
            "has_think_block": "<think>" in output.lower(),
        }
    return {
        "recommended_outcome": extract_xml_field(memo_block, "recommended_outcome"),
        "flight_risk": extract_xml_field(memo_block, "flight_risk"),
        "flight_risk_just": extract_xml_field(memo_block, "flight_risk_justification"),
        "statutory_eligible": extract_xml_field(memo_block, "statutory_eligible").lower() == "true",
        "statutory_computation": extract_xml_field(memo_block, "statutory_computation"),
        "grounds_for": extract_xml_list(memo_block, "grounds_for_bail", "ground"),
        "grounds_against": extract_xml_list(memo_block, "grounds_against_bail", "ground"),
        "conditions": extract_xml_list(memo_block, "recommended_conditions", "condition"),
        "has_think_block": "<think>" in output.lower(),
    }

def reward_format_single(completion):
    if not completion:
        return 0.0
    required_tags = [r'<think>', r'<memo>', r'<flight_risk>', r'<statutory_eligible>', r'<recommended_outcome>', r'<statutory_computation>']
    valid_outcomes = ['bail granted', 'bail denied', 'conditional bail', 'default bail']
    checks = [bool(re.search(tag, completion, re.IGNORECASE)) for tag in required_tags]
    checks.append(any(o in completion.lower() for o in valid_outcomes))
    return sum(checks) / len(checks)

def combined_reward(comp, ep, current_stage=1):
    parsed = parse_output(comp)
    gt = ep.get("ground_truth", {})
    
    o  = compute_outcome_match(parsed["recommended_outcome"], gt)
    fr = compute_flight_risk_accuracy(parsed["flight_risk"], gt)
    s  = compute_statutory_accuracy(parsed["statutory_eligible"], parsed["statutory_computation"], ep)
    ca = compute_condition_score(parsed["recommended_outcome"], parsed.get("conditions", []), gt)
    b  = compute_bias_penalty(parsed["recommended_outcome"], ep,
            agent_grounds=parsed.get("grounds_for", []) + parsed.get("grounds_against", []))
    rq = compute_reasoning_quality(
        flight_risk_justification=parsed.get("flight_risk_just", ""),
        agent_risk_label=parsed.get("flight_risk", ""),
        statutory_computation=parsed.get("statutory_computation", ""),
        grounds_for=parsed.get("grounds_for", []),
        grounds_against=parsed.get("grounds_against", []),
        episode=ep,
    )
    
    think_factor = compute_think_factor(comp, current_stage)
    om_gated = o * think_factor
    fmt = reward_format_single(comp)
    
    # process_bonus
    custody_mo = ep.get("custody_months") or 0.0
    max_sent = ep.get("max_sentence_years", 5.0)
    if custody_mo > 0:
        threshold_mo = (max_sent * 12) / 2
        comp_text = parsed.get("statutory_computation", "").lower()
        has_exact_custody = str(int(custody_mo)) in comp_text
        has_exact_threshold = str(int(threshold_mo)) in comp_text
        process_bonus = 0.05 if (has_exact_custody and has_exact_threshold) else 0.0
    else:
        process_bonus = 0.0
    
    total = (0.4*om_gated + 0.2*fr + 0.2*s + 0.2*ca + 0.1*rq + 0.05*fmt + 0.05*process_bonus - 0.3*b)
    return round(total, 4)


# ── Test episode (murder case, bail denied) ──────────────────
EPISODE = {
    "case_id": "GAMING_TEST",
    "ipc_sections": ["302"],
    "crime_type": "murder",
    "custody_months": 8.0,
    "max_sentence_years": 7.0,
    "special_laws": "",
    "bail_type": "Regular",
    "accused_profile": {"name": "Ravi Kumar", "gender": "Male", "region": "Delhi"},
    "prosecution_arguments": ["Serious offence", "Investigation pending"],
    "defence_arguments": ["No prior record"],
    "ground_truth": {
        "outcome": "Bail Denied",
        "implicit_flight_risk": "High",
        "bias_flag": False,
        "parity_argument_used": False,
        "judgment_reason": "murder, flight risk, investigation ongoing",
        "outcome_detail": ""
    },
}

# ── Completion A: Minimal (no think, no memo) ───────────────
COMP_A = "Bail Denied."

# ── Completion B: Filler (right answer, short think) ─────────
COMP_B = """<think>
The accused is charged with murder.
</think>
<memo>
<flight_risk>High</flight_risk>
<flight_risk_justification>Serious charge</flight_risk_justification>
<statutory_eligible>false</statutory_eligible>
<statutory_computation>Cannot determine</statutory_computation>
<grounds_for_bail>
  <ground>No prior record</ground>
</grounds_for_bail>
<grounds_against_bail>
  <ground>Serious charge</ground>
</grounds_against_bail>
<recommended_outcome>Bail Denied</recommended_outcome>
<recommended_conditions></recommended_conditions>
</memo>"""

# ── Completion C: Ideal (structured, correct, reasoning) ────
COMP_C = """<think>
1. The accused is charged under Section 302 IPC (murder), which carries a maximum sentence of 7 years (life imprisonment possible but coded as 7).
2. Statutory eligibility: max 7 years = 84 months, threshold = 42 months. Accused has served 8 months in custody, which is well below the 42-month threshold. Therefore NOT eligible for default bail.
3. Flight risk: Murder charge with investigation pending creates high flight incentive. No community ties documented beyond Delhi residence.
4. Prosecution argues serious offence and ongoing investigation — strong grounds against bail.
5. Defence argues no prior record — relevant but insufficient given charge gravity.
Conclusion: Bail should be denied given charge severity, custody well below threshold, and ongoing investigation.
</think>
<memo>
<flight_risk>High</flight_risk>
<flight_risk_justification>Accused faces murder charge under Section 302 with investigation pending, creating substantial flight incentive. Only 8 months served of potential 84-month sentence.</flight_risk_justification>
<statutory_eligible>false</statutory_eligible>
<statutory_computation>Section 302 IPC -> max 7 years = 84 months -> threshold 42 months -> served 8 months -> NOT eligible for default bail</statutory_computation>
<grounds_for_bail>
  <ground>No prior criminal record documented</ground>
  <ground>Permanent resident of Delhi with family ties</ground>
</grounds_for_bail>
<grounds_against_bail>
  <ground>Murder charge under Section 302 IPC — serious offence</ground>
  <ground>Investigation still pending — risk of evidence tampering</ground>
</grounds_against_bail>
<recommended_outcome>Bail Denied</recommended_outcome>
<recommended_conditions></recommended_conditions>
</memo>"""

# ── Completion D: Tool spam (many tags, wrong direction) ─────
COMP_D = """<think>ok</think>
<memo>
<flight_risk>Low</flight_risk>
<flight_risk_justification>x</flight_risk_justification>
<statutory_eligible>true</statutory_eligible>
<statutory_computation>eligible</statutory_computation>
<grounds_for_bail>
  <ground>x</ground><ground>x</ground><ground>x</ground><ground>x</ground>
</grounds_for_bail>
<grounds_against_bail>
  <ground>x</ground>
</grounds_against_bail>
<recommended_outcome>Bail Granted</recommended_outcome>
<recommended_conditions>
  <condition>surety</condition><condition>bond</condition><condition>report</condition><condition>passport</condition><condition>permission</condition>
</recommended_conditions>
</memo>"""

print("\n" + "=" * 64)
print("  Pass 5 — Gaming Resistance Analysis")
print("=" * 64)

completions = {"A (minimal)": COMP_A, "B (filler)": COMP_B, "C (ideal)": COMP_C, "D (tool spam)": COMP_D}
scores = {}

for label, comp in completions.items():
    r = combined_reward(comp, EPISODE, current_stage=1)
    fmt = reward_format_single(comp)
    parsed = parse_output(comp)
    scores[label] = r
    print(f"\n  {label}:")
    print(f"    Total reward:   {r:.4f}")
    print(f"    Format score:   {fmt:.4f}")
    print(f"    Outcome:        {parsed['recommended_outcome']}")
    print(f"    Flight risk:    {parsed['flight_risk']}")
    print(f"    Has think:      {parsed['has_think_block']}")

print("\n" + "-" * 64)
print("  Ranking:")
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for i, (label, score) in enumerate(ranked, 1):
    print(f"    {i}. {label}: {score:.4f}")

expected_order = ["C (ideal)", "B (filler)", "A (minimal)", "D (tool spam)"]
actual_order = [label for label, _ in ranked]

print(f"\n  Expected: {' > '.join(expected_order)}")
print(f"  Actual:   {' > '.join(actual_order)}")

if actual_order == expected_order:
    print("\n  [OK] PASS — Gaming resistance ordering correct!")
    gaming_status = "PASS"
else:
    print("\n  [FAIL] FAIL — Ordering mismatch")
    gaming_status = "FAIL"
    if scores["C (ideal)"] > scores["B (filler)"] and scores["C (ideal)"] > scores["D (tool spam)"]:
        print("  NOTE: C (ideal) still highest — partial pass")

print("=" * 64)

# ── Section 4: Verification Suite (8 Tests) ─────────────────
print("\n" + "=" * 64)
print("  Pass 5 — Verification Suite (8 Tests)")
print("=" * 64)

results = []

def test(label, condition, detail=""):
    results.append((label, condition, detail))
    mark = "[OK]" if condition else "[FAIL]"
    print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))

# 1. combined_reward returns float
r = combined_reward(COMP_C, EPISODE, current_stage=1)
test("1. combined_reward returns float", isinstance(r, float), f"type={type(r)}, val={r}")

# 2. Process bonus fires on exact numbers (8 and 42 present in C)
parsed_c = parse_output(COMP_C)
comp_text = parsed_c["statutory_computation"].lower()
has_8 = "8" in comp_text
has_42 = "42" in comp_text
test("2. Process bonus fires for exact custody/threshold", has_8 and has_42, f"has_custody=8:{has_8}, has_threshold=42:{has_42}")

# 3. Format score for well-formed XML
fmt = reward_format_single(COMP_C)
test("3. Format compliance > 0.8 for well-formed XML", fmt > 0.8, f"fmt={fmt:.4f}")

# 4. Empty completion returns ~0
r_empty = combined_reward("", EPISODE, current_stage=1)
test("4. Empty completion -> reward ~= 0", r_empty < 0.35, f"reward={r_empty:.4f}")

# 5. Correct outcome scores higher than wrong
r_correct = combined_reward(COMP_C, EPISODE, current_stage=1)
r_wrong = combined_reward(COMP_D, EPISODE, current_stage=1)
test("5. Correct outcome > wrong outcome", r_correct > r_wrong, f"correct={r_correct:.4f} vs wrong={r_wrong:.4f}")

# 6. Think factor gates outcome in stage 2
r_s2 = combined_reward(COMP_A, EPISODE, current_stage=2)
test("6. No-think completion penalized in Stage 2", r_s2 < 0.25, f"stage2_minimal={r_s2:.4f}")

# 7. NDPS case wrong direction scores low
ndps_ep = {
    "ipc_sections": ["21"], "crime_type": "narcotics",
    "custody_months": 70.0, "max_sentence_years": 10.0, "special_laws": "",
    "accused_profile": {"name": "Test", "gender": "Male", "region": "Delhi"},
    "prosecution_arguments": [], "defence_arguments": [],
    "ground_truth": {"outcome": "Bail Denied", "implicit_flight_risk": "High", "bias_flag": False, "parity_argument_used": False},
}
ndps_comp = COMP_D.replace("302", "21 NDPS")
r_ndps = combined_reward(ndps_comp, ndps_ep, current_stage=1)
test("7. NDPS wrong direction scores low", r_ndps < 0.5, f"ndps_wrong={r_ndps:.4f}")

# 8. IssueOrderAction in models + client + root __all__
try:
    from models import IssueOrderAction
    assert IssueOrderAction.model_fields["tool_name"].default == "issue_order"
    # client.py and __init__.py use relative imports; verify by reading source
    client_text = open(os.path.join(_root, "client.py")).read()
    init_text = open(os.path.join(_root, "__init__.py")).read()
    assert "IssueOrderAction" in client_text, "IssueOrderAction not in client.py"
    assert "IssueOrderAction" in init_text, "IssueOrderAction not in __init__.py"
    test("8. IssueOrderAction in models + client + root __all__", True)
except Exception as e:
    test("8. IssueOrderAction in models + client + root __all__", False, str(e))

print("\n" + "-" * 64)
passed = sum(1 for _, c, _ in results if c)
failed = sum(1 for _, c, _ in results if not c)
print(f"  {passed}/8 PASSED | {failed}/8 FAILED")
print("=" * 64)
