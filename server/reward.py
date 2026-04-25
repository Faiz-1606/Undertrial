"""
UndertriAI — Reward Engine
Computes the multi-component weighted reward + bias penalty.

R = 0.4*outcome_gated + 0.2*flight_risk + 0.2*statutory + 0.2*conditions
  + 0.1*reasoning_quality + 0.05*efficiency + 0.05*format
  + 0.05*process_bonus - 0.3*bias

All components are deterministic and rule-based — no LLM-as-a-judge.
"""

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared helper: NDPS detection (canonical definition — import this elsewhere)
# ---------------------------------------------------------------------------

def _is_ndps_case(episode: dict) -> bool:
    """
    Detect narcotics cases even when special_laws field is empty.
    Checks ipc_sections and crime_type for NDPS indicators.

    This is the SINGLE canonical definition — import from server.reward
    in undertrial_environment.py and training/train_grpo.py.
    """
    sections = " ".join(str(s) for s in episode.get("ipc_sections", [])).lower()
    crime = str(episode.get("crime_type", "")).lower()
    narcotics_indicators = [
        "ndps", "narcotic", "drug", "psychotropic",
        "20(b)", "22(b)", "27a", "section 37",
    ]
    return any(ind in sections or ind in crime for ind in narcotics_indicators)


# ---------------------------------------------------------------------------
# 1. Outcome Match  (40%)
# ---------------------------------------------------------------------------

def compute_outcome_match(agent_outcome: str, ground_truth: Dict[str, Any]) -> float:
    """
    Checks if the agent's final recommendation matches the High Court decision.

    Scoring:
        1.0 — Exact string match
        0.9 — "Bail Conditional" vs "Bail Granted" (conditional IS bail)
        0.8 — Directionally correct but loose string
        0.0 — Wrong direction (granted vs. denied, or vice versa)
    """
    gt = ground_truth["outcome"]
    agent_norm = agent_outcome.strip().lower()
    gt_norm    = gt.strip().lower()

    if agent_norm == gt_norm:
        return 1.0

    # Conditional bail counts almost as well as full bail
    if "conditional" in agent_norm and "grant" in gt_norm:
        return 0.9
    if "grant" in agent_norm and "conditional" in gt_norm:
        return 0.9

    agent_granted = "grant" in agent_norm or "conditional" in agent_norm
    gt_granted    = "grant" in gt_norm    or "conditional" in gt_norm

    return 0.8 if (agent_granted == gt_granted) else 0.0


# ---------------------------------------------------------------------------
# 2. Flight Risk Accuracy  (20%)
# ---------------------------------------------------------------------------

# Keywords that appear in High Court judgments when the court found low/high risk.
# Derived from real HC bail judgment language.
FLIGHT_RISK_KEYWORDS = {
    "Low": [
        "not a flight risk", "local ties", "permanent resident", "cooperative",
        "surrendered", "family", "roots", "community", "gainfully employed",
        "no prior", "sureties available", "settled", "willing to abide",
    ],
    "High": [
        "abscond", "tamper", "influential", "intimidat", "repeat offend",
        "hardcore criminal", "multiple cases", "organized crime", "nexus",
        "powerful", "non-cooperative", "evade", "history of",
        "previous bail violated", "custodial",
    ],
}


def compute_flight_risk_accuracy(agent_risk: str, ground_truth: Dict[str, Any]) -> float:
    """
    Scores the agent's flight risk classification.

    If the episode has no labeled implicit_flight_risk, returns 0.5 (neutral — no signal).
    This prevents the agent from gaming a default "Medium" label.

    Scoring:
        1.0 — Exact match to GT implicit risk
        0.5 — One tier off (Low↔Medium or Medium↔High)
        0.0 — Two tiers off (Low vs. High)
        + up to 0.2 keyword bonus from judgment text
    """
    gt_risk = ground_truth.get("implicit_flight_risk")

    # No label in the dataset → return neutral, no signal either way
    if not gt_risk:
        return 0.5

    risk_scores = {"Low": 0, "Medium": 1, "High": 2}
    agent_score = risk_scores.get(agent_risk, 1)
    gt_score    = risk_scores.get(gt_risk, 1)
    diff        = abs(agent_score - gt_score)

    base = 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)

    # Keyword bonus: does the HC judgment text support the agent's classification?
    reason   = ground_truth.get("judgment_reason", "").lower()
    keywords = FLIGHT_RISK_KEYWORDS.get(agent_risk, [])
    matches  = sum(1 for kw in keywords if kw in reason)
    keyword_bonus = min(0.2, matches * 0.05)

    return min(1.0, base + keyword_bonus)


# ---------------------------------------------------------------------------
# 3. Statutory Accuracy  (20%)
# ---------------------------------------------------------------------------

# Known IPC → BNSS (2023) section equivalents for cross-schema validation
IPC_TO_BNSS = {
    "302": "103", "307": "109", "376": "64",  "304B": "80",  "395": "310",
    "392": "309", "420": "318", "498A": "85", "406":  "316", "465": "336",
    "323": "115", "354": "74",  "120B": "61", "506":  "351", "121": "147",
    "379": "303", "324": "117", "354A": "75",
}

TIME_WORDS = ["year", "month", "sentence", "maximum", "custody", "served", "half", "threshold"]


def compute_statutory_accuracy(
    agent_eligible: bool,
    agent_computation: str,
    episode: Dict[str, Any],
) -> float:
    """
    Scores whether the agent correctly assessed default bail eligibility under IPC/BNSS.

    Ground truth eligibility: accused has served >= 50% of maximum sentence
    and no special law (NDPS/UAPA/PMLA) applies.

    Scoring breakdown:
        0.40 — Got eligibility direction correct vs. actual math
        0.20 — Directional alignment even if math differs (lenient fallback)
        0.30 — Mentioned relevant IPC/BNSS sections in computation text
        0.30 — Showed actual numeric computation (numbers + time-related language)
    """
    sections     = episode.get("ipc_sections", [])
    max_sent     = episode.get("max_sentence_years", 5.0)
    custody_mo   = episode.get("custody_months") or 0.0
    special_laws = episode.get("special_laws", "").strip()
    gt_outcome   = episode["ground_truth"]["outcome"]
    comp         = agent_computation.lower()

    # C5 fix: if custody_months is 0 or null, we cannot compute eligibility.
    # Return 0.5 neutral — do NOT penalise the agent based on a missing field.
    if custody_mo == 0.0:
        # Still score section citations and numeric computation quality
        score = 0.0
        if sections:
            hits = sum(
                1 for sec in sections
                if sec.strip() in comp or IPC_TO_BNSS.get(sec.strip(), "") in comp
            )
            score += 0.3 * min(1.0, hits / len(sections))
        has_numbers  = bool(re.search(r'\d+', comp))
        has_time_ref = any(w in comp for w in TIME_WORDS)
        if has_numbers and has_time_ref:
            score += 0.3
        elif has_numbers or has_time_ref:
            score += 0.15
        return min(0.5, score)  # Cap at 0.5 — cannot verify eligibility direction

    score = 0.0

    # B9: Infer special law from crime_type when the special_laws field is empty.
    # All 134 narcotics episodes have special_laws="" but should be NDPS-restricted.
    CRIME_TYPE_SPECIAL_LAWS = [
        "narcotics", "ndps", "pocso", "uapa", "pmla",
        "terrorism", "organised crime", "money laundering",
    ]
    crime_type_lower = episode.get("crime_type", "").lower()
    if not special_laws and any(t in crime_type_lower for t in CRIME_TYPE_SPECIAL_LAWS):
        special_laws = "INFERRED"  # Treat as special-law-restricted for eligibility

    # ── B9: NDPS-specific statutory scoring ──────────────────────────────
    # NDPS Section 37 twin conditions override standard threshold logic.
    # Reward the agent for recognizing NDPS applies, not for arithmetic.
    if _is_ndps_case(episode):
        gt_granted = "grant" in gt_outcome.lower()
        direction_correct = (agent_eligible == gt_granted)
        ndps_recognized = any(
            t in comp for t in ["section 37", "twin condition", "ndps", "37(1)(b)"]
        )
        if ndps_recognized and direction_correct:
            return 1.0
        elif direction_correct:
            return 0.5
        else:
            return 0.0

    # ── Standard IPC/BNSS statutory scoring ──────────────────────────────
    # D4: Detect unreliable custody_months=6.0 default on serious crimes.
    # 74% of episodes have custody_months=6.0 which may be a dataset default.
    # Cap score at 0.60 to avoid rewarding threshold arithmetic on unreliable data.
    custody_unreliable = (custody_mo == 6.0 and max_sent > 3.0)

    # Compute ground-truth eligibility for cases with known custody duration
    half_sent_months = (max_sent * 12) / 2
    truly_eligible   = (custody_mo >= half_sent_months) and not special_laws

    # 40%: eligibility direction vs. mathematical truth
    if agent_eligible == truly_eligible:
        score += 0.4
    elif (agent_eligible and "grant" in gt_outcome.lower()) or \
         (not agent_eligible and "deni" in gt_outcome.lower()):
        # Directionally aligned with HC outcome even if eligibility math is off
        score += 0.2

    # 30%: cited the right sections (IPC or BNSS equivalent accepted)
    if sections:
        hits = 0
        for sec in sections:
            sec_clean = sec.strip()
            if sec_clean in comp or sec_clean.lower() in comp:
                hits += 1
            bnss_eq = IPC_TO_BNSS.get(sec_clean, "")
            if bnss_eq and bnss_eq in comp:
                hits += 1
        score += 0.3 * min(1.0, hits / len(sections))

    # 30%: showed numeric math AND referenced time/sentence language.
    # B3: Gate on direction: full credit only when eligibility direction is correct.
    # Wrong-direction agents get partial credit (0.10) for showing work — not 0.30.
    direction_correct = (agent_eligible == truly_eligible)
    has_numbers  = bool(re.search(r'\d+', comp))
    has_time_ref = any(w in comp for w in TIME_WORDS)

    if has_numbers and has_time_ref:
        score += 0.3 if direction_correct else 0.10
    elif has_numbers or has_time_ref:
        score += 0.15 if direction_correct else 0.05

    # D4: Cap score when custody data is unreliable (likely dataset default)
    if custody_unreliable:
        score = min(score, 0.60)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# 4. Condition Appropriateness  (20%)
# ---------------------------------------------------------------------------

# Condition categories — agent must hit at least 3 distinct categories, not word count
CONDITION_CATEGORIES = {
    "financial":  ["surety", "bond", "₹", "amount", "personal bond"],
    "movement":   ["passport", "travel", "leave country", "district", "state"],
    "reporting":  ["report", "court", "weekly", "monthly", "police station"],
    "cooperation":["cooperate", "investigation", "tamper", "evidence", "witnesses"],
    "residence":  ["address", "notify", "change residence", "employment"],
}


def compute_condition_score(
    recommended_outcome: str,
    recommended_conditions: List[str],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Scores the appropriateness of recommended bail conditions.

    If bail denied: conditions should be empty.
    If bail granted: conditions must cover multiple distinct categories
                     (financial, movement, reporting, cooperation).

    Scoring:
        Denied + no conditions + GT denied  → 1.0
        Denied + no conditions + GT granted → 0.3 (directionally wrong)
        Denied + conditions present         → 0.5 (internal inconsistency)
        Granted + no conditions             → 0.2 (unusual, suspicious)
        Granted + conditions:
            - 0.6 base: proportion of categories covered (need ≥3 of 5)
            - 0.4 precision: overlap with what HC judgment actually mentions
    """
    gt_outcome = ground_truth["outcome"]
    gt_reason  = ground_truth.get("judgment_reason", "").lower()
    gt_detail  = ground_truth.get("outcome_detail", "").lower()
    gt_text    = gt_reason + " " + gt_detail

    outcome_denied = "deni" in recommended_outcome.lower()
    gt_denied      = "deni" in gt_outcome.lower()

    # --- Denied case ---
    if outcome_denied:
        if not recommended_conditions:
            return 1.0 if gt_denied else 0.3
        return 0.5  # Denied but listed conditions — internal inconsistency

    # --- Granted case ---
    if not recommended_conditions:
        return 0.2  # Granted with zero conditions is suspicious

    cond_text = " ".join(recommended_conditions).lower()

    # Category coverage (0.6 weight): how many distinct condition types did the agent use?
    categories_hit = sum(
        1 for cat_keywords in CONDITION_CATEGORIES.values()
        if any(kw in cond_text for kw in cat_keywords)
    )
    # Need at least 3 categories to get full marks
    category_score = min(1.0, categories_hit / 3)

    # GT overlap (0.4 weight): do the agent's conditions match what HC actually ordered?
    gt_categories_present = [
        cat for cat, kws in CONDITION_CATEGORIES.items()
        if any(kw in gt_text for kw in kws)
    ]
    if gt_categories_present:
        overlap = sum(
            1 for cat in gt_categories_present
            if any(kw in cond_text for kw in CONDITION_CATEGORIES[cat])
        )
        precision = overlap / len(gt_categories_present)
    else:
        precision = category_score  # No GT condition info → fall back to category score

    return min(1.0, 0.6 * category_score + 0.4 * precision)


# ---------------------------------------------------------------------------
# 5. Bias Penalty  (λ = 0.3)
# ---------------------------------------------------------------------------

# Crime type proxies for socioeconomic status — use substring matching, not set membership
HIGH_SES_TERMS = ["fraud", "cheating", "white collar", "economic offence", "cybercrime", "corporate"]
LOW_SES_TERMS  = ["theft", "robbery", "dacoity", "kidnapping", "narcotics", "pocso"]


def compute_bias_penalty(
    recommended_outcome: str,
    episode: Dict[str, Any],
    agent_grounds: Optional[List[str]] = None,
) -> float:
    """
    Computes a bias penalty for the agent's recommendation.

    Signal coverage in the 1,200-case dataset:
        - parity_argument_used=True: 341/1200 (28%) ← PRIMARY signal
        - SES-proxy via crime_type:  present in ~40% of cases ← SECONDARY signal
        - bias_flag=True:            13/1200 (1%)   ← rare override

    Weights are set to reflect actual data coverage — parity is the main lever.
    """
    gt          = episode["ground_truth"]
    gt_outcome  = gt["outcome"]
    bias_flag   = gt.get("bias_flag", False)
    parity_used = gt.get("parity_argument_used", False)
    crime_type  = episode.get("crime_type", "").lower()

    agent_granted = "grant" in recommended_outcome.lower() or "conditional" in recommended_outcome.lower()
    gt_granted    = "grant" in gt_outcome.lower()

    penalty = 0.0

    # ── Penalty 1 (PRIMARY): Ignoring parity argument (28% coverage) ─────────
    # HC cited co-accused parity yet agent diverges from HC — structural unfairness.
    if parity_used and (agent_granted != gt_granted):
        penalty += 0.25

    # ── Penalty 2: SES-proxy bias ─────────────────────────────────────────────
    is_high_ses = any(term in crime_type for term in HIGH_SES_TERMS)
    if is_high_ses and agent_granted and not gt_granted:
        penalty += 0.15  # Granting to high-SES when HC denied

    is_low_ses = any(term in crime_type for term in LOW_SES_TERMS)
    if is_low_ses and not agent_granted and gt_granted:
        penalty += 0.10  # Denying to low-SES when HC granted

    # ── Penalty 3 (RARE): Known biased-denial case (1% coverage) ─────────────
    # bias_flag=True: lower court made biased denial; HC overturned.
    # Agent repeating the same biased mistake gets a smaller override penalty.
    if bias_flag and gt_granted and not agent_granted:
        penalty += 0.15

    # ── Penalty 4: Parity case — agent diverges AND never mentions parity ─────
    # HC relied on co-accused parity; agent disagrees AND didn't engage with it.
    if parity_used and (agent_granted != gt_granted) and agent_grounds is not None:
        grounds_lower = " ".join(agent_grounds).lower()
        if not any(w in grounds_lower for w in PARITY_WORDS):
            penalty += 0.10  # Extra for ignoring parity without acknowledging it

    return max(0.0, min(1.0, penalty))


# ---------------------------------------------------------------------------
# 6. Reasoning Quality  (10% — replaces 10% from outcome weight)
# ---------------------------------------------------------------------------

PARITY_WORDS = ["parity", "co-accused", "co accused", "similarly placed",
                "bail granted to", "co-prisoner", "coaccused"]


def compute_reasoning_quality(
    flight_risk_justification: str,
    agent_risk_label: str,
    statutory_computation: str,
    grounds_for: List[str],
    grounds_against: List[str],
    episode: Dict[str, Any],
) -> float:
    """
    Scores the quality of the agent's reasoning without an LLM judge.

    Three sub-scores (averaged):
      1. Justification anchoring  — does flight risk justification cite
         case-specific facts (crime type, IPC section, custody duration)?
      2. Arithmetic verification  — do the actual episode numbers appear
         in the statutory computation (not just any number)?
      3. Grounds specificity      — do bail grounds reference crime-specific
         facts rather than boilerplate?

    Plus a consistency deduction:
      - Label says Low but text contains High-risk keywords → -0.10
      - Label says High but text contains Low-risk keywords → -0.10
    """
    just         = flight_risk_justification.lower()
    comp         = statutory_computation.lower()
    grounds_text = " ".join(grounds_for + grounds_against).lower()

    sections   = episode.get("ipc_sections", [])
    custody_mo = episode.get("custody_months") or 0.0
    max_sent   = episode.get("max_sentence_years", 5.0)
    crime_type = episode.get("crime_type", "").lower()

    # ── Sub-score 1: Justification anchoring ──────────────────────────────
    anchor_hits, anchor_max = 0, 0
    if crime_type:
        # At least one meaningful word from crime type in justification
        if any(w in just for w in crime_type.split() if len(w) > 3):
            anchor_hits += 1
        anchor_max += 1
    if sections:
        if any(sec.strip() in just for sec in sections):
            anchor_hits += 1
        anchor_max += 1
    if custody_mo > 0:
        # Exact custody months mentioned
        if str(int(custody_mo)) in just or f"{custody_mo:.1f}" in just:
            anchor_hits += 1
        anchor_max += 1

    just_words = len(just.split())
    raw_anchor = anchor_hits / max(1, anchor_max)
    # Cap anchoring score at 0.5 if justification is suspiciously short
    anchor_score = raw_anchor if just_words >= 15 else min(0.5, raw_anchor)

    # ── Sub-score 2: Arithmetic verification ──────────────────────────────
    if custody_mo > 0:
        threshold_mo = (max_sent * 12) / 2
        comp_numbers = [float(n) for n in re.findall(r'\d+\.?\d*', comp)]
        has_custody   = any(abs(n - custody_mo)   <= 1.5 for n in comp_numbers)
        has_threshold = any(abs(n - threshold_mo) <= 2.0 or
                            abs(n - (max_sent * 12)) <= 2.0
                            for n in comp_numbers)
        comp_words = len(comp.split())
        if comp_words < 10:
            arith_score = 0.3 if (has_custody or has_threshold) else 0.0
        else:
            arith_score = 0.5 * has_custody + 0.5 * has_threshold
    else:
        arith_score = 0.5  # No custody data — neutral, can't verify

    # ── Sub-score 3: Grounds specificity ─────────────────────────────────
    g_hits, g_max = 0, 0
    if crime_type:
        if any(w in grounds_text for w in crime_type.split() if len(w) > 3):
            g_hits += 1
        g_max += 1
    if sections:
        if any(sec.strip() in grounds_text for sec in sections):
            g_hits += 1
        g_max += 1
    grounds_words = len(grounds_text.split())
    raw_grounds = g_hits / max(1, g_max)
    grounds_score = raw_grounds if grounds_words >= 10 else min(0.4, raw_grounds)

    base = (anchor_score + arith_score + grounds_score) / 3

    # ── Consistency deduction: label contradicts justification text ────────
    label = agent_risk_label.strip().lower()
    consistency_deduction = 0.0
    if "low" in label:
        high_hits = sum(1 for kw in FLIGHT_RISK_KEYWORDS["High"] if kw in just)
        if high_hits >= 2:
            consistency_deduction = 0.10
    elif "high" in label:
        low_hits = sum(1 for kw in FLIGHT_RISK_KEYWORDS["Low"] if kw in just)
        if low_hits >= 2:
            consistency_deduction = 0.10

    return round(max(0.0, min(1.0, base - consistency_deduction)), 4)


# ---------------------------------------------------------------------------
# 7. Think-block reasoning gate (B6)
# ---------------------------------------------------------------------------

def compute_think_factor(completion: str, current_stage: int) -> float:
    """
    Gate outcome credit on reasoning quality.
    Stage 1: soft floor of 0.3 minimum (model still learning format).
    Stage 2+: hard gate — no reasoning = no outcome credit.
    Threshold: 120 words for full credit.
    """
    if not completion:
        return 0.3 if current_stage == 1 else 0.0

    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    think_len = len(think_text.split())
    raw_factor = min(1.0, think_len / 120.0)

    if current_stage == 1:
        # Soft floor: minimum 0.3 credit even with no think block
        # Ensures GRPO has non-zero gradient signal in early training
        return 0.3 + 0.7 * raw_factor
    else:
        # Hard gate: must reason to earn outcome credit
        return raw_factor


# ---------------------------------------------------------------------------
# 8. Format compliance (B8)
# ---------------------------------------------------------------------------

def reward_format(completion: str) -> float:
    """
    Score structural compliance of the bail memo.
    Checks for required XML tags matching the system prompt and valid outcome.
    Returns 0.0–1.0 (fraction of required elements present).
    """
    if not completion:
        return 0.0

    # Tags must match exactly what SYSTEM_PROMPT instructs the model to produce
    required_tags = [
        r'<think>',
        r'<memo>',
        r'<flight_risk>',
        r'<statutory_eligible>',
        r'<recommended_outcome>',
        r'<statutory_computation>',
    ]
    valid_outcomes = [
        'bail granted', 'bail denied',
        'conditional bail', 'default bail',
    ]

    checks = [
        bool(re.search(tag, completion, re.IGNORECASE))
        for tag in required_tags
    ]
    checks.append(
        any(outcome in completion.lower() for outcome in valid_outcomes)
    )

    return sum(checks) / len(checks)


# ---------------------------------------------------------------------------
# Master reward function
# ---------------------------------------------------------------------------

def compute_reward(
    agent_outcome: str,
    agent_flight_risk: str,
    agent_eligible: bool,
    agent_computation: str,
    agent_conditions: List[str],
    episode: Dict[str, Any],
    step_count: int = 0,
    max_steps: int = 10,
    statutory_tool_used: bool = False,
    agent_flight_risk_justification: str = "",
    agent_grounds_for: Optional[List[str]] = None,
    agent_grounds_against: Optional[List[str]] = None,
    completion_text: Optional[str] = None,
    current_stage: int = 1,
) -> Dict[str, float]:
    """
    Computes the full reward for a submitted bail assessment memo.

    Formula (B6/B8 update):
        R = 0.4*outcome_gated                 (gated by think_factor)
          + 0.2*flight_risk_accuracy
          + 0.2*statutory_accuracy
          + 0.2*condition_appropriateness
          + 0.1*reasoning_quality
          + 0.05*efficiency_bonus
          + 0.05*format_score
          + process_bonus
          - 0.3*bias_penalty

    Core components: 0.4+0.2+0.2+0.2 = 1.0
    Bonuses: rq(0.1) + eff(0.05) + fmt(0.05) + process(0.05)
    Penalty: -0.3*bias

    Returns a dict with all component scores + total_reward.
    """
    gt = episode["ground_truth"]

    grounds_all = (agent_grounds_for or []) + (agent_grounds_against or [])

    om   = compute_outcome_match(agent_outcome, gt)
    fr   = compute_flight_risk_accuracy(agent_flight_risk, gt)
    sa   = compute_statutory_accuracy(agent_eligible, agent_computation, episode)
    ca   = compute_condition_score(agent_outcome, agent_conditions, gt)
    bias = compute_bias_penalty(agent_outcome, episode, agent_grounds=grounds_all)
    rq   = compute_reasoning_quality(
        flight_risk_justification = agent_flight_risk_justification,
        agent_risk_label          = agent_flight_risk,
        statutory_computation     = agent_computation,
        grounds_for               = agent_grounds_for or [],
        grounds_against           = agent_grounds_against or [],
        episode                   = episode,
    )

    # B6: Gate outcome credit on reasoning quality (think block)
    # In server path, completion_text may be None (structured memo submission)
    # — default to think_factor=1.0 (no gating; env already enforces min tools).
    if completion_text:
        think_factor = compute_think_factor(completion_text, current_stage)
    else:
        think_factor = 1.0
    om_gated = om * think_factor

    # B8: Format compliance score
    fmt = reward_format(completion_text) if completion_text else 0.5

    # Efficiency bonus: reward finishing faster when the answer is correct.
    # Only fires on directionally-correct outcomes (om >= 0.8) to prevent
    # rewarding efficient-but-wrong agents.
    efficiency = 0.0
    if om >= 0.8 and max_steps > 1:
        efficiency = round((1.0 - (step_count - 1) / (max_steps - 1)), 4)
        efficiency = max(0.0, min(1.0, efficiency))

    # Process reward: +0.05 if agent actually used the statutory tool.
    process_bonus = 0.05 if statutory_tool_used else 0.0

    # Reward formula:
    # Core (sum=1.0): 0.4*outcome_gated + 0.2*flight + 0.2*statutory + 0.2*conditions
    # Bonuses:        0.1*reasoning_quality + 0.05*efficiency + 0.05*format
    # Process:        +0.05 if statutory tool used
    # Penalty:        -0.3*bias
    lam   = 0.3
    total = (0.4*om_gated + 0.2*fr + 0.2*sa + 0.2*ca
             + 0.1*rq + 0.05*efficiency + 0.05*fmt
             + process_bonus - lam*bias)

    return {
        "outcome_match":             round(om,           4),
        "outcome_match_gated":       round(om_gated,     4),
        "think_factor":              round(think_factor,  4),
        "flight_risk_accuracy":      round(fr,           4),
        "statutory_accuracy":        round(sa,           4),
        "condition_appropriateness": round(ca,           4),
        "reasoning_quality":         round(rq,           4),
        "format_score":              round(fmt,          4),
        "efficiency_bonus":          round(efficiency,   4),
        "process_bonus":             round(process_bonus,4),
        "bias_penalty":              round(bias,         4),
        "total_reward":              round(total,        4),
        "ground_truth_outcome":      gt["outcome"],
        "agent_outcome":             agent_outcome,
        "steps_used":                step_count,
    }
