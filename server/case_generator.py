"""
UndertriAI — Synthetic Case Generator (Theme 4: Self-Improvement)

When the agent masters a domain, this generates harder synthetic variants
of existing cases. All generation is deterministic string manipulation —
no LLM calls.

5 perturbation types:
  1. custody_escalation  — custody just below statutory threshold
  2. co_accused_conflict — co-accused with opposite bail outcome
  3. section_ambiguity   — IPC ↔ BNSS section swap
  4. evidence_reversal   — retracted witness / unreliable evidence
  5. surety_complexity   — non-resident surety complication
"""

import copy
import re
from typing import Any, Dict, List, Optional


# IPC → BNSS mapping (subset used by the environment)
IPC_TO_BNSS = {
    "302": "103", "307": "109", "376": "64",  "304B": "80",  "395": "310",
    "392": "309", "420": "318", "498A": "85", "406":  "316", "465": "336",
    "323": "115", "354": "74",  "120B": "61", "506":  "351", "121": "147",
    "379": "303", "324": "117", "354A": "75",
}
BNSS_TO_IPC = {v: k for k, v in IPC_TO_BNSS.items()}


# ── Required fields for schema validation ────────────────────────────

REQUIRED_FIELDS = {
    "case_id": str,
    "crime_type": str,
    "ipc_sections": list,
    "custody_months": (int, float),
    "charge_sheet": str,
    "ground_truth": dict,
    "curriculum_stage": (int, float),
}


def is_schema_valid(episode: Dict[str, Any]) -> bool:
    """
    Check that all required fields are present and correct types.
    Returns True/False — used to filter out malformed synthetic cases.
    """
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in episode:
            return False
        if not isinstance(episode[field], expected_type):
            return False

    # ground_truth must have 'outcome'
    gt = episode.get("ground_truth", {})
    if "outcome" not in gt:
        return False

    return True


def generate_variants(
    source_episode: Dict[str, Any],
    n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate up to n synthetic harder variants of a real episode.
    Each variant applies exactly ONE perturbation.

    Returns only valid variants (may be fewer than n if some
    perturbations can't be applied cleanly).
    """
    if not is_schema_valid(source_episode):
        return []

    perturbations = [
        _custody_escalation,
        _co_accused_conflict,
        _section_ambiguity,
        _evidence_reversal,
        _surety_complexity,
    ]

    variants = []
    for i, perturb_fn in enumerate(perturbations[:n]):
        try:
            variant = perturb_fn(source_episode)
            if variant is not None and is_schema_valid(variant):
                variants.append(variant)
        except Exception:
            # Skip perturbation on any error
            continue

    return variants


# ── Perturbation 1: Custody Escalation ───────────────────────────────

def _custody_escalation(episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Set custody_months to exactly 2 months below the statutory threshold.
    Forces careful computation — case is NOT yet eligible for default bail.
    """
    ep = copy.deepcopy(episode)
    max_sent = ep.get("max_sentence_years", 5.0)

    # Threshold is 50% of max sentence in months
    threshold_months = (max_sent * 12) / 2
    new_custody = max(1.0, threshold_months - 2.0)

    old_custody = ep.get("custody_months", 0)
    ep["custody_months"] = round(new_custody, 1)

    # Update charge sheet text if it mentions custody duration
    charge = ep.get("charge_sheet", "")
    if str(int(old_custody)) in charge:
        charge = charge.replace(
            f"{int(old_custody)} months",
            f"{int(new_custody)} months",
        )
    ep["charge_sheet"] = charge

    # Metadata
    parent_id = ep.get("case_id", "UNKNOWN")
    ep["case_id"] = f"SYN_{parent_id}_CUST"
    ep["source"] = "synthetic"
    ep["parent_case_id"] = parent_id
    ep["perturbation_type"] = "custody_escalation"
    ep["difficulty"] = "hard"

    return ep


# ── Perturbation 2: Co-Accused Conflict ──────────────────────────────

def _co_accused_conflict(episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Add a co-accused with the OPPOSITE bail outcome.
    Forces the agent to make a parity argument.
    """
    ep = copy.deepcopy(episode)
    gt = ep.get("ground_truth", {})
    gt_outcome = gt.get("outcome", "Bail Granted")

    # Opposite outcome
    if "grant" in gt_outcome.lower():
        co_outcome = "Bail Denied"
    else:
        co_outcome = "Bail Granted"

    ep["co_accused"] = [{
        "name": "Co-Accused A",
        "bail_outcome": co_outcome,
        "sections": ep.get("ipc_sections", []),
    }]

    gt["parity_argument_used"] = True
    ep["ground_truth"] = gt

    # Add parity context to defence arguments
    defence = ep.get("defence_arguments", [])
    defence.append(
        f"Co-accused was {'granted' if 'grant' in co_outcome.lower() else 'denied'} "
        f"bail under identical charges — parity principle applies."
    )
    ep["defence_arguments"] = defence

    # Metadata
    parent_id = ep.get("case_id", "UNKNOWN")
    ep["case_id"] = f"SYN_{parent_id}_COAC"
    ep["source"] = "synthetic"
    ep["parent_case_id"] = parent_id
    ep["perturbation_type"] = "co_accused_conflict"
    ep["difficulty"] = "hard"

    return ep


# ── Perturbation 3: Section Ambiguity (IPC ↔ BNSS) ──────────────────

def _section_ambiguity(episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Swap IPC sections to BNSS equivalents (or vice versa).
    Tests schema drift adaptability.
    """
    ep = copy.deepcopy(episode)
    sections = ep.get("ipc_sections", [])

    if not sections:
        return None

    new_sections = []
    swapped = False
    for sec in sections:
        sec_clean = sec.strip()
        if sec_clean in IPC_TO_BNSS:
            new_sections.append(IPC_TO_BNSS[sec_clean])
            swapped = True
        elif sec_clean in BNSS_TO_IPC:
            new_sections.append(BNSS_TO_IPC[sec_clean])
            swapped = True
        else:
            new_sections.append(sec_clean)

    if not swapped:
        return None

    ep["ipc_sections"] = new_sections

    # Update charge sheet references
    charge = ep.get("charge_sheet", "")
    for old_sec, new_sec in zip(sections, new_sections):
        if old_sec != new_sec:
            charge = charge.replace(f"Section {old_sec}", f"Section {new_sec}")
            charge = charge.replace(f"section {old_sec}", f"section {new_sec}")
    ep["charge_sheet"] = charge

    # Metadata
    parent_id = ep.get("case_id", "UNKNOWN")
    ep["case_id"] = f"SYN_{parent_id}_SECT"
    ep["source"] = "synthetic"
    ep["parent_case_id"] = parent_id
    ep["perturbation_type"] = "section_ambiguity"
    ep["difficulty"] = "hard"

    return ep


# ── Perturbation 4: Evidence Reversal ────────────────────────────────

def _evidence_reversal(episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Add a contradicting element to the strongest evidence.
    Tests whether the agent updates assessment when evidence weakens.
    """
    ep = copy.deepcopy(episode)

    # Find the strongest evidence mention
    evidence_keywords = ["witness", "evidence", "testimony", "eyewitness"]
    pros_args = ep.get("prosecution_arguments", [])
    charge = ep.get("charge_sheet", "")

    # Check prosecution arguments first
    target_arg = None
    for arg in pros_args:
        if any(kw in arg.lower() for kw in evidence_keywords):
            target_arg = arg
            break

    if target_arg is None:
        # Check charge sheet sentences
        sentences = [s.strip() for s in charge.split('.') if s.strip()]
        for sent in sentences:
            if any(kw in sent.lower() for kw in evidence_keywords):
                target_arg = sent
                break

    if target_arg is None:
        return None  # No evidence to reverse

    # Add reversal to defence arguments
    defence = ep.get("defence_arguments", [])
    defence.append(
        "However, the key prosecution evidence was subsequently found "
        "unreliable — the primary witness retracted their statement and "
        "forensic analysis raised doubts about the physical evidence."
    )
    ep["defence_arguments"] = defence

    # Metadata
    parent_id = ep.get("case_id", "UNKNOWN")
    ep["case_id"] = f"SYN_{parent_id}_EVID"
    ep["source"] = "synthetic"
    ep["parent_case_id"] = parent_id
    ep["perturbation_type"] = "evidence_reversal"
    ep["difficulty"] = "hard"

    return ep


# ── Perturbation 5: Surety Complexity ────────────────────────────────

def _surety_complexity(episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Add a surety complication forcing careful condition assessment.
    """
    ep = copy.deepcopy(episode)

    # Add surety complication to defence arguments
    defence = ep.get("defence_arguments", [])
    defence.append(
        "Proposed surety is a non-resident relative with no verifiable "
        "local assets or employment in the jurisdiction. Surety bond "
        "amount of Rs. 5,00,000 proposed."
    )
    ep["defence_arguments"] = defence

    # Add surety info to accused profile
    profile = ep.get("accused_profile", {})
    profile["surety_status"] = "non-resident, unverified assets"
    ep["accused_profile"] = profile

    # Metadata
    parent_id = ep.get("case_id", "UNKNOWN")
    ep["case_id"] = f"SYN_{parent_id}_SURE"
    ep["source"] = "synthetic"
    ep["parent_case_id"] = parent_id
    ep["perturbation_type"] = "surety_complexity"
    ep["difficulty"] = "hard"

    return ep
