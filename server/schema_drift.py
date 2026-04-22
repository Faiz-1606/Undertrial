"""
UndertriAI — Schema Drift Injector (Patronus AI Bonus Track)
Applies realistic schema drift to simulate IPC→BNSS transition and
cross-state FIR format variation, testing agent robustness.
"""

import random
import re
from typing import Any, Dict

random.seed(42)

# ---------------------------------------------------------------------------
# IPC → BNSS Section Remapping (Bharatiya Nyaya Sanhita 2023)
# ---------------------------------------------------------------------------

IPC_TO_BNSS: Dict[str, str] = {
    "120B": "61",   "121": "147",  "121A": "148",
    "302":  "103",  "304": "105",  "304B": "80",
    "306":  "108",  "307": "109",  "323": "115",
    "324":  "117",  "325": "118",  "326A": "124",
    "341":  "126",  "354": "74",   "354A": "75",
    "376":  "64",   "377": "377",  "379": "303",
    "380":  "305",  "392": "309",  "395": "310",
    "396":  "311",  "406": "316",  "420": "318",
    "465":  "336",  "467": "337",  "468": "338",
    "471":  "340",  "474": "341",  "498A": "85",
    "504":  "352",  "506": "351",  "509": "79",
    "34":   "3(5)", "149": "190",
}

BNSS_TO_IPC = {v: k for k, v in IPC_TO_BNSS.items()}


def remap_sections_to_bnss(sections: list) -> list:
    """Remap IPC section numbers to BNSS equivalents."""
    result = []
    for sec in sections:
        clean = sec.strip()
        mapped = IPC_TO_BNSS.get(clean, clean)
        result.append(f"BNS {mapped}")
    return result


# ---------------------------------------------------------------------------
# Regional FIR Format Templates
# ---------------------------------------------------------------------------

REGIONAL_HEADERS = {
    "Tamil Nadu": (
        "FIRST INFORMATION REPORT\n"
        "நம்பகமான தகவல் அறிக்கை (F.I.R)\n"
        "Tamil Nadu Police | State Crime Records Bureau\n"
        "---\n"
    ),
    "Kerala": (
        "FIRST INFORMATION REPORT\n"
        "Kerala Police | District Crime Records Bureau\n"
        "Registered under CrPC Section 154 / BNSS Section 173\n"
        "---\n"
    ),
    "Punjab": (
        "FIRST INFORMATION REPORT\n"
        "Punjab Police | District: __\n"
        "Thana No. __ | FIR No. __ | Year: __\n"
        "Under Sections: Bharatiya Nyaya Sanhita, 2023\n"
        "---\n"
    ),
    "Maharashtra": (
        "प्रथम सूचना अहवाल (F.I.R)\n"
        "FIRST INFORMATION REPORT\n"
        "Maharashtra Police | Taluka: __ | District: __\n"
        "---\n"
    ),
    "Assam": (
        "FIRST INFORMATION REPORT\n"
        "Assam Police | Registration No.: __\n"
        "Under Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023\n"
        "---\n"
    ),
}

BNSS_PROCEDURAL_INSERTS = [
    "This matter is registered under the Bharatiya Nyaya Sanhita (BNS), 2023, "
    "which supersedes the Indian Penal Code (IPC), 1860 with effect from July 1, 2024.",

    "Bail conditions are governed by Chapter XXXV of the Bharatiya Nagarik Suraksha "
    "Sanhita (BNSS), 2023 (formerly CrPC), specifically Sections 479–483.",

    "Default bail under Section 479 BNSS: accused is eligible where custody exceeds "
    "one-half of the maximum sentence for the offence, subject to no special law restriction.",

    "The Bharatiya Sakshya Adhiniyam (BSA) 2023 governs evidentiary standards in this matter.",
]


# ---------------------------------------------------------------------------
# Main drift injector
# ---------------------------------------------------------------------------

def apply_schema_drift(episode: Dict[str, Any], drift_type: str = "auto") -> Dict[str, Any]:
    """
    Apply schema drift to an episode dict. Returns a modified copy.
    drift_type: "bnss" | "regional" | "auto" (chooses randomly)
    """
    import copy
    ep = copy.deepcopy(episode)
    region = ep.get("region", "")

    if drift_type == "auto":
        if region in REGIONAL_HEADERS:
            drift_type = random.choice(["bnss", "regional", "combined"])
        else:
            drift_type = "bnss"

    # --- Apply BNSS section remapping ---
    if drift_type in ("bnss", "combined"):
        ep["ipc_sections"] = remap_sections_to_bnss(ep.get("ipc_sections", []))
        ep["schema_variant"] = "bnss"

        # Insert BNSS procedural text into charge sheet
        insert = random.choice(BNSS_PROCEDURAL_INSERTS)
        ep["charge_sheet"] = insert + "\n\n" + ep.get("charge_sheet", "")

        # Replace "IPC Section" mentions in text with "BNS Section"
        ep["charge_sheet"] = re.sub(
            r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?I\.P\.C\.?|IPC\s+[Ss]ection\s+(\d+[A-Z]?)",
            lambda m: f"BNS Section {IPC_TO_BNSS.get(m.group(1) or m.group(2), m.group(1) or m.group(2))}",
            ep["charge_sheet"],
        )

    # --- Apply regional FIR header ---
    if drift_type in ("regional", "combined") and region in REGIONAL_HEADERS:
        header = REGIONAL_HEADERS[region]
        ep["charge_sheet"] = header + ep.get("charge_sheet", "")
        ep["schema_variant"] = f"regional_{region.lower().replace(' ', '_')}"

    # Mark as drifted
    ep["schema_drifted"] = True
    return ep


def maybe_apply_drift(episode: Dict[str, Any], probability: float = 0.25) -> Dict[str, Any]:
    """Apply drift with the given probability (used during Stage 4 training)."""
    if episode.get("schema_drift_eligible") and random.random() < probability:
        return apply_schema_drift(episode)
    return episode
