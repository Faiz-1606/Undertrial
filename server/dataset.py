"""
UndertriAI — Dataset Loader + Curriculum Sampler
Loads JSONL episode files and samples according to the current training stage.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema_drift import maybe_apply_drift


class BailDataset:
    """
    Loads and manages episode files for curriculum training.
    Falls back to in-memory episodes if JSONL files are not available.
    """

    def __init__(self, episodes_dir: Optional[str] = None):
        self._episodes: Dict[int, List[Dict]] = {1: [], 2: [], 3: [], 4: []}
        self._current_stage = 1
        self._episode_index: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

        # Determine episodes directory
        if episodes_dir is None:
            # Look relative to this file or env variable
            episodes_dir = os.environ.get(
                "UNDERTRIAL_EPISODES_DIR",
                str(Path(__file__).parent.parent / "data" / "episodes")
            )

        self._load(episodes_dir)

        if self.total_episodes == 0:
            print("[BailDataset] No JSONL files found — loading built-in demo episodes.")
            self._load_demo_episodes()

    def _load(self, episodes_dir: str) -> None:
        for stage in range(1, 5):
            path = os.path.join(episodes_dir, f"episodes_stage_{stage}.jsonl")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    episodes = [json.loads(line) for line in f if line.strip()]
                # Do NOT shuffle here: keeps episode order deterministic so
                # seed=0 (demo) always returns the same known first episode.
                # Training diversity comes from sequential _episode_index iteration.
                self._episodes[stage] = episodes
                print(f"[BailDataset] Stage {stage}: {len(episodes)} episodes loaded from {path}")

    def _load_demo_episodes(self) -> None:
        """Built-in minimal demo episodes so the env works without data files."""
        demo = [
            {
                "case_id": "DEMO001",
                "case_title": "Ramesh Kumar vs State of Delhi",
                "court": "Delhi High Court",
                "date": "2023-05-10",
                "charge_sheet": (
                    "The accused Ramesh Kumar, a 34-year-old auto-rickshaw driver, "
                    "was arrested on 14 February 2023 under IPC Section 420 (Cheating) "
                    "in connection with an alleged Rs. 50,000 fraud. He has been in "
                    "judicial custody for 8 months. He has no prior criminal record, "
                    "permanent residence in Delhi, and his family including two minor "
                    "children are dependent on him. The maximum sentence for IPC 420 "
                    "is 7 years. The prosecution has not cited any flight risk."
                ),
                "ipc_sections": ["420"],
                "crime_type": "Fraud or Cheating",
                "bail_type": "Regular",
                "prosecution_arguments": [
                    "The accused allegedly duped the complainant of Rs. 50,000.",
                    "Investigation is still pending and accused may tamper with evidence.",
                ],
                "defence_arguments": [
                    "Accused has been in custody for 8 months; threshold under BNSS 479 for a 7-year offence is 42 months — not yet met. Bail is sought on community ties and clean record, not statutory default.",
                    "No prior criminal record. Permanent resident of Delhi with family ties.",
                    "No evidence of flight risk or evidence tampering.",
                ],
                "legal_principles": ["Default bail under Section 436A CrPC / 479 BNSS"],
                "documents_available": ["FIR Copy", "Charge Sheet", "Surety Affidavit"],
                "summary": "Regular bail application by auto-rickshaw driver in cheating case after 8 months custody.",
                "accused_profile": {
                    "name": "Ramesh Kumar", "gender": "Male",
                    "occupation": "Auto-rickshaw driver", "region": "Delhi",
                    "prior_cases": "None", "bail_type": "Regular",
                },
                "custody_months": 8.0,
                "max_sentence_years": 7.0,
                "ground_truth": {
                    "outcome": "Bail Granted",
                    "implicit_flight_risk": "Low",
                    "judgment_reason": "Accused has deep roots in community, no flight risk, and custody approaching half of max sentence.",
                    "outcome_detail": "Bail granted with surety of Rs. 25,000 and weekly reporting.",
                    "bias_flag": False,
                    "parity_argument_used": False,
                },
                "curriculum_stage": 1,
                "landmark_case": True,
                "bail_cancellation_case": False,
                "region": "Delhi",
                "special_laws": "",
                "schema_drift_eligible": False,
            },
            {
                "case_id": "DEMO002",
                "case_title": "State of UP vs Santosh Singh",
                "court": "Allahabad High Court",
                "date": "2022-11-20",
                "charge_sheet": (
                    "Santosh Singh, 28, was arrested under IPC Sections 302 (Murder) "
                    "and 34 (Common Intention) for an alleged gang-related killing. "
                    "He has been in custody for 14 months. There are three eyewitnesses "
                    "and the prosecution argues he is a known associate of an organized "
                    "criminal syndicate. The accused has two prior cases including one "
                    "under the Arms Act. The maximum sentence for IPC 302 is life imprisonment."
                ),
                "ipc_sections": ["302", "34"],
                "crime_type": "Murder",
                "bail_type": "Regular",
                "prosecution_arguments": [
                    "Offence is grave — murder charge with life imprisonment.",
                    "Three eyewitnesses may be intimidated if accused is released.",
                    "Accused is part of organized criminal network with resources to abscond.",
                    "Two prior cases including Arms Act — repeat offender profile.",
                ],
                "defence_arguments": [
                    "14 months in custody — prolonged detention without trial.",
                    "Trial unlikely to conclude for several years.",
                ],
                "legal_principles": [
                    "Triple test: flight risk, evidence tampering, repeat offence",
                    "Gravity of offence is paramount in murder cases",
                ],
                "documents_available": ["FIR Copy", "Charge Sheet", "Criminal History Record"],
                "summary": "Bail denied to accused in murder case with organized crime links and eyewitnesses.",
                "accused_profile": {
                    "name": "Santosh Singh", "gender": "Male",
                    "occupation": None, "region": "Uttar Pradesh",
                    "prior_cases": "2 prior cases including Arms Act", "bail_type": "Regular",
                },
                "custody_months": 14.0,
                "max_sentence_years": 99.0,
                "ground_truth": {
                    "outcome": "Bail Denied",
                    "implicit_flight_risk": "High",
                    "judgment_reason": "Gravity of offence, organized crime nexus, eyewitness intimidation risk, and prior criminal record all weigh heavily against bail.",
                    "outcome_detail": "Bail rejected. Trial court directed to expedite proceedings.",
                    "bias_flag": False,
                    "parity_argument_used": False,
                },
                "curriculum_stage": 2,
                "landmark_case": False,
                "bail_cancellation_case": False,
                "region": "Uttar Pradesh",
                "special_laws": "",
                "schema_drift_eligible": False,
            },
            {
                "case_id": "DEMO003",
                "case_title": "Priya Sharma vs State of Maharashtra",
                "court": "Bombay High Court",
                "date": "2023-08-15",
                "charge_sheet": (
                    "Priya Sharma, 26-year-old domestic worker, was arrested under IPC "
                    "Section 498A (Cruelty by husband/relatives) following a matrimonial "
                    "dispute. She has been in custody for 11 months. The lower court denied "
                    "bail citing 'character concerns' — widely criticised as gender-biased. "
                    "Two co-accused sisters-in-law were granted bail 3 months ago under "
                    "identical charges (parity argument). Maximum sentence under 498A is 3 years."
                ),
                "ipc_sections": ["498A", "323"],
                "crime_type": "Domestic Violence",
                "bail_type": "Regular",
                "prosecution_arguments": [
                    "Accused allegedly instigated discord in the matrimonial home.",
                    "Complainant (husband) fears further harassment if accused is released.",
                ],
                "defence_arguments": [
                    "11 months custody on a 3-year max offence — threshold crossed.",
                    "Co-accused sisters-in-law granted bail under identical charges — parity demands same treatment.",
                    "Lower court's 'character' reasoning has no statutory basis — gender-discriminatory.",
                    "Accused is primary caregiver of infant child currently in custody.",
                ],
                "legal_principles": [
                    "Parity principle: similarly situated co-accused granted bail",
                    "Section 436A default bail — custody exceeds half of max sentence",
                ],
                "documents_available": ["FIR Copy", "Charge Sheet", "Surety Affidavit", "Medical Report"],
                "summary": "HC reversal: domestic worker denied bail on biased grounds; HC grants bail citing parity.",
                "accused_profile": {
                    "name": "Priya Sharma", "gender": "Female",
                    "occupation": "Domestic worker", "region": "Maharashtra",
                    "prior_cases": "None", "bail_type": "Regular",
                },
                "custody_months": 11.0,
                "max_sentence_years": 3.0,
                "ground_truth": {
                    "outcome": "Bail Granted",
                    "implicit_flight_risk": "Low",
                    "judgment_reason": "Co-accused granted bail on parity. Custody exceeds half of max sentence. Lower court's 'character' reasoning is not a valid statutory ground.",
                    "outcome_detail": "Bail granted with surety Rs. 10,000 and monthly reporting.",
                    "bias_flag": True,
                    "parity_argument_used": True,
                },
                "curriculum_stage": 3,
                "landmark_case": False,
                "bail_cancellation_case": True,
                "region": "Maharashtra",
                "special_laws": "",
                "schema_drift_eligible": True,
            },
            {
                "case_id": "DEMO004",
                "case_title": "Mohammed Irfan vs State of Kerala (BNSS)",
                "court": "Kerala High Court",
                "date": "2024-03-10",
                "charge_sheet": (
                    "FIRST INFORMATION REPORT — Kerala Police | BNSS Section 173\n"
                    "Mohammed Irfan, 32, school teacher, was arrested under BNS Section 318 "
                    "(formerly IPC 420) and BNS Section 316 (formerly IPC 406) for "
                    "misappropriating Rs. 1.2 lakh from a school parents' committee. "
                    "He has been in custody for 6 months. No prior criminal record. "
                    "Bail governed by Chapter XXXV BNSS 2023. Default bail under Section 479 BNSS."
                ),
                "ipc_sections": ["318", "316"],
                "crime_type": "Cheating",
                "bail_type": "Regular",
                "prosecution_arguments": [
                    "Misappropriation of school funds — breach of public trust.",
                    "Investigation of digital records still ongoing.",
                ],
                "defence_arguments": [
                    "BNS 318 max 7 years — Section 479 BNSS threshold is 42 months.",
                    "Digital records already seized — no tampering risk.",
                    "Permanent resident, employed teacher, first-time offence.",
                ],
                "legal_principles": [
                    "Section 479 BNSS 2023 — default bail (replaces Section 436A CrPC)",
                    "BNS Section 318 = former IPC 420",
                ],
                "documents_available": ["FIR Copy", "Charge Sheet", "Surety Affidavit", "Employment Proof"],
                "summary": "BNSS schema drift case: IPC sections remapped to BNS, bail under new BNSS procedural framework.",
                "accused_profile": {
                    "name": "Mohammed Irfan", "gender": "Male",
                    "occupation": "School teacher", "region": "Kerala",
                    "prior_cases": "None", "bail_type": "Regular",
                },
                "custody_months": 6.0,
                "max_sentence_years": 7.0,
                "ground_truth": {
                    "outcome": "Bail Granted",
                    "implicit_flight_risk": "Low",
                    "judgment_reason": "No flight risk. Permanent resident. Digital records secured. Eligible under Section 479 BNSS.",
                    "outcome_detail": "Bail granted under BNSS Section 479. Surety Rs. 20,000. Monthly reporting.",
                    "bias_flag": False,
                    "parity_argument_used": False,
                },
                "curriculum_stage": 4,
                "landmark_case": False,
                "bail_cancellation_case": False,
                "region": "Kerala",
                "special_laws": "",
                "schema_drift_eligible": True,
                "schema_drifted": True,
            },
        ]
        for ep in demo:
            stage = ep["curriculum_stage"]
            self._episodes[stage].append(ep)
        print(f"[BailDataset] Loaded {len(demo)} built-in demo episodes.")

    @property
    def total_episodes(self) -> int:
        return sum(len(eps) for eps in self._episodes.values())

    def set_stage(self, stage: int) -> None:
        assert 1 <= stage <= 4, "Stage must be 1–4"
        self._current_stage = stage
        print(f"[BailDataset] Curriculum stage set to {stage}")

    def sample_episode(
        self,
        stage: Optional[int] = None,
        apply_drift: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Sample an episode from the requested curriculum stage.

        Args:
            stage: Curriculum stage 1-4. Defaults to current stage.
            apply_drift: Apply BNSS schema drift for stage 4 episodes.
            seed: If set, deterministically picks episode at index (seed % len).
                  Used by the demo to always show the same illustrative case.
        """
        s = stage if stage is not None else self._current_stage

        # Fallback: if stage is empty, try adjacent stages
        for candidate in [s, s-1, s+1, 1, 2, 3, 4]:
            if 1 <= candidate <= 4 and self._episodes[candidate]:
                eps = self._episodes[candidate]
                if seed is not None:
                    idx = seed % len(eps)
                else:
                    idx = self._episode_index[candidate] % len(eps)
                    self._episode_index[candidate] += 1
                ep = eps[idx]
                if apply_drift and s == 4:
                    ep = maybe_apply_drift(ep, probability=0.4, seed=seed)
                return ep

        raise RuntimeError("No episodes available in any stage!")

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        return [ep for eps in self._episodes.values() for ep in eps]
