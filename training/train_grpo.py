"""
UndertriAI — GRPO Training Script
Fine-tunes Qwen2.5-3B-Instruct using Group Relative Policy Optimization
against the UndertriAI bail assessment environment.

Run in Google Colab (T4 GPU recommended):
    !pip install unsloth trl openenv-core
    !pip install git+https://huggingface.co/spaces/Draken1606/undertrial-ai
    # Then run this script

Or locally:
    python training/train_grpo.py --episodes_dir ./data/episodes --output ./output
"""

# ============================================================
# CELL 1 — Install dependencies  (paste into Colab cell)
# ============================================================
INSTALL_COMMANDS = """
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps trl peft accelerate bitsandbytes xformers
!pip install -q openenv-core datasets
"""

# ============================================================
# CELL 2 — Imports
# ============================================================

import os, sys, json, re, argparse, random, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import urllib.request
import urllib.parse

import torch

# ── Environment API (Gap 1) ─────────────────────────────────────────────────
ENV_API_URL = os.environ.get(
    "UNDERTRIAL_ENV_URL",
    "https://draken1606-undertrial-ai.hf.space",
)

# ── Fix 1: Import authoritative reward functions from server/reward.py ──────
# This ensures training optimises the SAME signal the deployed demo evaluates.
try:
    _SERVER_ROOT = str(Path(__file__).parent.parent)
    if _SERVER_ROOT not in sys.path:
        sys.path.insert(0, _SERVER_ROOT)
    from server.reward import (
        compute_outcome_match,
        compute_flight_risk_accuracy,
        compute_statutory_accuracy,
        compute_condition_score,
        compute_bias_penalty as _server_bias,
        compute_reasoning_quality,
        compute_think_factor,
        reward_format as server_reward_format,
        _is_ndps_case,
    )
    _USE_SERVER_REWARDS = True
    print("[reward] Using authoritative server/reward.py functions.")
except ImportError:
    _USE_SERVER_REWARDS = False
    print("[reward] server/reward.py not found — using local fallback functions.")

    # Local fallback definition of _is_ndps_case (mirrors server/reward.py)
    def _is_ndps_case(episode: dict) -> bool:
        sections = " ".join(str(s) for s in episode.get("ipc_sections", [])).lower()
        crime = str(episode.get("crime_type", "")).lower()
        narcotics_indicators = [
            "ndps", "narcotic", "drug", "psychotropic",
            "20(b)", "22(b)", "27a", "section 37",
        ]
        return any(ind in sections or ind in crime for ind in narcotics_indicators)

    # Local fallback definition of compute_think_factor (mirrors server/reward.py)
    def compute_think_factor(completion: str, current_stage: int) -> float:
        if not completion:
            return 0.3 if current_stage == 1 else 0.0
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else ""
        think_len = len(think_text.split())
        raw_factor = min(1.0, think_len / 120.0)
        if current_stage == 1:
            return 0.3 + 0.7 * raw_factor
        else:
            return raw_factor

    # Local fallback server_reward_format
    server_reward_format = None  # Will use local reward_format below
from datasets import Dataset

# ============================================================
# CELL 3 — Prompt template
# ============================================================

SYSTEM_PROMPT = """You are a senior judicial clerk AI preparing a bail assessment memo for the judge.
You must read the case carefully and produce a structured assessment.

Your response MUST be in this exact XML format:

<think>
[Your step-by-step legal reasoning here. Consider:
1. The charges and applicable IPC/BNSS sections
2. Maximum sentence and time served (default bail eligibility)
3. Flight risk factors from the case facts
4. Prosecution objections vs defence arguments
5. Relevant legal precedents]
</think>

<memo>
<flight_risk>Low|Medium|High</flight_risk>
<flight_risk_justification>[specific reasons from facts]</flight_risk_justification>
<statutory_eligible>true|false</statutory_eligible>
<statutory_computation>[Section X → max Y years → threshold Z months → served W months → eligible/not]</statutory_computation>
<grounds_for_bail>
  <ground>[ground 1]</ground>
  <ground>[ground 2]</ground>
</grounds_for_bail>
<grounds_against_bail>
  <ground>[ground 1]</ground>
</grounds_against_bail>
<recommended_outcome>Bail Granted|Bail Denied</recommended_outcome>
<recommended_conditions>
  <condition>[condition if granted]</condition>
</recommended_conditions>
</memo>"""


def format_case_prompt(episode: Dict[str, Any]) -> str:
    """Format a case episode into a prompt string."""
    profile = episode.get("accused_profile", {})
    ipc = ", ".join(episode.get("ipc_sections", []))
    pros = "\n".join(f"  • {a}" for a in episode.get("prosecution_arguments", []))
    defe = "\n".join(f"  • {a}" for a in episode.get("defence_arguments", []))

    prompt = f"""═══ BAIL CASE: {episode.get('case_title', 'Unknown')} ═══
Court: {episode.get('court', 'Unknown')} | Date: {episode.get('date', 'Unknown')}

CHARGE SHEET:
{episode.get('charge_sheet', '')}

SECTIONS INVOKED: {ipc}
CRIME TYPE: {episode.get('crime_type', 'Unknown')}
BAIL TYPE: {episode.get('bail_type', 'Regular')}

ACCUSED PROFILE:
  Name:        {profile.get('name', 'Unknown')}
  Gender:      {profile.get('gender', 'Unknown')}
  Region:      {profile.get('region', 'Unknown')}
  Prior Cases: {profile.get('prior_cases', 'Unknown')}

CUSTODY DURATION: {episode.get('custody_months', 0):.1f} months
MAX SENTENCE:     {episode.get('max_sentence_years', 5):.1f} years

PROSECUTION ARGUMENTS:
{pros or '  • None specified'}

DEFENCE ARGUMENTS:
{defe or '  • None specified'}

LEGAL PRINCIPLES CITED:
{chr(10).join(f"  • {p}" for p in episode.get('legal_principles', [])) or '  • None cited'}

SCHEMA VARIANT: {episode.get('schema_variant', 'standard')}
"""
    return prompt


# ============================================================
# CELL 4 — Reward functions (deterministic, rule-based)
# ============================================================

def extract_xml_field(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_xml_list(text: str, tag: str, item_tag: str = "ground") -> List[str]:
    block = extract_xml_field(text, tag)
    return re.findall(rf"<{item_tag}>(.*?)</{item_tag}>", block, re.DOTALL)


def parse_model_output(output: str) -> Dict[str, Any]:
    """Parse model's XML output into structured fields.

    IMPORTANT: If no <memo> tag is found, returns a zero-scored empty dict.
    We do NOT fall back to parsing raw output text — that's an exploit vector
    where an agent could scatter keywords in free text to trigger regex hits.
    """
    memo_block = extract_xml_field(output, "memo")
    if not memo_block:
        # Return empty/zero dict — no reward for malformed output
        return {
            "recommended_outcome":   "",
            "flight_risk":           "",
            "flight_risk_just":      "",
            "statutory_eligible":    False,
            "statutory_computation": "",
            "grounds_for":           [],
            "grounds_against":       [],
            "conditions":            [],
            "has_think_block":       "<think>" in output.lower(),
        }

    return {
        "recommended_outcome":  extract_xml_field(memo_block, "recommended_outcome"),
        "flight_risk":          extract_xml_field(memo_block, "flight_risk"),
        "flight_risk_just":     extract_xml_field(memo_block, "flight_risk_justification"),
        "statutory_eligible":   extract_xml_field(memo_block, "statutory_eligible").lower() == "true",
        "statutory_computation":extract_xml_field(memo_block, "statutory_computation"),
        "grounds_for":          extract_xml_list(memo_block, "grounds_for_bail", "ground"),
        "grounds_against":      extract_xml_list(memo_block, "grounds_against_bail", "ground"),
        "conditions":           extract_xml_list(memo_block, "recommended_conditions", "condition"),
        "has_think_block":      "<think>" in output.lower(),
    }


def reward_format(completions: List[str], **kwargs) -> List[float]:
    """Reward well-formed XML output structure (batch API for GRPO compatibility)."""
    return [reward_format_single(c) for c in completions]


def reward_format_single(completion: str) -> float:
    """
    Score structural compliance of the bail memo.
    Checks for required XML tags matching the system prompt and valid outcome.
    Returns 0.0–1.0 (fraction of required elements present).
    """
    if not completion:
        return 0.0
    # Tags match exactly what SYSTEM_PROMPT instructs the model to produce
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


def reward_outcome_match(completions: List[str], episode_batch: List[Dict], **kwargs) -> List[float]:
    """40% weight: does the agent's recommendation match the HC decision?"""
    scores = []
    for comp, ep in zip(completions, episode_batch):
        parsed = parse_model_output(comp)
        agent_out = parsed["recommended_outcome"].lower()
        gt_out    = ep["ground_truth"]["outcome"].lower()

        if not agent_out:
            scores.append(0.0)
            continue

        if ("grant" in agent_out and "grant" in gt_out) or \
           ("den" in agent_out and "den" in gt_out):
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


def reward_flight_risk(completions: List[str], episode_batch: List[Dict], **kwargs) -> List[float]:
    """20% weight: flight risk classification vs implicit GT."""
    scores = []
    for comp, ep in zip(completions, episode_batch):
        parsed    = parse_model_output(comp)
        agent_fr  = parsed["flight_risk"].strip()
        gt_fr     = ep["ground_truth"].get("implicit_flight_risk", "Medium")
        risk_vals = {"Low": 0, "Medium": 1, "High": 2}
        diff      = abs(risk_vals.get(agent_fr, 1) - risk_vals.get(gt_fr, 1))
        scores.append(1.0 if diff == 0 else (0.5 if diff == 1 else 0.0))
    return scores


def reward_statutory(completions: List[str], episode_batch: List[Dict], **kwargs) -> List[float]:
    """20% weight: correct statutory eligibility computation.

    B3: Direction-gated computation bonus — wrong direction gets 0.10 not 0.30.
    B9: NDPS cases use crime_type detection and reward Section 37 recognition.
    """
    TIME_WORDS = ["month", "year", "sentence", "custody", "half", "served", "threshold"]
    scores = []
    for comp, ep in zip(completions, episode_batch):
        parsed    = parse_model_output(comp)
        comp_text = parsed["statutory_computation"].lower()
        sections  = ep.get("ipc_sections", [])
        max_sent  = ep.get("max_sentence_years", 5.0)
        custody   = ep.get("custody_months", 0.0)
        special_laws = ep.get("special_laws", "").strip()
        gt_outcome = ep.get("ground_truth", {}).get("outcome", "")
        agent_eligible = parsed["statutory_eligible"]

        # B9: NDPS-specific scoring
        if _is_ndps_case(ep):
            gt_granted = "grant" in gt_outcome.lower()
            direction_correct = (agent_eligible == gt_granted)
            ndps_recognized = any(
                t in comp_text for t in ["section 37", "twin condition", "ndps", "37(1)(b)"]
            )
            if ndps_recognized and direction_correct:
                scores.append(1.0)
            elif direction_correct:
                scores.append(0.5)
            else:
                scores.append(0.0)
            continue

        # Infer special law from crime_type
        CRIME_TYPE_SPECIAL_LAWS = [
            "narcotics", "ndps", "pocso", "uapa", "pmla",
            "terrorism", "organised crime", "money laundering",
        ]
        crime_type_lower = ep.get("crime_type", "").lower()
        if not special_laws and any(t in crime_type_lower for t in CRIME_TYPE_SPECIAL_LAWS):
            special_laws = "INFERRED"

        # Standard IPC/BNSS threshold computation
        half_sent_months = (max_sent * 12) / 2
        truly_eligible = (custody >= half_sent_months) and not special_laws

        score = 0.0

        # 40%: eligibility direction
        direction_correct = (agent_eligible == truly_eligible)
        if direction_correct:
            score += 0.4
        elif (agent_eligible and "grant" in gt_outcome.lower()) or \
             (not agent_eligible and "deni" in gt_outcome.lower()):
            score += 0.2

        # 30%: cited relevant sections
        if sections:
            hits = sum(1 for sec in sections if sec.strip().lower() in comp_text or sec.strip() in comp)
            score += 0.3 * min(1.0, hits / len(sections))

        # 30%: numeric computation (B3: direction-gated)
        has_numbers = bool(re.search(r'\d+', comp_text))
        has_time_ref = any(w in comp_text for w in TIME_WORDS)
        if has_numbers and has_time_ref:
            score += 0.3 if direction_correct else 0.10
        elif has_numbers or has_time_ref:
            score += 0.15 if direction_correct else 0.05

        scores.append(min(1.0, score))
    return scores

def reward_conditions(completions: List[str], episode_batch: List[Dict], **kwargs) -> List[float]:
    """20% weight: appropriate bail conditions for the case type and risk profile."""
    scores = []
    for comp, ep in zip(completions, episode_batch):
        parsed     = parse_model_output(comp)
        outcome    = parsed["recommended_outcome"].lower()
        conditions = parsed["conditions"]
        score = 0.0
        if "grant" in outcome:
            if len(conditions) >= 1: score += 0.5
            if len(conditions) >= 2: score += 0.3
            cond_text = " ".join(conditions).lower()
            for kw in ["surety", "bond", "report", "passport", "permission"]:
                if kw in cond_text:
                    score = min(1.0, score + 0.04)
        else:
            # Denial: empty conditions is correct ONLY when GT also denied
            gt_outcome = ep.get("ground_truth", {}).get("outcome", "").lower()
            gt_denied = "deni" in gt_outcome
            if len(conditions) == 0:
                score = 1.0 if gt_denied else 0.3  # H3: 0.3 not 1.0 when GT=granted
            else:
                score = 0.5  # Denied but listed conditions — inconsistent
        scores.append(min(1.0, score))
    return scores

def reward_no_bias(completions: List[str], episode_batch: List[Dict], **kwargs) -> List[float]:
    """
    Bias penalty: penalise outputs that appear demographically skewed.
    Returns penalty values (subtract from total).
    """
    scores = []
    for comp, ep in zip(completions, episode_batch):
        parsed   = parse_model_output(comp)
        outcome  = parsed["recommended_outcome"].lower()
        just     = (parsed["flight_risk_just"] + comp).lower()
        bias_flag = ep["ground_truth"].get("bias_flag", False)
        penalty  = 0.0

        # Name/religion proxy: agent should NOT mention accused's name/community
        # in flight risk justification (only legal factors should drive it)
        profile_name = ep.get("accused_profile", {}).get("name", "").lower()
        if profile_name and profile_name in just:
            penalty += 0.1  # Using name (possible proxy for community) in reasoning

        # Bias-flagged case: check if agent blindly follows biased lower-court
        if bias_flag and "denied" in outcome:
            penalty += 0.15

        scores.append(penalty)
    return scores


def combined_reward(
    completions: List[str],
    episode_batch: List[Dict],
    current_stage: int = 1,
    **kwargs
) -> List[float]:
    """
    Master reward combining all components.

    Formula (B6/B8 update):
        R = 0.4*outcome_gated + 0.2*flight_risk + 0.2*statutory + 0.2*condition
          + 0.1*reasoning_quality + 0.05*format
          - 0.3*bias

    Core (sum=1.0): 0.4*om_gated + 0.2*fr + 0.2*s + 0.2*ca
    Bonuses:        0.1*rq + 0.05*fmt
    Penalty:        -0.3*bias

    Uses server/reward.py functions when available (Fix 1).
    B6: Outcome gated by think_factor (stage-aware).
    B8: Format compliance score included with 0.05 weight.
    """
    rewards = []

    for comp, ep in zip(completions, episode_batch):
        parsed = parse_model_output(comp)
        gt     = ep.get("ground_truth", {})

        if _USE_SERVER_REWARDS:
            # Use the authoritative server functions
            o  = compute_outcome_match(parsed["recommended_outcome"], gt)
            fr = compute_flight_risk_accuracy(parsed["flight_risk"], gt)
            s  = compute_statutory_accuracy(
                parsed["statutory_eligible"],
                parsed["statutory_computation"],
                ep,
            )
            ca = compute_condition_score(
                parsed["recommended_outcome"],
                parsed.get("conditions", []),
                gt,
            )
            b  = _server_bias(
                parsed["recommended_outcome"], ep,
                agent_grounds=parsed.get("grounds_for", []) + parsed.get("grounds_against", []),
            )
            rq = compute_reasoning_quality(
                flight_risk_justification = parsed.get("flight_risk_just", ""),
                agent_risk_label          = parsed.get("flight_risk", ""),
                statutory_computation     = parsed.get("statutory_computation", ""),
                grounds_for               = parsed.get("grounds_for", []),
                grounds_against           = parsed.get("grounds_against", []),
                episode                   = ep,
            )
        else:
            # Local fallback
            o  = reward_outcome_match([comp], [ep])[0]
            fr = reward_flight_risk([comp], [ep])[0]
            s  = reward_statutory([comp], [ep])[0]
            ca = reward_conditions([comp], [ep])[0]
            b  = reward_no_bias([comp], [ep])[0]
            rq = 0.5  # Neutral when server functions unavailable

        # B6: Gate outcome credit on reasoning quality (think block)
        think_factor = compute_think_factor(comp, current_stage)
        om_gated = o * think_factor

        # B8: Format compliance score
        if _USE_SERVER_REWARDS and server_reward_format is not None:
            fmt = server_reward_format(comp)
        else:
            fmt = reward_format_single(comp)

        # Reward formula:
        # Core (sum=1.0): 0.4*outcome_gated + 0.2*flight + 0.2*statutory + 0.2*conditions
        # Bonuses:        0.1*reasoning_quality + 0.05*format
        # Penalty:        -0.3*bias
        total = (0.4*om_gated + 0.2*fr + 0.2*s + 0.2*ca
                 + 0.1*rq + 0.05*fmt - 0.3*b)
        rewards.append(round(total, 4))  # No max(0.0) clamp — bias can go negative
    return rewards


# ============================================================
# CELL 5 — Dataset builder
# ============================================================

def load_episodes(
    episodes_dir: str,
    stage: int = 1,
    split: str = "train",
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
) -> List[Dict]:
    """
    Load episodes for a given split (Gap 2: train/val/test split).

    Split fractions (applied deterministically by index, no shuffle):
        train  = first (1 - val - test) fraction
        val    = next val_fraction
        test   = last test_fraction
    """
    path = Path(episodes_dir) / f"episodes_stage_{stage}.jsonl"
    use_all_fallback = False
    if not path.exists():
        path = Path(episodes_dir) / "episodes_all.jsonl"
        use_all_fallback = True
    if not path.exists():
        raise FileNotFoundError(f"No episodes found in {episodes_dir}.")
    with open(path, encoding="utf-8") as f:
        all_eps = [json.loads(l) for l in f if l.strip()]
    # H1: filter by curriculum_stage when falling back to episodes_all.jsonl
    if use_all_fallback:
        filtered = [ep for ep in all_eps if ep.get("curriculum_stage") == stage]
        if filtered:
            all_eps = filtered

    n = len(all_eps)
    n_test = max(1, int(n * test_fraction))
    n_val  = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    if split == "train":
        return all_eps[:n_train]
    elif split == "val":
        return all_eps[n_train:n_train + n_val]
    elif split == "test":
        return all_eps[n_train + n_val:]
    else:
        return all_eps  # all: for backward compat


def rollout_via_env_api(
    completion: str,
    episode: Dict,
    env_url: str = ENV_API_URL,
    session_id: Optional[str] = None,
    timeout: float = 10.0,
) -> float:
    """
    Gap 1: Route reward through the live deployed environment API.

    Sends the model's completion to the environment server via HTTP,
    replaying the parsed submit_memo action, and returns the official reward.
    Falls back to local reward on any network error.
    """
    import urllib.error
    try:
        from server.reward import compute_reward as _local_reward
    except ImportError:
        _local_reward = None

    parsed = parse_model_output(completion)
    if not parsed["recommended_outcome"]:
        return 0.0  # Malformed output

    try:
        # Step 1: Reset the environment with the correct episode
        episode_stage = episode.get("curriculum_stage", 1)
        reset_url = f"{env_url}/reset?stage={episode_stage}"
        req = urllib.request.Request(reset_url, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reset_data = json.loads(resp.read())
        sid = session_id or reset_data.get("session_id", "")

        # Step 2: Submit the parsed memo
        memo_payload = json.dumps({
            "session_id": sid,
            "action": {
                "tool_name": "submit_memo",
                "flight_risk": parsed["flight_risk"] or "Medium",
                "flight_risk_justification": parsed["flight_risk_just"] or "Not specified",
                "statutory_eligible": parsed["statutory_eligible"],
                "statutory_computation": parsed["statutory_computation"] or "Not computed",
                "grounds_for_bail": parsed["grounds_for"] or [],
                "grounds_against_bail": parsed["grounds_against"] or [],
                "recommended_outcome": parsed["recommended_outcome"],
                "recommended_conditions": parsed["conditions"] or [],
                "confidence": "Medium",
            }
        }).encode()
        step_req = urllib.request.Request(
            f"{env_url}/step",
            data=memo_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(step_req, timeout=timeout) as resp:
            step_data = json.loads(resp.read())
        return float(step_data.get("reward", 0.0))

    except Exception as e:
        # Network / parse error: fall back to local reward
        print(f"[env_api] Falling back to local reward: {e}")
        if _local_reward and episode:
            rd = _local_reward(
                agent_outcome=parsed["recommended_outcome"],
                agent_flight_risk=parsed["flight_risk"] or "Medium",
                agent_eligible=parsed["statutory_eligible"],
                agent_computation=parsed["statutory_computation"] or "",
                agent_conditions=parsed["conditions"] or [],
                episode=episode,
            )
            return rd["total_reward"]
        return 0.0


def build_hf_dataset(episodes: List[Dict], tokenizer) -> Dataset:
    """Build HuggingFace Dataset with prompt/episode pairs."""
    rows = []
    for ep in episodes:
        prompt = format_case_prompt(ep)
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rows.append({
            "prompt":        formatted,
            "episode":       json.dumps(ep),   # stored as string for Dataset compat
            "case_id":       ep.get("case_id",""),
            "ground_truth":  ep["ground_truth"]["outcome"],
        })
    return Dataset.from_list(rows)


# ============================================================
# CELL 6 — Training
# ============================================================

# ── Fix 3: Generation Inspection Callback ───────────────────────────────────
try:
    from transformers import TrainerCallback  # type: ignore

    class GenerationInspectionCallback(TrainerCallback):
        """Prints 2 raw model completions every 25 steps to catch reward hacking."""
        def __init__(self, tokenizer, dataset, every_n_steps=25):
            self.tokenizer    = tokenizer
            self.dataset      = dataset
            self.every_n      = every_n_steps
            self._sample_idxs = random.sample(range(len(dataset)), min(2, len(dataset)))

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.every_n != 0 or model is None:
                return
            print(f"\n{'─'*60}")
            print(f"[InspectionCallback] Step {state.global_step} — sample completions:")
            from unsloth import FastLanguageModel  # type: ignore
            FastLanguageModel.for_inference(model)
            for idx in self._sample_idxs:
                row = self.dataset[idx]
                inputs = self.tokenizer(
                    row["prompt"], return_tensors="pt", truncation=True, max_length=1024
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                gt = row.get("ground_truth", "?")
                print(f"  [GT={gt}] {text[:300]}...")
            FastLanguageModel.for_training(model)
            print(f"{'─'*60}\n")
except ImportError:
    GenerationInspectionCallback = None


def train(
    episodes_dir: str = "./data/episodes",
    output_dir:   str = "./output/undertrial_grpo",
    stage:        int = 1,
    max_steps:    int = 200,
    batch_size:   int = 4,
    grad_accum:   int = 4,
    lr:           float = 5e-6,
    max_seq_len:  int = 3072,
    eval_after:   bool = False,
):
    print("=" * 60)
    print("  UndertriAI — GRPO Training with Unsloth")
    print(f"  Model: Qwen2.5-3B-Instruct | Stage: {stage}")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = "unsloth/Qwen2.5-3B-Instruct",
        max_seq_length = max_seq_len,
        load_in_4bit = True,
        fast_inference = False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha     = 16,
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )

    # ── Load dataset ─────────────────────────────────────────
    print(f"\nLoading Stage {stage} episodes from {episodes_dir}...")
    episodes = load_episodes(episodes_dir, stage=stage)
    print(f"Loaded {len(episodes)} episodes.")

    dataset = build_hf_dataset(episodes, tokenizer)

    # Reward wrapper that unpacks the stored JSON episode
    def reward_fn(completions: List[str], episode: List[str], **kwargs) -> List[float]:
        ep_objs = [json.loads(e) for e in episode]
        return combined_reward(completions, ep_objs, current_stage=stage)

    # ── GRPO Config ──────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    config = GRPOConfig(
        output_dir              = output_dir,
        learning_rate           = lr,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        num_train_epochs        = 1,
        max_steps               = max_steps,
        num_generations         = 6,           # G in GRPO
        max_completion_length   = 1024,
        temperature             = 0.7,
        beta                    = 0.01,        # KL penalty coefficient
        logging_steps           = 5,
        save_steps              = 50,
        report_to               = "none",
        remove_unused_columns   = False,
    )

    # A3 fix: reuse the already-loaded model — evaluate_baseline() loads a second
    # FastLanguageModel internally which OOMs on T4 with a model already in memory.
    print("\nRunning baseline evaluation (before training)...")
    baseline_reward, _ = evaluate_on_stage(model, tokenizer, episodes_dir, stage=stage, n_samples=20)
    print(f"Baseline reward: {baseline_reward:.4f}")

    # ── Trainer ──────────────────────────────────────────────
    callbacks = []
    if GenerationInspectionCallback is not None:
        callbacks.append(GenerationInspectionCallback(tokenizer, dataset, every_n_steps=25))

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        config           = config,
        train_dataset    = dataset,
        reward_funcs     = [reward_fn],
        callbacks        = callbacks,
    )

    print("\nStarting GRPO training...")
    print(f"  Steps: {max_steps} | Batch: {batch_size} x {grad_accum} grad_accum")
    print(f"  Generations per prompt: 6 | KL beta: 0.01")
    print(f"  Inspection callback: every 25 steps")
    print()

    trainer.train()

    # ── Fix 2: Post-training eval + results.json ──────────────
    post_reward = None
    if eval_after:
        print("\nRunning post-training evaluation...")
        # A3 fix: reuse model, no second load
        post_reward, _ = evaluate_on_stage(model, tokenizer, episodes_dir, stage=stage, n_samples=20)
        print(f"Post-training reward: {post_reward:.4f}")
        print(f"Improvement: {baseline_reward:.4f} → {post_reward:.4f} (+{post_reward-baseline_reward:.4f})")

    results = {
        "stage":          stage,
        "max_steps":      max_steps,
        "baseline_reward": round(baseline_reward, 4),
        "post_reward":    round(post_reward, 4) if post_reward is not None else None,
        "delta":          round(post_reward - baseline_reward, 4) if post_reward else None,
        "training_log":   [
            {k: v for k, v in e.items() if isinstance(v, (int, float, str))}
            for e in trainer.state.log_history
        ],
    }
    results_path = Path(output_dir) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Save LoRA adapters only — safe for 4-bit quantized models
    model.save_pretrained(output_dir, save_adapters_only=True)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel adapters saved to {output_dir}")

    # Save training plots (C6)
    save_training_plots(trainer.state.log_history, output_dir)

    return results



# ============================================================
# Plot saving utility (C6)
# ============================================================

def save_training_plots(log_history: list, output_dir: str) -> None:
    """
    Save training reward curve and loss plots.
    Called at the end of train(), train_curriculum(), and train_adaptive().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARNING] matplotlib not installed — skipping plot generation.")
        return

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Extract reward data from training log
    steps   = [e["step"]   for e in log_history if "reward" in e]
    rewards = [e["reward"] for e in log_history if "reward" in e]

    if not steps:
        print("[WARNING] No reward data in training log — skipping plots.")
        return

    # Plot 1: Reward curve
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0d1a")
    ax.set_facecolor("#0a0d1a")
    ax.plot(steps, rewards, color="#6366f1", linewidth=1.5, alpha=0.6, label="Raw")
    if len(rewards) > 5:
        smooth = np.convolve(rewards, np.ones(5) / 5, mode="valid")
        ax.plot(steps[2:-2], smooth, color="#14b8a6", linewidth=2, label="Smoothed")
    ax.set_xlabel("Training Step", color="#94a3b8")
    ax.set_ylabel("Reward", color="#94a3b8")
    ax.set_title("UndertriAI — Training Reward Curve", color="#e2e8f0", pad=12)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, alpha=0.2)
    ax.legend(facecolor="#111827", edgecolor="#1e2d45", labelcolor="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1e2d45")
    fig.tight_layout()
    reward_path = plots_dir / "reward_curve.png"
    fig.savefig(str(reward_path), dpi=150, bbox_inches="tight", facecolor="#0a0d1a")
    plt.close(fig)
    print(f"  Plot saved: {reward_path}")

    # Plot 2: Loss curve (if available)
    loss_steps  = [e["step"] for e in log_history if "loss" in e]
    loss_values = [e["loss"] for e in log_history if "loss" in e]
    if loss_steps:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor("#0a0d1a")
        ax2.set_facecolor("#0a0d1a")
        ax2.plot(loss_steps, loss_values, color="#f97316", linewidth=1.5)
        ax2.set_xlabel("Training Step", color="#94a3b8")
        ax2.set_ylabel("Loss", color="#94a3b8")
        ax2.set_title("UndertriAI — Training Loss", color="#e2e8f0", pad=12)
        ax2.tick_params(colors="#94a3b8")
        ax2.grid(True, alpha=0.2)
        for spine in ax2.spines.values():
            spine.set_color("#1e2d45")
        fig2.tight_layout()
        loss_path = plots_dir / "training_loss.png"
        fig2.savefig(str(loss_path), dpi=150, bbox_inches="tight", facecolor="#0a0d1a")
        plt.close(fig2)
        print(f"  Plot saved: {loss_path}")


# ============================================================
# CELL 7 — Evaluate baseline (before training)
# ============================================================

def evaluate_baseline(episodes_dir: str, n_samples: int = 20):
    """
    Quick evaluation of a zero-shot Qwen2.5-3B-Instruct on bail cases.
    Run this BEFORE training to get the baseline reward curve starting point.
    """
    print("\nEvaluating zero-shot baseline...")
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = "unsloth/Qwen2.5-3B-Instruct",
        max_seq_length = 3072,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    episodes = load_episodes(episodes_dir, stage=1)[:n_samples]
    rewards = []

    for ep in episodes:
        prompt = format_case_prompt(ep)
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

        completion = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        r = combined_reward([completion], [ep], current_stage=1)[0]
        rewards.append(r)
        print(f"  Case {ep['case_id']}: reward={r:.3f} | GT={ep['ground_truth']['outcome']}")

    avg = sum(rewards) / len(rewards) if rewards else 0
    print(f"\nBaseline average reward: {avg:.4f} (over {len(rewards)} cases)")
    print("(Expected ~0.25–0.35 for zero-shot; target after training: >0.60)")
    return avg


# ============================================================
# CELL 8 — Self-Improving Curriculum Training (Theme 4)
# ============================================================

STAGE_NAMES = {
    1: "Landmark (clear-cut cases)",
    2: "Contested (judgment calls)",
    3: "Bias Reversal (parity cases)",
    4: "Schema Drift (IPC→BNSS)",
}

STAGE_THRESHOLD = 0.60  # 60% outcome accuracy to unlock next stage


def evaluate_on_stage(
    model,
    tokenizer,
    episodes_dir: str,
    stage: int,
    n_samples: int = 20,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate the current model on held-out cases from a specific stage.
    Returns (average_reward, list of {episode, completion, reward} dicts).
    """
    from unsloth import FastLanguageModel  # type: ignore
    FastLanguageModel.for_inference(model)

    episodes = load_episodes(episodes_dir, stage=stage, split="val")[:n_samples]
    if not episodes:
        episodes = load_episodes(episodes_dir, stage=stage, split="train")[:n_samples]

    results = []
    rewards = []

    for ep in episodes:
        prompt = format_case_prompt(ep)
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

        completion = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        r = combined_reward([completion], [ep], current_stage=stage)[0]
        rewards.append(r)
        results.append({"episode": ep, "completion": completion, "reward": r})

    FastLanguageModel.for_training(model)
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    return avg, results


def extract_good_traces(
    eval_results: List[Dict],
    min_reward: float = 0.6,
    top_k: int = 3,
) -> List[str]:
    """
    Extract the best reasoning traces from evaluation results.
    Returns up to top_k completions that scored above min_reward.
    """
    good = [r for r in eval_results if r["reward"] >= min_reward]
    good.sort(key=lambda x: -x["reward"])
    traces = []
    for r in good[:top_k]:
        ep = r["episode"]
        # Compact trace: case_id + GT + agent completion
        trace = (
            f"--- Example: {ep.get('case_id', '?')} "
            f"(GT: {ep['ground_truth']['outcome']}) ---\n"
            f"{r['completion'][:600]}"
        )
        traces.append(trace)
    return traces


def inject_examples(base_prompt: str, traces: List[str]) -> str:
    """
    Add successful traces as few-shot examples to the system prompt.
    This is the core self-improvement mechanism: the model's own good
    reasoning from stage N becomes instructional context for stage N+1.
    """
    if not traces:
        return base_prompt

    examples_block = "\n\nHere are examples of CORRECT bail assessments from simpler cases:\n\n"
    examples_block += "\n\n".join(traces)
    examples_block += "\n\nNow apply the same structured reasoning to the following case:\n"

    return base_prompt + examples_block


def train_curriculum(
    episodes_dir: str = "./data/episodes",
    output_dir: str = "./output/undertrial_grpo",
    stages: List[int] = None,
    max_steps_per_stage: int = 150,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 5e-6,
    threshold: float = STAGE_THRESHOLD,
):
    """
    Self-improving curriculum training.

    The agent trains on stage N, then its best reasoning traces are
    harvested and injected as few-shot examples into stage N+1's prompt.
    Stage N+1 is only unlocked when stage N accuracy exceeds the threshold.

    This is the key self-improvement mechanism for Theme 4.
    """
    if stages is None:
        stages = [1, 2, 3, 4]

    print("=" * 60)
    print("  UndertriAI — Self-Improving Curriculum Training")
    print(f"  Stages: {stages} | Threshold: {threshold:.0%}")
    print("=" * 60)

    from unsloth import FastLanguageModel  # type: ignore
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    # Load model once — reused across all stages
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-Instruct",
        max_seq_length=3072,
        load_in_4bit=True,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    accumulated_traces: List[str] = []
    stage_results = {}
    current_prompt = SYSTEM_PROMPT

    for stage in stages:
        print(f"\n{'━' * 60}")
        print(f"  STAGE {stage}: {STAGE_NAMES.get(stage, '?')}")
        print(f"{'━' * 60}")

        # ── Inject traces from previous stages into prompt ──
        if accumulated_traces:
            current_prompt = inject_examples(SYSTEM_PROMPT, accumulated_traces)
            print(f"  Injected {len(accumulated_traces)} successful traces from earlier stages")
        else:
            current_prompt = SYSTEM_PROMPT

        # ── Baseline eval for this stage ──
        print(f"\n  Evaluating baseline on Stage {stage}...")
        baseline_reward, _ = evaluate_on_stage(
            model, tokenizer, episodes_dir, stage, n_samples=20
        )
        print(f"  Stage {stage} baseline: {baseline_reward:.4f}")

        # ── Build dataset for this stage ──
        episodes = load_episodes(episodes_dir, stage=stage, split="train")
        if not episodes:
            print(f"  ⚠ No episodes for stage {stage} — skipping")
            continue
        print(f"  Training on {len(episodes)} episodes...")

        # Build HF dataset with potentially enriched prompt
        rows = []
        for ep in episodes:
            case_prompt = format_case_prompt(ep)
            messages = [
                {"role": "system", "content": current_prompt},
                {"role": "user", "content": case_prompt},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            rows.append({
                "prompt": formatted,
                "episode": json.dumps(ep),
                "case_id": ep.get("case_id", ""),
                "ground_truth": ep["ground_truth"]["outcome"],
            })
        dataset = Dataset.from_list(rows)

        def reward_fn(completions: List[str], episode: List[str], **kwargs) -> List[float]:
            ep_objs = [json.loads(e) for e in episode]
            return combined_reward(completions, ep_objs, current_stage=stage)

        stage_output = f"{output_dir}/stage_{stage}"
        config = GRPOConfig(
            output_dir=stage_output,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=1,
            max_steps=max_steps_per_stage,
            num_generations=6,
            max_completion_length=1024,
            temperature=0.7,
            beta=0.01,
            logging_steps=5,
            save_steps=50,
            report_to="none",
            remove_unused_columns=False,
        )

        # ── Switch model back to training mode before trainer.train() ──
        # evaluate_on_stage calls FastLanguageModel.for_inference(model);
        # without this reset, stages 2-4 train in inference mode silently.
        FastLanguageModel.for_training(model)

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            config=config,
            train_dataset=dataset,
            reward_funcs=[reward_fn],
        )
        trainer.train()

        # ── Post-training eval ──
        print(f"\n  Evaluating after Stage {stage} training...")
        post_reward, eval_results = evaluate_on_stage(
            model, tokenizer, episodes_dir, stage, n_samples=20
        )
        improvement = post_reward - baseline_reward
        print(f"  Stage {stage}: {baseline_reward:.4f} → {post_reward:.4f} "
              f"(Δ = {improvement:+.4f})")

        stage_results[stage] = {
            "baseline": round(baseline_reward, 4),
            "post": round(post_reward, 4),
            "delta": round(improvement, 4),
        }

        # ── Harvest good traces for next stage ──
        new_traces = extract_good_traces(eval_results, min_reward=0.6, top_k=2)
        if new_traces:
            accumulated_traces.extend(new_traces)
            print(f"  ✓ Harvested {len(new_traces)} good traces for next stage")

        # ── Check threshold for stage progression ──
        if post_reward >= threshold:
            print(f"  ✓ Stage {stage} PASSED (reward {post_reward:.2f} ≥ {threshold:.2f})")
            if stage < max(stages):
                print(f"  → Unlocking Stage {stage + 1}")
        else:
            print(f"  ✗ Stage {stage} below threshold ({post_reward:.2f} < {threshold:.2f})")
            print(f"  → Continuing to next stage anyway (curriculum mode)")

        # Save LoRA adapters only — safe for 4-bit models (save_pretrained_merged
        # requires a full merge which can OOM on T4)
        model.save_pretrained(stage_output, save_adapters_only=True)
        tokenizer.save_pretrained(stage_output)
        print(f"  Checkpoint saved (adapters): {stage_output}")

    # ── Final summary ──
    print(f"\n{'═' * 60}")
    print("  CURRICULUM TRAINING COMPLETE")
    print(f"{'═' * 60}")
    for s, r in stage_results.items():
        status = "✓" if r["post"] >= threshold else "✗"
        print(f"  {status} Stage {s}: {r['baseline']:.4f} → {r['post']:.4f} "
              f"(Δ = {r['delta']:+.4f})")
    print(f"  Total traces harvested: {len(accumulated_traces)}")

    # Save final model (adapters only — merge separately if needed)
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir, save_adapters_only=True)
    tokenizer.save_pretrained(final_dir)
    print(f"\n  Final model saved (adapters): {final_dir}")

    # Save results
    results_path = Path(output_dir) / "curriculum_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "stages": stage_results,
        "traces_harvested": len(accumulated_traces),
        "threshold": threshold,
    }, indent=2))
    print(f"  Results saved: {results_path}")

    # Save training plots (C6) — use last trainer's log
    try:
        save_training_plots(trainer.state.log_history, output_dir)
    except Exception:
        print("  [WARNING] Could not save training plots.")

    return stage_results


# ============================================================
# CELL 9 — Adaptive Training (Theme 4: Self-Improvement)
# ============================================================

def train_adaptive(
    episodes_dir: str = "./data/episodes",
    output_dir: str = "./output/undertrial_adaptive",
    steps_per_assessment: int = 50,
    max_total_steps: int = 2000,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 5e-6,
    base_url: str = "http://localhost:8000",
):
    """
    Self-directed curriculum training (Theme 4).

    Uses the /profile endpoint to check stage readiness every
    steps_per_assessment steps and promotes automatically.

    This function communicates with the server via HTTP — it does NOT
    import server internals. OpenEnv client/server separation is preserved.

    Training loop:
      1. Start at stage 1
      2. Train for steps_per_assessment steps
      3. Query /profile for suggested_stage
      4. If suggested_stage > current_stage, promote
      5. Repeat until max_total_steps or stage 4 mastered
    """
    print("=" * 60)
    print("  UndertriAI — Adaptive Self-Improvement Training")
    print(f"  Assessment every {steps_per_assessment} steps | Max {max_total_steps} steps")
    print(f"  Server: {base_url}")
    print("=" * 60)

    from unsloth import FastLanguageModel  # type: ignore
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    # Load model once
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-Instruct",
        max_seq_length=3072,
        load_in_4bit=True,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    # HTTP helper for server communication
    def query_profile(session_id: str) -> Optional[Dict]:
        """Query the performance profile from the server via HTTP."""
        try:
            url = f"{base_url}/profile?session_id={urllib.parse.quote(session_id)}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print(f"  [adaptive] Could not reach server profile: {e}")
            return None

    def notify_reset(session_id: str, stage: int) -> Optional[str]:
        """Call /reset with adaptive=true on the server."""
        try:
            url = f"{base_url}/reset?session_id={urllib.parse.quote(session_id)}&stage={stage}&adaptive=true"
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read())
                return data.get("session_id", session_id)
        except Exception:
            return None

    current_stage = 1
    total_steps = 0
    session_id = f"adaptive_{uuid.uuid4().hex[:8]}" if 'uuid' in dir() else "adaptive_training"

    # Try to initialise session on server
    import uuid as _uuid_mod
    session_id = f"adaptive_{_uuid_mod.uuid4().hex[:8]}"
    notify_reset(session_id, current_stage)

    # Tracking
    stage_promotion_steps = []
    reward_curve = []
    stage_rewards = {1: [], 2: [], 3: [], 4: []}

    while total_steps < max_total_steps:
        print(f"\n{'━' * 60}")
        print(f"  ADAPTIVE BLOCK: Steps {total_steps}–{total_steps + steps_per_assessment}")
        print(f"  Current Stage: {current_stage} — {STAGE_NAMES.get(current_stage, '?')}")
        print(f"{'━' * 60}")

        # Load episodes for current stage
        try:
            episodes = load_episodes(episodes_dir, stage=current_stage, split="train")
        except FileNotFoundError:
            print(f"  No episodes for stage {current_stage} — breaking")
            break

        if not episodes:
            print(f"  Empty episode list for stage {current_stage} — breaking")
            break

        # Build dataset
        dataset = build_hf_dataset(episodes, tokenizer)
        stage_for_closure = current_stage  # Capture for closure

        def reward_fn(completions: List[str], episode: List[str], **kwargs) -> List[float]:
            ep_objs = [json.loads(e) for e in episode]
            return combined_reward(completions, ep_objs, current_stage=stage_for_closure)

        block_output = f"{output_dir}/block_{total_steps}"
        config = GRPOConfig(
            output_dir=block_output,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=1,
            max_steps=steps_per_assessment,
            num_generations=6,
            max_completion_length=1024,
            temperature=0.7,
            beta=0.01,
            logging_steps=5,
            save_steps=steps_per_assessment,
            report_to="none",
            remove_unused_columns=False,
        )

        FastLanguageModel.for_training(model)
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            config=config,
            train_dataset=dataset,
            reward_funcs=[reward_fn],
        )
        trainer.train()
        total_steps += steps_per_assessment

        # Evaluate current performance
        eval_reward, _ = evaluate_on_stage(
            model, tokenizer, episodes_dir, stage=current_stage, n_samples=15
        )
        stage_rewards[current_stage].append(eval_reward)
        reward_curve.append((total_steps, round(eval_reward, 4)))
        print(f"  Stage {current_stage} eval reward: {eval_reward:.4f}")

        # Query server for stage promotion suggestion
        profile_data = query_profile(session_id)
        suggested_stage = current_stage

        if profile_data and "profile" in profile_data:
            suggested_stage = profile_data["profile"].get(
                "suggested_stage", current_stage
            )
        else:
            # Fallback: use local heuristic
            if eval_reward >= 0.65 and current_stage == 1:
                suggested_stage = 2
            elif eval_reward >= 0.55 and current_stage == 2:
                suggested_stage = 3
            elif eval_reward >= 0.50 and current_stage == 3:
                suggested_stage = 4

        if suggested_stage > current_stage:
            old_stage = current_stage
            old_reward = eval_reward
            current_stage = suggested_stage
            stage_promotion_steps.append(
                (total_steps, old_stage, current_stage, round(old_reward, 4))
            )
            print(
                f"[SELF-IMPROVEMENT] Step {total_steps}: "
                f"Promoted to Stage {current_stage}. "
                f"Stage {old_stage} mean reward: {old_reward:.3f} → "
                f"Stage {current_stage} begins."
            )
            # Notify server of promotion
            notify_reset(session_id, current_stage)

        # Check completion
        if current_stage == 4:
            s4_rewards = stage_rewards.get(4, [])
            if s4_rewards and s4_rewards[-1] >= 0.50:
                print(
                    f"\n[SELF-IMPROVEMENT] Stage 4 mastered at step {total_steps}! "
                    f"Reward: {s4_rewards[-1]:.3f}"
                )
                break

        # Save checkpoint
        model.save_pretrained(block_output, save_adapters_only=True)
        tokenizer.save_pretrained(block_output)

    # ── Final summary ──
    print(f"\n{'═' * 60}")
    print("  ADAPTIVE TRAINING COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Total steps: {total_steps}")
    print(f"  Stage promotions: {len(stage_promotion_steps)}")
    for step_n, from_s, to_s, reward in stage_promotion_steps:
        print(f"    Step {step_n}: Stage {from_s} → {to_s} (reward {reward:.3f})")
    print(f"  Final stage: {current_stage}")

    # Compute final reward per stage
    final_reward_per_stage = {}
    for s, rewards_list in stage_rewards.items():
        if rewards_list:
            final_reward_per_stage[str(s)] = round(rewards_list[-1], 4)

    # Save results
    results = {
        "stage_promotion_steps": [
            {"step": s, "from_stage": f, "to_stage": t, "reward": r}
            for s, f, t, r in stage_promotion_steps
        ],
        "final_reward_per_stage": final_reward_per_stage,
        "total_steps_completed": total_steps,
        "reward_curve": [{"step": s, "reward": r} for s, r in reward_curve],
    }
    results_path = Path(output_dir) / "results_adaptive.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {results_path}")

    # Save final model
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir, save_adapters_only=True)
    tokenizer.save_pretrained(final_dir)
    print(f"  Final model saved: {final_dir}")

    # Save training plots (C6)
    # Build a synthetic log_history from reward_curve for adaptive mode
    adaptive_log = [{"step": s, "reward": r} for s, r in reward_curve]
    save_training_plots(adaptive_log, output_dir)

    return results


# ============================================================
# CELL 10 — Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UndertriAI GRPO Training")
    parser.add_argument("--episodes_dir", default="./data/episodes")
    parser.add_argument("--output",       default="./output/undertrial_grpo")
    parser.add_argument("--stage",        type=int, default=1)
    parser.add_argument("--steps",        type=int, default=200)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--baseline_only", action="store_true",
                        help="Only run baseline evaluation, skip training")
    parser.add_argument("--eval_after",    action="store_true",
                        help="Run evaluation after training to measure improvement")
    parser.add_argument("--curriculum",    action="store_true",
                        help="Run self-improving curriculum training (all 4 stages)")
    parser.add_argument("--adaptive",      action="store_true",
                        help="Run adaptive self-improvement training (Theme 4)")
    parser.add_argument("--env_url",       default="http://localhost:8000",
                        help="Server URL for adaptive training")

    args = parser.parse_args()

    if args.baseline_only:
        evaluate_baseline(args.episodes_dir)
    elif args.curriculum:
        train_curriculum(
            episodes_dir=args.episodes_dir,
            output_dir=args.output,
            max_steps_per_stage=args.steps,
            batch_size=args.batch_size,
        )
    elif args.adaptive:
        train_adaptive(
            episodes_dir=args.episodes_dir,
            output_dir=args.output,
            steps_per_assessment=args.steps,
            max_total_steps=2000,
            batch_size=args.batch_size,
            base_url=args.env_url,
        )
    else:
        train(
            episodes_dir = args.episodes_dir,
            output_dir   = args.output,
            stage        = args.stage,
            max_steps    = args.steps,
            batch_size   = args.batch_size,
            eval_after   = args.eval_after,
        )

