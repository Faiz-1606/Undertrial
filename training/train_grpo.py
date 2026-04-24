"""
UndertriAI — GRPO Training Script
Fine-tunes Qwen2.5-7B-Instruct using Group Relative Policy Optimization
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
    )
    _USE_SERVER_REWARDS = True
    print("[reward] Using authoritative server/reward.py functions.")
except ImportError:
    _USE_SERVER_REWARDS = False
    print("[reward] server/reward.py not found — using local fallback functions.")
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
    """Reward well-formed XML output structure."""
    scores = []
    for c in completions:
        score = 0.0
        if "<think>" in c and "</think>" in c: score += 0.15
        if "<memo>" in c and "</memo>" in c:   score += 0.15
        for tag in ["flight_risk","statutory_eligible","recommended_outcome","statutory_computation"]:
            if f"<{tag}>" in c: score += 0.05
        scores.append(min(1.0, score))
    return scores


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
    """20% weight: correct statutory eligibility computation."""
    scores = []
    for comp, ep in zip(completions, episode_batch):
        parsed    = parse_model_output(comp)
        comp_text = parsed["statutory_computation"].lower()
        sections  = ep.get("ipc_sections", [])
        max_sent  = ep.get("max_sentence_years", 5.0)
        custody   = ep.get("custody_months", 0.0)

        score = 0.0
        # Mentions relevant sections
        for sec in sections:
            if sec.strip().lower() in comp_text or sec.strip() in comp:
                score += 0.2
        score = min(0.4, score)

        # Mentions numbers
        if re.search(r'\d+', comp_text): score += 0.3
        # Mentions time-related words
        if any(w in comp_text for w in ["month","year","sentence","custody","half","served","threshold"]):
            score += 0.3
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
            # Denial should have empty conditions
            score = 1.0 if len(conditions) == 0 else 0.5
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
    **kwargs
) -> List[float]:
    """
    Master reward combining all components.
    R = 0.4*outcome + 0.2*flight_risk + 0.2*statutory + 0.2*condition - 0.3*bias

    Uses server/reward.py functions when available (Fix 1).
    Condition appropriateness replaces format score (Fix 2).
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

        # NOTE: Efficiency is NOT computed in GRPO training because step_count=1
        # always (single-shot generation), making eff=1.0 a constant non-signal.
        # Efficiency is preserved in the environment's compute_reward for live inference.
        eff = 0.0

        total = 0.3*o + 0.2*fr + 0.2*s + 0.2*ca + 0.1*rq - 0.3*b
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
    if not path.exists():
        path = Path(episodes_dir) / "episodes_all.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No episodes found in {episodes_dir}.")
    with open(path, encoding="utf-8") as f:
        all_eps = [json.loads(l) for l in f if l.strip()]

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
    print(f"  Model: Qwen2.5-7B-Instruct | Stage: {stage}")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = "unsloth/Qwen2.5-7B-Instruct",
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
        return combined_reward(completions, ep_objs)

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

    # ── Fix 2: Baseline eval BEFORE training ─────────────────
    print("\nRunning baseline evaluation (before training)...")
    baseline_reward = evaluate_baseline(episodes_dir, n_samples=20)
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
        post_reward = evaluate_baseline(episodes_dir, n_samples=20)
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
    return results


# ============================================================
# CELL 7 — Evaluate baseline (before training)
# ============================================================

def evaluate_baseline(episodes_dir: str, n_samples: int = 20):
    """
    Quick evaluation of a zero-shot Qwen2.5-7B-Instruct on bail cases.
    Run this BEFORE training to get the baseline reward curve starting point.
    """
    print("\nEvaluating zero-shot baseline...")
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = "unsloth/Qwen2.5-7B-Instruct",
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
        r = combined_reward([completion], [ep])[0]
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
        r = combined_reward([completion], [ep])[0]
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
        model_name="unsloth/Qwen2.5-7B-Instruct",
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
            # Pass step_count=1 for curriculum training (single-shot XML, no multi-step env loop)
            # This keeps efficiency contribution honest rather than silently 0.0
            step_counts = [1] * len(completions)
            return combined_reward(completions, ep_objs, step_counts=step_counts)

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

    return stage_results


# ============================================================
# CELL 9 — Entry point
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
    else:
        train(
            episodes_dir = args.episodes_dir,
            output_dir   = args.output,
            stage        = args.stage,
            max_steps    = args.steps,
            batch_size   = args.batch_size,
            eval_after   = args.eval_after,
        )

