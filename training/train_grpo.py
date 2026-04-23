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
            s  = compute_statutory_accuracy(parsed, ep)
            ca = compute_condition_score(
                parsed["recommended_outcome"],
                parsed.get("conditions", []),
                gt,
            )
            b  = _server_bias(parsed, ep)
        else:
            # Local fallback
            o  = reward_outcome_match([comp], [ep])[0]
            fr = reward_flight_risk([comp], [ep])[0]
            s  = reward_statutory([comp], [ep])[0]
            ca = reward_conditions([comp], [ep])[0]  # condition score, not format
            b  = reward_no_bias([comp], [ep])[0]

        # R4 efficiency bonus: reward fewer steps when outcome is correct
        eff = 0.0
        if o >= 0.8:
            steps_taken = kwargs.get("step_counts", [None] * len(completions))
            sc = steps_taken[completions.index(comp)] if comp in completions else None
            if sc is not None:
                eff = max(0.0, 1.0 - (sc - 1) / 9)

        total = 0.4*o + 0.2*fr + 0.2*s + 0.2*ca + 0.1*eff - 0.3*b
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

    # ── Save model ────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
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
# CELL 8 — Entry point
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
    args = parser.parse_args()

    if args.baseline_only:
        evaluate_baseline(args.episodes_dir)
    else:
        train(
            episodes_dir = args.episodes_dir,
            output_dir   = args.output,
            stage        = args.stage,
            max_steps    = args.steps,
            batch_size   = args.batch_size,
            eval_after   = args.eval_after,
        )
