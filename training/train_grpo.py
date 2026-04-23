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

import os, sys, json, re, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
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
    """Parse model's XML output into structured fields."""
    memo_block = extract_xml_field(output, "memo")
    if not memo_block:
        memo_block = output  # fallback: parse directly

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
    R = 0.4*outcome + 0.2*flight_risk + 0.2*statutory + 0.2*format - 0.3*bias
    """
    fmt   = reward_format(completions)
    out   = reward_outcome_match(completions, episode_batch)
    fr    = reward_flight_risk(completions, episode_batch)
    stat  = reward_statutory(completions, episode_batch)
    bias  = reward_no_bias(completions, episode_batch)

    rewards = []
    for f, o, r, s, b in zip(fmt, out, fr, stat, bias):
        total = 0.4*o + 0.2*r + 0.2*s + 0.2*f - 0.3*b
        rewards.append(round(max(0.0, total), 4))
    return rewards


# ============================================================
# CELL 5 — Dataset builder
# ============================================================

def load_episodes(episodes_dir: str, stage: int = 1) -> List[Dict]:
    path = Path(episodes_dir) / f"episodes_stage_{stage}.jsonl"
    if not path.exists():
        # Try the combined file
        path = Path(episodes_dir) / "episodes_all.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No episodes found in {episodes_dir}. Run data/prepare_dataset.py first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


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

def train(
    episodes_dir: str = "./data/episodes",
    output_dir:   str = "./output/undertrial_grpo",
    stage:        int = 1,
    max_steps:    int = 200,
    batch_size:   int = 4,
    grad_accum:   int = 4,
    lr:           float = 5e-6,
    max_seq_len:  int = 3072,
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

    # ── Trainer ──────────────────────────────────────────────
    trainer = GRPOTrainer(
        model          = model,
        processing_class = tokenizer,
        config         = config,
        train_dataset  = dataset,
        reward_funcs   = [reward_fn],
    )

    print("\nStarting GRPO training...")
    print(f"  Steps: {max_steps} | Batch: {batch_size} × {grad_accum} grad_accum")
    print(f"  Generations per prompt: 6 | KL beta: 0.01")
    print()

    trainer.train()

    # ── Save ─────────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model saved to {output_dir}")


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
    parser.add_argument("--baseline_only",action="store_true",
                        help="Only run baseline evaluation, skip training")
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
        )
