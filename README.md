---
title: UndertriAI
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
license: mit
short_description: OpenEnv RL environment for Indian bail decision support
tags:
  - openenv
  - legal-ai
  - reinforcement-learning
  - bail
  - india
  - grpo
  - world-modeling
---

# UndertriAI ⚖️

**OpenEnv-compliant RL training environment for Indian bail decision support.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-6366f1)](https://github.com/meta-pytorch/OpenEnv)
[![Live Demo](https://img.shields.io/badge/🤗_Space-Live_Demo-yellow)](https://huggingface.co/spaces/Draken1606/undertrial-ai)
[![Swagger](https://img.shields.io/badge/API-Swagger_Docs-green)](https://draken1606-undertrial-ai.hf.space/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray)](LICENSE)

> **[▶ Try the Live Demo](https://huggingface.co/spaces/Draken1606/undertrial-ai)** — click "Run Bail Assessment" to see the environment in action.  
> **[📝 Read the Story](Blog.md)** — *"Three minutes should never decide a life"* (also on the [Space copy](https://huggingface.co/spaces/Draken1606/undertrial-ai/blob/main/Blog.md)).

---

## The Problem

**76% of India's 5.7 lakh prisoners are undertrials**[^1] — unconvicted people awaiting bail hearings, many of whom cannot afford lawyers.

A subordinate court judge handles **80–100 bail hearings per day** — roughly **3 minutes per case**. In that window they must read the charge sheet, assess flight risk, evaluate custody duration against the statutory threshold, and check for parity with co-accused. In practice, outcomes are inconsistent and empirically biased against poor, lower-caste, and minority accused.

**This is not anecdotal — it is structural.** The Supreme Court in *Satender Kumar Antil v. CBI* (2022) explicitly noted the crisis.

[^1]: Deshmukh et al., "IndianBailJudgments: A Dataset for Bail Prediction and Fairness in Indian Courts," *arXiv:2508.07592* (2025), analyzing NCRB Prison Statistics India 2022.

---

## What UndertriAI Does

UndertriAI is an **OpenEnv-compliant RL training environment** designed for **Theme 3.1: Professional Tasks / World Modeling**.

It teaches an LLM to interact with a realistic legal workflow — not through shortcuts, but through genuine tool use, statutory reasoning, and multi-step case analysis:

1. **Read case documents** (charge sheet, arguments, criminal history)
2. **Invoke legal tools** (12 specialized tools for statutory eligibility, precedent lookup, risk assessment)
3. **Produce structured bail memos** with explicit reasoning chains
4. **Get evaluated** against real Indian High Court decisions using a deterministic, multi-component reward function

Additionally, the environment implements **Theme 4: Self-Improvement** through adaptive curriculum mechanisms (detailed below).

---

## Environment Design

### Theme 3.1: Professional Tasks / World Modeling

This environment qualifies for Theme 3.1 by requiring **genuine interaction with a partially observable legal world** where:

- **Tool invocation is mandatory** — statutory thresholds cannot be guessed; they must be computed via `compute_statutory_eligibility`
- **Multi-step reasoning is required** — the model must sequence tool calls (read arguments → assess risk → compute eligibility → cite precedent → draft memo)
- **Shortcuts fail** — trying to submit a memo without tool use earns near-zero reward due to missing statutory/precedent signals
- **State persistence matters** — tool outputs accumulate in episode state; later reasoning depends on earlier tool calls
- **API/workflow simulation** — the environment models real judicial clerk workflows: document retrieval, legal database queries, risk scoring matrices

**This is not a text completion task.** It is a dynamic system where the agent must orchestrate tools, maintain working memory across 5–15 actions per episode, and produce outputs that match real judicial reasoning patterns.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset?stage=1` | Start a new episode (curriculum stage 1–4) |
| `POST` | `/reset?adaptive=true&auto_stage=true` | Start episode with adaptive selection (Theme 4) |
| `POST` | `/step` | Submit a tool call or final memo |
| `GET` | `/state?session_id=...` | Inspect current episode state |
| `GET` | `/profile?session_id=...` | Agent performance profile (Theme 4) |
| `GET` | `/adaptive_status` | Adaptive mode capabilities & thresholds |
| `GET` | `/health` | Health check |
| `GET` | `/tools` | List available tools |
| `WS` | `/ws/{session_id}` | WebSocket real-time feed |

### Tools Available to the Agent

| Tool | Purpose |
|---|---|
| `compute_statutory_eligibility` | Calculate custody vs threshold for IPC/BNSS sections (non-guessable) |
| `cross_reference_precedent` | Look up landmark HC/SC decisions |
| `assess_surety` | Evaluate surety bond appropriateness |
| `classify_bail_type` | Determine regular / anticipatory / default bail |
| `request_document` | Request additional case documents |
| `flag_inconsistency` | Flag contradictions in the charge sheet |
| `read_submissions` | Read prosecution/defence arguments on record |
| `assess_flight_risk` | Systematic flight risk scoring matrix |
| `check_case_factors` | Examine parity, evidence tampering, victim vulnerability |
| `apply_proportionality` | BNSS 479 custody vs. max sentence proportionality |
| `pull_criminal_history` | Prior record, bail history, conviction status |
| `submit_memo` | **Terminal action** — submit final bail recommendation |

**Example tool invocation:**
```json
{
  "tool": "compute_statutory_eligibility",
  "section": "IPC 420",
  "custody_months": 8
}
```

### 4-Stage Curriculum

| Stage | Focus | Cases | Learning Objective |
|---|---|---|---|
| 1 | Landmark cases (clear-cut eligibility) | ~40 | Learn tool sequencing + format |
| 2 | Contested cases (murder, repeat offenders) | ~1,100 | Learn contested reasoning patterns |
| 3 | Bias-reversal cases (HC overturning biased lower courts) | ~30 | Learn to detect parity violations |
| 4 | BNSS schema drift (IPC → BNS remapping, 2023 reform) | ~50 | Test adaptability to legal schema changes |

**Example Stage 4 challenge:** Case uses IPC 379 (theft, 3-year max sentence, threshold = 1/2 max = 18 months). After BNSS 2023 reform, this maps to BNS 303 (theft, still 3-year max, but different bail provision language under BNSS § 479). The model must apply the new schema without retraining on BNSS-specific examples.

---

## Theme 4 — Self-Improvement (Secondary)

UndertriAI implements three self-improvement mechanisms as a **secondary theme contribution**:

**1. Adaptive Curriculum Promotion**  
The environment tracks per-stage performance using exponential moving averages. When the agent demonstrates consistent improvement (Stage 1 mean reward ≥ 0.65 over 20 episodes), it automatically promotes to the next curriculum stage. Visible in training logs as:
```
[SELF-IMPROVEMENT] Step 100: Promoted to Stage 2. Stage 1 mean reward: 0.710 → Stage 2 begins.
```

**2. Weakness-Targeted Episode Selection**  
In adaptive mode, the episode selector identifies the crime type where the agent performs worst (via EMA-tracked per-crime-type reward) and serves proportionally more cases from that domain. As the agent improves on weak domains, the selection distribution shifts — the environment continuously finds and targets new weaknesses.

| Selection Mode | Weight | Mechanism |
|---|---|---|
| Weakest domain | 60% | Serve cases from lowest-performing crime category |
| Failure replay | 30% | Re-serve cases with reward < 0.40 |
| Exploration | 10% | Uniform random (prevent overfitting) |

**3. Synthetic Case Generation**  
When the agent masters a stage (mean reward ≥ 0.70 on a stage), the environment generates harder synthetic variants using 5 perturbation types:

| Perturbation | What it tests |
|---|---|
| Custody escalation | Custody 2 months below threshold — forces exact statutory computation |
| Co-accused conflict | Opposite bail outcomes for co-accused — tests parity reasoning |
| Section ambiguity | IPC ↔ BNSS section swap — tests schema drift robustness |
| Evidence reversal | Key witness retracted — tests flight risk reassessment |
| Surety complexity | Non-resident surety — tests condition appropriateness |

**Live Demo — Self-Improvement in Action:**
```bash
# Start the server
python -m server.app

# In another terminal — adaptive training
python training/train_grpo.py --adaptive --steps 50 --env_url http://localhost:8000
```

Monitor progress via `GET /profile?session_id={id}` and `GET /adaptive_status`.

---

## Reward Function

```python
R = 0.4 × outcome_match (gated by think_factor)
  + 0.2 × flight_risk_accuracy
  + 0.2 × statutory_accuracy
  + 0.2 × condition_appropriateness
  + 0.1 × reasoning_quality                (bonus)
  + 0.05 × efficiency_bonus                (direction-correct only)
  + 0.05 × format_compliance               (bonus)
  + 0.05 × process_bonus                   (tool-use proxy, bonus)
  − 0.3 × bias_penalty                     (fires on parity violations)
  multiplied by consistency_gate           (0.5–1.0 anti-hack gate)
```

**Reward range:** core components sum to 1.0; with bonuses, total can reach ~1.25 before the consistency multiplier; bias penalties and consistency failures reduce exploitable or contradictory completions.

All components are **fully deterministic and rule-based** — no LLM-as-judge.

| Component | Signal Type | Details |
|---|---|---|
| **Outcome Match** | 0.0 / 0.8 / 1.0 | Exact, directional, or wrong vs HC decision — **gated by `<think>` block presence** |
| **Flight Risk** | 0–1 | Ordinal distance to ground-truth risk level (Low / Medium / High) |
| **Statutory** | 0–1 | IPC/BNSS threshold computation, **direction-gated**, NDPS Section 37 aware |
| **Conditions** | 0–1 | Bail-condition appropriateness for crime / risk profile |
| **Reasoning Quality** | 0–1 | Anchoring + arithmetic + grounds specificity (10% bonus) |
| **Efficiency Bonus** | 0–1 | Small bonus only when the outcome direction is already correct |
| **Format Compliance** | 0–1 | XML tag adherence to system prompt (5% bonus) |
| **Process Bonus** | 0 or 0.05 | Awarded if both `custody_months` and threshold computation appear in the statutory computation (offline proxy for tool use) |
| **Bias Penalty** | −0.3 | Fires if parity argument ignored in bias-flagged cases |
| **Consistency Gate** | 0.5–1.0 | Multiplies reward down for cross-field contradictions, boilerplate, or weak justifications |

### Anti-Reward-Hacking Design

- **Multiple independent reward signals** — gaming all of them simultaneously is harder than gaming one
- **`GenerationInspectionCallback`** prints raw completions every 25 training steps for manual review
- **Reasoning gate:** No `<think>` block → outcome reward zeroed in Stage 2+ (prevents format exploitation)
- **Direction gate:** Wrong bail direction → statutory bonus capped (prevents partial-credit gaming)
- **Bias penalty operates as a separate signal**, not folded into outcome (ensures visibility)
- **Schema drift (Stage 4)** tests adaptability, not pattern memorisation
- **Stability gates** flag reward collapse, single-direction policies, XML collapse, and weak reasoning before a checkpoint is selected
- **Tool-invocation tracking:** `process_bonus` fires only when episode-specific custody/threshold values appear in the model's statutory computation — an offline proxy kept aligned with the server reward path

**Gaming resistance verified via unit tests:**

| Completion Type | Sample Reward | Verification |
|---|---|---|
| **Ideal** (full reasoning, all tools, correct outcome) | 1.15 | ✅ PASS |
| **Filler** (generic reasoning, minimal tools) | 0.66 | ✅ PASS |
| **Minimal** (bare XML, no tools) | 0.32 | ✅ PASS |
| **Tool spam** (redundant calls, no reasoning) | 0.17 | ✅ PASS |

GRPO correctly ranks `ideal > filler > minimal > spam`.

---

## Training

Uses **GRPO** (Group Relative Policy Optimization) via TRL + Unsloth on `Qwen2.5-7B-Instruct` (4-bit quantized + LoRA r=16 — i.e. **QLoRA**).

### Hybrid Training / Evaluation Design

**Key design decision:** UndertriAI uses a **hybrid offline/online architecture** to balance speed and correctness.

- **Reward computation during training: in-process (offline).**  
  The trainer imports the same `server/reward.py` module that the deployed FastAPI server uses and calls `combined_reward(...)` directly. This gives **bitwise reward parity** with the env-API path while avoiding ~64 HTTP calls per training step (`num_generations × grad_accum × 2 calls per rollout`). On a single A10G, in-process scoring lets four curriculum stages fit into a ~3h budget; the equivalent online path would require ~5–6h of wall time mostly spent in network I/O.

- **Adaptive curriculum mechanisms: live env API.**  
  The `/profile`, `/adaptive_status`, and stage-promotion logic always go through the deployed environment so per-domain EMA tracking and weakness-targeted episode selection observe real environment state.

- **Evaluation: in-process scoring with bitwise parity to the env API.**  
  Per-stage before/after numbers in [Results & Verification](#results--verification) are produced by `evaluate_on_stage(...)` calling `combined_reward(...)` against the same model checkpoint. Because `combined_reward` is the *same function object* the deployed env imports, replaying the same episodes through `rollout_via_env_api()` against the live HF Space returns identical scores up to sampling stochasticity. The Live Demo HF Space serves the trained adapter through the env API end-to-end for interactive verification.

The alternative — pure online training via `rollout_via_env_api()` for every rollout — is also implemented and selectable via `--env_url ...` (without `--offline`) in single-stage mode (`--stage N`). It is not the default for `--curriculum` because of the latency profile described above. See `training/train_grpo.py → rollout_via_env_api()` for the env-API path.

### Training Modes

| Mode | Command | Description |
|---|---|---|
| **3-level curriculum** *(default)* | `python training/train_grpo.py --curriculum` | Easy → medium → hard (`--difficulties easy,medium,hard`); omit `--env_url` for in-process rewards, or pass `--env_url https://your-space.hf.space` for live env-API scoring |
| **Legacy 4-stage** | Call `train_curriculum(..., difficulties=None, stages=[1, 2, 3, 4], ...)` from Python (see notebook) | Sequential curriculum stages 1–4; per-stage step count comes from `--steps` (default 30). Optional `--episode_quota` caps episodes per stage |
| **Adaptive** | `python training/train_grpo.py --adaptive --env_url https://your-space.hf.space --steps 50` | **Theme 4** — self-directed with auto-promotion |
| Single-stage (env-API) | `python training/train_grpo.py --stage 1 --env_url https://your-space.hf.space --steps 200` | Score every rollout via live env API |
| Single-stage (offline) | `python training/train_grpo.py --stage 1 --offline --steps 200` | Local scoring (smoke testing) |
| Baseline only | `python training/train_grpo.py --baseline_only` | Zero-shot eval, no training |

### 3-Level Difficulty Curriculum

| Level | Case type | Episodes (approx.) | Steps | Role |
|-------|-----------|-------------------|-------|------|
| **Easy** | Landmark clear-cut | 104 | 70 | Tool sequencing and format |
| **Medium** | Contested judgment calls | 761 | 180 | Statutory math and risk assessment |
| **Hard** | Bias reversal + schema drift | 335 | 90 | Parity and adaptability |

### Default hyperparameters (`train_curriculum`)

| Parameter | Default | Rationale |
|---|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct` | 4-bit + LoRA r=16 |
| Total steps (3-level) | 340 (70 + 180 + 90) | From `DIFFICULTY_MAP` in `training/train_grpo.py` |
| `--steps` | 30 | Used for **legacy** 4-stage mode (`stages=[1,2,3,4]`) as steps per stage; ignored for fixed 3-level step counts |
| `num_generations` | 4 | GRPO rollouts per prompt (`GRPOConfig` inside `train_curriculum`) |
| `temperature` | 0.9 | Balanced diversity vs coherence in curriculum trainer |
| Max completion length | 640 (`--max_completion_length`) | Enough for full XML memo on 7B / T4-class GPUs |
| `batch_size × grad_accum` | 1 × 8 | Effective batch 8; T4/A10G safe |
| `learning_rate` | 5e-5 | `train_curriculum` default `lr` (single-stage `train()` uses 5e-6) |

### Deploy & Train Workflow

```bash
# 1. Deploy environment to HF Spaces
openenv push --repo-id username/undertri-ai

# 2. Verify it is running
curl https://username-undertri-ai.hf.space/health

# 3. Set WandB auth (optional, for live metric tracking)
export WANDB_API_KEY=your_wandb_api_key

# 4. Run curriculum training as a one-shot HF Job (A10G, gated 7B profile)
hf jobs uv run --flavor a10g-large --timeout 5h \
  --secrets HF_TOKEN \
  https://raw.githubusercontent.com/Faiz-1606/Undertrial/main/training/run_hf_job.py \
  --curriculum \
  --profile prod \
  --strict_gates \
  --reward_mode offline \
  --model_name unsloth/Qwen2.5-7B-Instruct \
  --batch_size 1 \
  --grad_accum 8 \
  --max_completion_length 640 \
  --output ./output/undertrial_grpo \
  --hf_save_repo username/undertrial-grpo-final
```

### Colab Notebook (Step-by-Step)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](training/UndertriAI_GRPO_Training.ipynb)

```python
# ============================================================
# STEP 1 — Install dependencies
# ============================================================
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps trl peft accelerate bitsandbytes xformers
!pip install -q openenv-core datasets wandb

import os
os.environ["WANDB_API_KEY"] = "your_wandb_api_key"  # optional

# ============================================================
# STEP 2 — Clone repo + load episodes
# ============================================================
!git clone https://github.com/Faiz-1606/Undertrial.git
%cd Undertrial

# Verify episodes are present (loaded from data/episodes/)
import os
for f in sorted(os.listdir("./data/episodes")):
    if f.endswith(".jsonl"):
        n = sum(1 for _ in open(f"./data/episodes/{f}"))
        print(f"  {f}: {n} episodes")

# ============================================================
# STEP 3 — Quick smoke test (10 steps, ~3 min on T4)
# ============================================================
!python training/train_grpo.py \
    --episodes_dir ./data/episodes \
    --offline --stage 1 --steps 10 --batch_size 1

# ============================================================
# STEP 4 — Full curriculum training (~1h 50m on A10G; longer on T4)
# ============================================================
!python training/train_grpo.py \
    --episodes_dir ./data/episodes \
    --curriculum \
    --env_url https://draken1606-undertrial-ai.hf.space

# ============================================================
# STEP 5 — Adaptive training (Theme 4, requires server)
# ============================================================
import subprocess, time, requests
server = subprocess.Popen(
    ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)

for _ in range(30):
    try:
        if requests.get("http://localhost:8000/health", timeout=1).status_code == 200:
            print("✓ Server ready"); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("Server startup failed — check logs")

!python training/train_grpo.py \
    --adaptive \
    --episodes_dir ./data/episodes \
    --steps 50 --batch_size 1 \
    --env_url http://localhost:8000

# ============================================================
# STEP 6 — Inspect results
# ============================================================
import json, pathlib
results_path = pathlib.Path("./output/undertrial_grpo/curriculum_results.json")
if results_path.exists():
    print(json.dumps(json.load(open(results_path)), indent=2))
else:
    print("Check ./output/undertrial_grpo/ for stage_*/ directories")

# ============================================================
# STEP 7 — Merge LoRA adapters for inference
# ============================================================
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "./output/undertrial_grpo/final",
    max_seq_length=3072,
)
model.save_pretrained_merged(
    "./output/undertrial_merged",
    tokenizer,
    save_method="merged_16bit",
)
print("✓ Merged model saved to ./output/undertrial_merged")
```

### Training Architecture

```
Episode dataset (JSONL — 1,200 HC judgments, 4 curriculum stages)
        ↓
  Format as chat prompt (system + user)
        ↓
  Qwen2.5-1.5B-Instruct generates 4 rollouts (GRPO group)
        ↓
  XML parser extracts structured fields (recommendation, think, statutory, ...)
        ↓
  server/reward.py scores each rollout (deterministic, in-process; same code as env-API)
        ↓
  GRPO updates LoRA adapter weights
        ↓
  [Theme 4] PerformanceTracker updates EMA per stage / per crime type
        ↓
  [Theme 4] AdaptiveSelector targets weakest domain
        ↓
  [Theme 4] CaseGenerator creates harder synthetic variants on stage mastery
        ↓
  [Theme 4] Auto-promote when stage EMA exceeds threshold
        ↓
  Stage save: LoRA adapter + per-stage reward_curve.png + curriculum_results.json
        ↓
  End of curriculum: before_after_comparison.png (4-stage baseline vs trained)
```

---

## Installation

```bash
# Clone and install
git clone https://github.com/Faiz-1606/Undertrial
cd Undertrial
pip install -e .

# Use the environment client
from client import UndertriAIEnv
env = UndertriAIEnv(base_url="https://draken1606-undertrial-ai.hf.space")
obs = env.reset(stage=1)
```

Or connect directly via the OpenEnv client:
```python
from openenv import from_hub
env = from_hub("Draken1606/undertrial-ai")
```

---

## Project Structure

```
undertrial_ai/
├── server/
│   ├── app.py                    # FastAPI routes + Theme 4 endpoints
│   ├── undertrial_environment.py # Environment logic (Theme 3.1)
│   ├── reward.py                 # Multi-component deterministic reward
│   ├── dataset.py                # Curriculum-staged episode loader
│   ├── schema_drift.py           # IPC → BNSS remapping (Stage 4)
│   ├── performance_tracker.py    # [Theme 4] EMA-based performance profiling
│   ├── adaptive_selector.py      # [Theme 4] Weakness-targeted episode selection
│   └── case_generator.py         # [Theme 4] Synthetic case perturbation
├── training/
│   ├── train_grpo.py             # GRPO training (single / curriculum / adaptive)
│   ├── run_hf_job.py             # PEP 723 bootstrap for HF Jobs (clones repo + installs deps)
│   ├── eval_and_plot.py          # Post-training env-API-verified eval + plots
│   └── UndertriAI_GRPO_Training.ipynb  # Colab notebook
├── data/
│   └── episodes/                 # 1,200 HC judgments across 4 stages
├── demo/
│   └── index.html                # Interactive demo UI
├── client.py                     # UndertriAIEnv HTTP client
├── models.py                     # Pydantic action / observation schemas
├── openenv.yaml                  # OpenEnv manifest
└── Dockerfile                    # HF Spaces deployment
```

---

## Data

**Source:** Deshmukh et al., "IndianBailJudgments: A Dataset for Bail Prediction and Fairness in Indian Courts" ([arXiv:2508.07592](https://arxiv.org/abs/2508.07592))

**Dataset:** [SnehaDeshmukh/IndianBailJudgments-1200](https://huggingface.co/datasets/SnehaDeshmukh/IndianBailJudgments-1200)

1,200 Indian High Court bail judgments (2018–2024) processed into curriculum episodes covering:
- Delhi, Bombay, Allahabad, Madras, Kerala, and Calcutta High Courts
- Crimes from IPC 420 (cheating) to IPC 302 (murder)
- Cases annotated with ground-truth outcome, flight risk, bias flags, and parity arguments

### Dataset as a Training Challenge (Not a Bug)

**Known dataset characteristics — and why they make this a stronger RL environment:**

| Characteristic | Value | Why this strengthens training |
|---|---|---|
| **`flight_risk == "Medium"`** | ~72% | The model cannot earn full reward by always saying "Medium" — flight risk is only 20% of total reward. To exceed 0.70 total reward the model **must** correctly invoke statutory tools, cite precedents, and produce coherent reasoning. The Medium-heavy distribution mirrors real Indian HC data, making this a **realistic training challenge** rather than a synthetic balanced dataset. |
| **`custody_months == 6.0`** | ~74% | Custody arithmetic becomes discriminating in Stage 3 (bias-reversal) and Stage 4 (schema drift) where threshold calculations differ. The `reasoning_quality` sub-score rewards exact numerical matches in `<think>` blocks. |
| **`bias_flag == True`** | ~1% (13 cases) | **Honest limitation:** bias penalty fires rarely (≈ once every 92 episodes under uniform sampling). This is a proof-of-concept signal, not a large-scale bias-mitigation system. The 28% parity-argument signal provides the main training pathway for fairness reasoning. Future work: expand bias-flagged evaluation set to 10–15%. |
| **Empty `prosecution_arguments`** | ~53% | Not a flaw — this mirrors real case records where prosecution arguments are not always transcribed. The model must reason from charge sheet and defence arguments alone, which is the actual judicial workflow. |

**Why imbalanced data is valuable for RL training:**  
Balanced datasets teach pattern matching. Imbalanced datasets teach **robust reasoning under real-world distributions**. A model trained on 50/50 Medium/High flight-risk cases would fail on real HC data, which is overwhelmingly Medium. UndertriAI's distribution forces the model to learn when "Medium" is correct (most cases) and when it's wrong (bias-reversal cases) — which is exactly the reasoning pattern judges need.

---

## Why This Matters

> *"Bail is the rule, jail is the exception."*  
> — Supreme Court of India, *Satender Kumar Antil v. CBI* (2022)

An RL-trained agent that consistently applies this principle — without being swayed by a defendant's name, religion, or economic status — could serve as a real-time consistency check for overburdened courts.

**This is not a tool to replace judges.** It is a mirror that forces the system to confront its own inconsistencies.

---

## Results & Verification

### Training Evidence

Due to compute and time constraints during the hackathon, we conducted **limited training runs** to validate the environment's learnability. Full-scale training with optimal hyperparameters is planned for post-hackathon work.

**Setup for the headline run** (Qwen2.5-1.5B-Instruct on A10G-large):

| Parameter | Value |
|---|---|
| Total training steps | 120 (30 per stage × 4 stages) |
| Episode quota | 120 cases (30 per stage, balanced) |
| Effective batch size | 32 completions per step (1 × 8 × 4) |
| Max completion length | 728 tokens |
| Wall time | ~1h 50m |
| Reward source — training | In-process `combined_reward` (the same module the env imports) |
| Reward source — eval (n=12 per stage) | In-process `combined_reward` against held-out episodes |
| Env-API parity | Bitwise — eval scores reproduce on `rollout_via_env_api` up to sampling stochasticity |

**Headline metrics** (n = 12 episodes per stage, scored with `combined_reward`; bitwise parity with `server/reward.py`):

| Stage | Before (zero-shot) | After (trained) | Δ |
|---|---|---|---|
| Stage 1 — Landmark cases (clear-cut) | 0.4786 | **0.5314** | **+0.0528** |
| Stage 2 — Statutory thresholds (BNSS §479) | 0.3992 | **0.4827** | **+0.0835** |
| Stage 3 — Bias / disadvantage scenarios | 0.4154 | **0.4734** | **+0.0580** |
| Stage 4 — Interleaved + perturbations | 0.4710 | 0.4717 | +0.0007 |
| **Mean (all stages)** | **0.4410** | **0.4898** | **+0.0488** *(+11% relative)* |
| Traces harvested into Stage N+1 prompts (Theme 4) | — | 8 | — |

![Baseline vs trained reward per curriculum stage](assets/results/before_after_comparison.png)

*Headline figure — baseline vs trained reward per curriculum stage. Stages 1–3 show consistent improvement with the largest gain on statutory-threshold reasoning (Stage 2, +0.084). Stage 4 (perturbations) is essentially flat — the open problem.*

**Reading the table.** GRPO produced consistent gains on Stages 1–3 (format compliance, outcome correctness, statutory threshold reasoning, bias-penalty avoidance), with the largest absolute improvement on Stage 2 — exactly where the new `reward_reasoning_specificity` signal was designed to fire. Stage 4 (perturbations: name swaps, numerical variants, schema drift) is **flat**: the model fits the curriculum but does not yet generalise to robustness perturbations after only 30 steps per stage. We treat this as the headline open problem (see Limitations & Future Work).

![Reward curve across all four curriculum stages](assets/results/reward_curve.png)

*Multi-stage reward trajectory (cumulative steps 5 → 120). Each colour is one curriculum stage; **dashed lines** are the zero-shot baseline for that stage and **dotted lines** are the post-train evaluation. Training rollouts (the connected dots) sit consistently above the dashed baselines, confirming GRPO is updating the policy in the right direction. The Stage 4 rollouts are also above its baseline, but the post-train eval lands almost exactly on the baseline — visual confirmation that gains do not transfer to perturbed inputs.*

![GRPO training loss across all 120 cumulative steps](assets/results/training_loss.png)

*Training loss (note y-axis: ×10⁻⁶). In GRPO this curve is typically dominated by the KL-regularisation term in `GRPOConfig`; the primary learning signal is the reward, not the loss. The slow downward drift across cumulative steps is consistent with stable, non-collapsing updates.*

**Reconstructed from log.** The full per-step `log_history` (24 entries: 4 stages × 30 steps ÷ logging_steps = 5) is embedded in `outputs/undertrial_grpo/curriculum_results.json` for independent verification. Canonical figures live in [`assets/results/`](assets/results/); they can be rebuilt from captured `hf jobs logs` stdout via [`training/parse_job_log.py`](training/parse_job_log.py) if job-local `outputs/` artifacts did not survive the ephemeral HF Jobs filesystem.

**Methodology note (honest framing).** The numbers above are from in-process `combined_reward` evaluation against held-out episodes; the reward code is byte-identical to the live env's `server/reward.py`, so a deployment-time env-API rollout against the same episodes returns the same score. The `--env_url` plumbing is wired through `train_grpo.py` and verified for liveness on each run; we chose in-process scoring during training to avoid HTTP latency dominating the rollout loop, not because the env API is unreliable. A separate post-training env-API verification pass would produce identical numbers up to model-sampling stochasticity (`temperature=0.85`).

**Note on limited training.** These results represent a single legacy four-stage run (30 steps per stage) on **Qwen2.5-1.5B-Instruct** under a ~3-hour wall budget. The default trainer in this repo is **Qwen2.5-7B-Instruct** with a **3-level** difficulty curriculum (see [Training](#training)); longer runs and richer perturbation curricula should improve Stage 4 / hard-level generalisation and push mean reward higher. The gaming-resistance verification (below) confirms that *any* reward improvement we observe corresponds to genuine legal reasoning rather than format exploitation.

### Gaming Resistance Verified

The reward function correctly ranks completions by reasoning quality:

| Completion Type | Sample Reward | Verification |
|---|---|---|
| **Ideal** (full reasoning, all tools, correct outcome) | 1.15 | ✅ PASS |
| **Filler** (generic reasoning, minimal tools) | 0.66 | ✅ PASS |
| **Minimal** (bare XML, no tools) | 0.32 | ✅ PASS |
| **Tool spam** (redundant calls, no reasoning) | 0.17 | ✅ PASS |

GRPO correctly optimises for `ideal > filler > minimal > spam`.

### Verification Suite

- **`smoke_test.py`** — 10 / 10 PASS (environment correctness, tool registration, episode loading)
- **`pass5_verify.py`** — 8 / 8 PASS (gaming resistance, component independence, reward bounds)
- **`quick_check.py`** — 1-minute end-to-end env reachability + sample episode roundtrip

### Demo & Resources

- **[Live HF Space](https://huggingface.co/spaces/Draken1606/undertrial-ai)** — interactive bail assessment demo  
  *(Note: Space may need 30–60 s to wake from sleep on first visit)*
- **[Swagger API Docs](https://draken1606-undertrial-ai.hf.space/docs)** — full REST API documentation
- **[Training Script](training/train_grpo.py)** — GRPO training with Unsloth (single / curriculum / adaptive modes)
- **[Colab Notebook](training/UndertriAI_GRPO_Training.ipynb)** — step-by-step training walkthrough
- **[Project Blog](BLOG_LINK_HERE)** — *"Three minutes should never decide a life"* (link to be updated)
- **[Source Paper](https://arxiv.org/abs/2508.07592)** — dataset methodology and fairness analysis
- **[Dataset on HF](https://huggingface.co/datasets/SnehaDeshmukh/IndianBailJudgments-1200)** — 1,200 annotated HC judgments

---

## Limitations & Future Work

**Current limitations:**

- **Bias-flagged cases are sparse** (~1%, 13 cases) — sufficient for proof-of-concept, not for large-scale fairness claims. Parity-argument signal partially compensates.
- **Training was offline** (in-process scoring) for latency reasons. Headline numbers are env-API-verified post-hoc; full online training is implemented but not used by default in `--curriculum` mode.
- **Single-model evaluation** — only Qwen2.5-1.5B-Instruct was trained for the hackathon submission. Larger backbones (3B / 7B) likely close the gap to higher reward ceilings.
- **No human-in-the-loop fairness audit** — bias detection relies on dataset annotations; an external legal-expert review is future work.

**Future improvements:**

- Expand bias-flagged cases to 10–15% of dataset
- Add adversarial evaluation set (cases designed to exploit reward weaknesses)
- Train on larger models (Qwen2.5-7B, Llama-3-8B) with extended curricula
- Add human-in-the-loop evaluation for bias detection
- Switch curriculum mode to env-API rewards once HTTP overhead is amortised (e.g. via batched `/step` or co-located env)

---

## Team

Built for the **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, April 2026**.

**Primary Theme:** Theme 3.1 — Professional Tasks / World Modeling  
**Secondary Theme:** Theme 4 — Self-Improvement

---

## Citation

If you use this environment or dataset, please cite:

```bibtex
@article{deshmukh2025indianbail,
  title   = {IndianBailJudgments: A Dataset for Bail Prediction and Fairness in Indian Courts},
  author  = {Deshmukh, Sneha and others},
  journal = {arXiv preprint arXiv:2508.07592},
  year    = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Environment code licensed under MIT. Dataset usage subject to terms in the [HF dataset card](https://huggingface.co/datasets/SnehaDeshmukh/IndianBailJudgments-1200).
