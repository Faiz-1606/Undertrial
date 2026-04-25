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
  - bias-mitigation
---

# UndertriAI ⚖️

**OpenEnv-compliant RL training environment for Indian bail decision support.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-6366f1)](https://github.com/meta-pytorch/OpenEnv)
[![Live Demo](https://img.shields.io/badge/🤗_Space-Live_Demo-yellow)](https://huggingface.co/spaces/Draken1606/undertrial-ai)
[![Swagger](https://img.shields.io/badge/API-Swagger_Docs-green)](https://draken1606-undertrial-ai.hf.space/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray)](LICENSE)

> **[▶ Try the Live Demo](https://huggingface.co/spaces/Draken1606/undertrial-ai)** — click "Run Bail Assessment" to see the environment in action.

---

## The Problem

**76% of India's 5.7 lakh prisoners are undertrials** — unconvicted people awaiting bail hearings, many of whom cannot afford lawyers.

A subordinate court judge handles **80–100 bail hearings per day** — roughly **3 minutes per case**. In that window they must read the charge sheet, assess flight risk, evaluate custody duration against the statutory threshold, and check for parity with co-accused. In practice, outcomes are inconsistent and empirically biased against poor, lower-caste, and minority accused.

**This is not anecdotal — it is structural.** The Supreme Court in Satender Kumar Antil (2022) explicitly noted the crisis.

---

## What UndertriAI Does

UndertriAI is an **OpenEnv-compliant RL training environment** that teaches an LLM to reason like a careful, consistent, and unbiased judicial clerk:

1. Read the full charge sheet and arguments
2. Invoke legal tools (statutory eligibility, precedent lookup, surety assessment)
3. Produce a structured bail memo with explicit reasoning
4. Get rewarded based on agreement with real High Court decisions — with an **explicit bias penalty**

---

## Environment Design

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
| `compute_statutory_eligibility` | Calculate custody vs threshold for IPC/BNSS sections |
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

### 4-Stage Curriculum

| Stage | Focus | Cases |
|---|---|---|
| 1 | Landmark cases (clear-cut eligibility) | ~40 |
| 2 | Contested cases (murder, repeat offenders) | ~1,100 |
| 3 | Bias-reversal cases (HC overturning biased lower courts) | ~30 |
| 4 | BNSS schema drift (IPC → BNS remapping, 2023 reform) | ~50 |

---

## Theme 4 — Self-Improvement

UndertriAI qualifies for Theme 4 through three mechanisms:

**1. Adaptive Curriculum Promotion**
The environment tracks per-domain and per-stage performance using exponential
moving averages. When the agent demonstrates consistent improvement
(Stage 1 mean reward ≥ 0.65 over 20 episodes), it automatically promotes
to the next curriculum stage. This is visible in training logs as:
```
[SELF-IMPROVEMENT] Step 100: Promoted to Stage 2. Stage 1 mean reward: 0.710 → Stage 2 begins.
```

**2. Weakness-Targeted Episode Selection**
In adaptive mode, the episode selector identifies the crime type where the
agent performs worst and serves proportionally more cases from that domain.
As the agent improves on weak domains, the selection distribution shifts —
the environment continuously finds and targets new weaknesses.

| Selection | Weight | Mechanism |
|---|---|---|
| Weakest domain | 60% | EMA-tracked per-crime-type reward |
| Failure replay | 30% | Re-serve cases with reward < 0.40 |
| Exploration | 10% | Uniform random (prevent overfitting) |

**3. Synthetic Case Generation**
When the agent masters a domain (mean reward > 0.70), the environment
generates harder synthetic variants using 5 perturbation types:

| Perturbation | What it tests |
|---|---|
| Custody escalation | Custody 2 months below threshold — forces careful statutory computation |
| Co-accused conflict | Opposite bail outcome for co-accused — tests parity reasoning |
| Section ambiguity | IPC ↔ BNSS section swap — tests schema drift adaptability |
| Evidence reversal | Key witness retracted — tests flight risk reassessment |
| Surety complexity | Non-resident surety — tests condition appropriateness |

**Live Demo — Self-Improvement in Action**
```bash
# Start the server
python -m server.app

# In another terminal — start adaptive training
python training/train_grpo.py --adaptive --steps 50 --env_url http://localhost:8000
```

Monitor progress via:
```
GET /profile?session_id={id}
GET /adaptive_status
```

Watch stage promotions in the training log.

---

## Reward Function

```
R = 0.4 × outcome_match (gated by reasoning quality)
  + 0.2 × flight_risk_accuracy
  + 0.2 × statutory_accuracy
  + 0.2 × condition_appropriateness
  + 0.1 × reasoning_quality (bonus)
  + 0.05 × format_compliance (bonus)
  − 0.3 × bias_penalty
```

All components are **fully deterministic and rule-based** — no LLM-as-judge.

| Component | Signal | Details |
|---|---|---|
| **Outcome Match** | 0.0 / 0.8 / 1.0 | Exact, directional, or wrong vs HC decision — gated by `<think>` block |
| **Flight Risk** | 0–1 | Ordinal distance to ground-truth risk level |
| **Statutory** | 0–1 | IPC/BNSS threshold computation, direction-gated, NDPS Section 37 aware |
| **Conditions** | 0–1 | Appropriate bail conditions for crime/risk profile |
| **Reasoning Quality** | 0–1 | Anchoring + arithmetic + grounds specificity (10% bonus) |
| **Format Compliance** | 0–1 | XML tag adherence to system prompt (5% bonus) |
| **Bias Penalty** | −0.3 | Fired if parity argument ignored in bias-flagged cases |

### Anti-Reward-Hacking Design

- 7 independent reward signals (harder to simultaneously game all)
- `GenerationInspectionCallback` prints raw completions every 25 training steps
- Reasoning gate: no `<think>` block → outcome reward zeroed in Stage 2+
- Direction gate: wrong bail direction → statutory bonus capped
- Bias penalty operates as a separate signal, not folded into outcome
- Schema drift (Stage 4) tests adaptability, not pattern memorisation

---

## Training

Uses **GRPO** (Group Relative Policy Optimization) via TRL + Unsloth on `Qwen2.5-7B-Instruct`.

### Training Modes

| Mode | Command | Description |
|---|---|---|
| Single stage | `python training/train_grpo.py --stage 1 --steps 200` | Train on one stage |
| Curriculum | `python training/train_grpo.py --curriculum --steps 150` | Sequential 4-stage with trace harvesting |
| **Adaptive** | `python training/train_grpo.py --adaptive --steps 50` | **Theme 4** — self-directed with auto-promotion |

### Google Colab Training Walkthrough

```python
# ============================================================
# STEP 1 — Install dependencies (run in first cell)
# ============================================================
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps trl peft accelerate bitsandbytes xformers
!pip install -q openenv-core datasets

# ============================================================
# STEP 2 — Clone the repository
# ============================================================
!git clone https://github.com/Faiz-1606/Undertrial.git
%cd Undertrial

# ============================================================
# STEP 3 — Verify episodes are available
# ============================================================
import os
episodes_dir = "./data/episodes"
if not os.path.exists(episodes_dir):
    print("No episodes directory — will use built-in demo episodes")
else:
    for f in os.listdir(episodes_dir):
        if f.endswith('.jsonl'):
            count = sum(1 for _ in open(f"{episodes_dir}/{f}"))
            print(f"  {f}: {count} episodes")

# ============================================================
# STEP 4 — Option A: Single-stage training (quick, ~20 min on T4)
# ============================================================
!python training/train_grpo.py \
    --episodes_dir ./data/episodes \
    --stage 1 \
    --steps 200 \
    --batch_size 1 \
    --eval_after

# ============================================================
# STEP 4 — Option B: Curriculum training (full, ~90 min on T4)
# ============================================================
!python training/train_grpo.py \
    --episodes_dir ./data/episodes \
    --curriculum \
    --steps 150 \
    --batch_size 1

# ============================================================
# STEP 4 — Option C: Adaptive training (Theme 4, ~60 min on T4)
# (Requires server running — start in a background cell first)
# ============================================================
# Background cell: start the server
import subprocess
server = subprocess.Popen(
    ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
import time; time.sleep(5)  # Wait for server startup

# Then run adaptive training
!python training/train_grpo.py \
    --adaptive \
    --episodes_dir ./data/episodes \
    --steps 50 \
    --batch_size 1 \
    --env_url http://localhost:8000

# ============================================================
# STEP 5 — View results
# ============================================================
import json
# For single/curriculum:
results = json.load(open("./output/undertrial_grpo/results.json"))
print(json.dumps(results, indent=2))

# For adaptive:
# results = json.load(open("./output/undertrial_grpo/results_adaptive.json"))

# ============================================================
# STEP 6 — (Optional) Merge LoRA adapters for inference
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
print("Merged model saved to ./output/undertrial_merged")
```

### Training Architecture

```
Episode Dataset (JSONL)
        ↓
  Format as chat prompt
        ↓
  Qwen2.5 generates 4 rollouts (T4-safe)
        ↓
  XML parser extracts structured fields
        ↓
  server/reward.py scores each rollout (deterministic, offline)
        ↓
  GRPO updates model weights
        ↓
  [Theme 4] PerformanceTracker updates EMA per domain/stage
        ↓
  [Theme 4] AdaptiveSelector targets weakest domain
        ↓
  [Theme 4] CaseGenerator creates harder synthetic variants
        ↓
  [Theme 4] Auto-promote when stage EMA exceeds threshold
```

> **Design decision — Offline vs Environment-API scoring**: Training uses
> offline GRPO (completions are scored locally by `server/reward.py` without
> a live `/step` API call). This avoids ~200ms network latency per rollout,
> making a 200-step training run feasible on a free Colab T4 GPU in ~20 minutes.
> The alternative (`rollout_via_env_api()`) is implemented and available for
> production training where full environment interaction is required. See
> `training/train_grpo.py → rollout_via_env_api()` for the env-API path.

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
│   ├── undertrial_environment.py # Environment logic
│   ├── reward.py                 # 7-component deterministic reward
│   ├── dataset.py                # Curriculum-staged episode loader
│   ├── schema_drift.py           # IPC → BNSS remapping (Stage 4)
│   ├── performance_tracker.py    # [Theme 4] EMA-based performance profiling
│   ├── adaptive_selector.py      # [Theme 4] Weakness-targeted episode selection
│   └── case_generator.py         # [Theme 4] Synthetic case perturbation
├── training/
│   ├── train_grpo.py             # GRPO training (single/curriculum/adaptive)
│   └── UndertriAI_GRPO_Training.ipynb  # Colab notebook
├── data/
│   └── episodes/                 # 1,200 HC judgments across 4 stages
├── demo/
│   └── index.html                # Interactive demo UI
├── client.py                     # UndertriAIEnv HTTP client
├── models.py                     # Pydantic action/observation schemas
├── openenv.yaml                  # OpenEnv manifest
└── Dockerfile                    # HF Spaces deployment
```

---

## Data

1,200 Indian High Court bail judgments (2018–2024) processed into curriculum episodes covering:
- Delhi, Bombay, Allahabad, Madras, Kerala, and Calcutta HCs
- Crimes from IPC 420 (cheating) to IPC 302 (murder)
- Cases annotated with ground-truth outcome, flight risk, bias flags, and parity arguments

**Known dataset characteristics and their impact on training:**

| Characteristic | Value | Training effect |
|---|---|---|
| Episodes with `flight_risk = Medium` | ~72% | Model can earn ~0.86 flight risk score by always saying "Medium" — this is a weak learning signal. Stage 3 bias-reversal cases are specifically selected to force non-Medium reasoning. |
| Episodes with `custody_months = 6.0` | ~74% | Custody arithmetic is less discriminating since most cases share the same duration. The `reasoning_quality` sub-score partially compensates by rewarding exact numerical matches. |
| Episodes with `bias_flag = True` | ~1% (13 cases) | Rare but high-penalty (−0.3). The parity-argument signal (28% of cases) provides the main bias-mitigation training signal. |
| Episodes with empty `prosecution_arguments` | ~53% | No prosecution text available for half the dataset — the agent must reason from charge sheet and defence arguments alone. |

---

## Why This Matters

> *"Bail is the rule, jail is the exception."*  
> — Supreme Court of India, Satender Kumar Antil v. CBI (2022)

An RL-trained agent that consistently applies this principle — without being swayed by a defendant's name, religion, or economic status — could serve as a real-time consistency check for overburdened courts.

This isn't a tool to replace judges. It's a mirror that forces the system to confront its own inconsistencies.

---

## Team

Built for the **OpenEnv Hackathon, April 2026** 
