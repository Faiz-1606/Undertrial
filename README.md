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
| `POST` | `/step` | Submit a tool call or final memo |
| `GET` | `/state?session_id=...` | Inspect current episode state |
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
| `submit_memo` | **Terminal action** — submit final bail recommendation |

### 4-Stage Curriculum

| Stage | Focus | Cases |
|---|---|---|
| 1 | Landmark cases (clear-cut eligibility) | ~40 |
| 2 | Contested cases (murder, repeat offenders) | ~1,100 |
| 3 | Bias-reversal cases (HC overturning biased lower courts) | ~30 |
| 4 | BNSS schema drift (IPC → BNS remapping, 2023 reform) | ~50 |

---

## Reward Function

```
R = 0.4 × outcome_match
  + 0.2 × flight_risk_accuracy
  + 0.2 × statutory_accuracy
  + 0.2 × condition_appropriateness
  − 0.3 × bias_penalty
```

All components are **fully deterministic and rule-based** — no LLM-as-judge.

| Component | Signal | Details |
|---|---|---|
| **Outcome Match** | 0.0 / 0.8 / 1.0 | Exact, directional, or wrong vs HC decision |
| **Flight Risk** | 0–1 | Ordinal distance to ground-truth risk level |
| **Statutory** | 0–1 | IPC/BNSS section, sentence threshold, custody duration |
| **Conditions** | 0–1 | Appropriate bail conditions for crime/risk profile |
| **Bias Penalty** | −0.3 | Fired if parity argument ignored in bias-flagged cases |

### Anti-Reward-Hacking Design

- 5 independent reward signals (harder to simultaneously game all)
- `GenerationInspectionCallback` prints raw completions every 25 training steps
- Bias penalty operates as a separate signal, not folded into outcome
- Schema drift (Stage 4) tests adaptability, not pattern memorisation

---

## Training

Uses **GRPO** (Group Relative Policy Optimization) via TRL + Unsloth on `Qwen2.5-3B-Instruct`.

```bash
# Run with before/after eval and results.json
python training/train_grpo.py \
  --episodes_dir ./data/episodes \
  --stage 1 \
  --steps 200 \
  --eval_after
```

Or use the Colab notebook: [`training/UndertriAI_GRPO_Training.ipynb`](training/UndertriAI_GRPO_Training.ipynb)

### Training Architecture

```
Episode Dataset (JSONL)
        ↓
  Format as chat prompt
        ↓
  Qwen2.5 generates 6 rollouts
        ↓
  XML parser extracts structured fields
        ↓
  server/reward.py scores each rollout (deterministic)
        ↓
  GRPO updates model weights
        ↓
  GenerationInspectionCallback logs samples every 25 steps
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
│   ├── app.py                  # FastAPI routes
│   ├── undertrial_environment.py  # Environment logic
│   ├── reward.py               # 5-component deterministic reward
│   ├── dataset.py              # Curriculum-staged episode loader
│   └── schema_drift.py         # IPC → BNSS remapping (Stage 4)
├── training/
│   ├── train_grpo.py           # GRPO training script
│   └── UndertriAI_GRPO_Training.ipynb  # Colab notebook
├── data/
│   └── episodes/               # 1,200 HC judgments across 4 stages
├── demo/
│   └── index.html              # Interactive demo UI
├── client.py                   # UndertriAIEnv HTTP client
├── models.py                   # Pydantic action/observation schemas
└── Dockerfile                  # HF Spaces deployment
```

---

## Data

1,200 Indian High Court bail judgments (2018–2024) processed into curriculum episodes covering:
- Delhi, Bombay, Allahabad, Madras, Kerala, and Calcutta HCs
- Crimes from IPC 420 (cheating) to IPC 302 (murder)
- Cases annotated with ground-truth outcome, flight risk, bias flags, and parity arguments

---

## Why This Matters

> *"Bail is the rule, jail is the exception."*  
> — Supreme Court of India, Satender Kumar Antil v. CBI (2022)

An RL-trained agent that consistently applies this principle — without being swayed by a defendant's name, religion, or economic status — could serve as a real-time consistency check for overburdened courts.

This isn't a tool to replace judges. It's a mirror that forces the system to confront its own inconsistencies.

---

## Team

Built for the **OpenEnv Hackathon, April 2026** by **Faiz (Draken1606)**.
