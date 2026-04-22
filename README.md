# UndertriAI ⚖️

**OpenEnv-compliant RL training environment for Indian bail decision support.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces/Draken1606/undertrial-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem

76% of India's 5.7 lakh prisoners are **undertrials** — unconvicted people awaiting bail hearings.  
A subordinate court judge handles **80–100 bail hearings per day** — roughly **3 minutes per case**.

In that window a judge must read the charge, assess flight risk, evaluate custody duration, consider financial standing, and cross-reference precedent. In practice, decisions are heavily influenced by whichever lawyer speaks loudest.

**Result:** Poor undertrials remain incarcerated for years on offences carrying 2-year maximum sentences. This is not anecdotal — it is a structural failure at scale.

---

## What UndertriAI Does

UndertriAI trains an LLM agent to do what a thorough senior judge would do:

- Read the full case
- Apply the statute (IPC and BNSS 2023)
- Cross-reference landmark precedents
- Produce a **consistent recommendation regardless of who is asking**

---

## Environment Architecture

```
Agent (Qwen2.5-7B-Instruct, fine-tuned via GRPO)
        │
        │  6 tool calls + 1 terminal action
        ▼
UndertriAIEnvironment (OpenEnv-compliant FastAPI server)
        │
        ├── request_document
        ├── flag_inconsistency
        ├── cross_reference_precedent
        ├── compute_statutory_eligibility
        ├── assess_surety
        ├── classify_bail_type
        └── submit_memo  ← triggers reward computation
                │
                ▼
        Reward Engine
        R = 0.4×outcome_match + 0.2×flight_risk_acc
          + 0.2×statutory_acc + 0.2×condition_acc
          − 0.3×bias_score
```

---

## Quick Start

### 1. Install

```bash
pip install openenv-core
pip install git+https://huggingface.co/spaces/Draken1606/undertrial-ai
```

### 2. Use the environment

```python
from undertrial_ai import UndertriAIEnv, SubmitMemoAction

async with UndertriAIEnv(base_url="https://draken1606-undertrial-ai.hf.space") as env:
    obs = await env.reset(stage=1)
    print(obs.charge_sheet)       # The case facts
    print(obs.ipc_sections)       # Sections invoked
    print(obs.prosecution_arguments)

    result = await env.step(SubmitMemoAction(
        flight_risk="Low",
        flight_risk_justification="Accused has permanent residence and no prior record.",
        statutory_eligible=True,
        statutory_computation="IPC 420 → max 7 years → threshold 42 months → served 8 months",
        grounds_for_bail=["No flight risk", "Family ties", "Custody approaching threshold"],
        grounds_against_bail=["Investigation pending"],
        recommended_outcome="Bail Granted",
        recommended_conditions=["Surety of ₹25,000", "Weekly court reporting", "Surrender passport"],
    ))
    print(result.reward)    # e.g. 0.78
    print(result.info)      # Full reward breakdown
```

### 3. Prepare your dataset

```bash
python data/prepare_dataset.py \
    --csv /path/to/indian_bail_judgments.csv \
    --output ./data/episodes
```

### 4. Train with GRPO

```bash
python training/train_grpo.py \
    --episodes_dir ./data/episodes \
    --stage 1 \
    --steps 200
```

---

## Reward Formula

| Component | Weight | Description |
|-----------|--------|-------------|
| Outcome Match | 40% | Agent recommendation vs. High Court decision |
| Flight Risk Accuracy | 20% | Flight risk vs. implicit appellate reasoning |
| Statutory Accuracy | 20% | Correct eligibility computation and section citation |
| Condition Appropriateness | 20% | Conditions consistent with appellate order |
| **Bias Penalty** | −30% | Demographic variance penalty (λ = 0.3) |

---

## Curriculum Stages

| Stage | Cases | Description |
|-------|-------|-------------|
| 1 | Landmark cases | Legally clear-cut, near-automatic outcomes |
| 2 | Standard contested | Sessions and HC agree |
| 3 | Reversal cases | `bail_cancellation_case=True` — richest signal |
| 4 | Schema drift | IPC→BNSS remapping, regional FIR formats (Patronus AI bonus) |

---

## Bonus Tracks

- **Patronus AI** — Schema Drift: Stage 4 applies IPC→BNSS section remapping and injects regional FIR format headers (Tamil Nadu, Kerala, Punjab, Maharashtra, Assam)
- **Theme #4** — Self-Improvement: Counterfactual case generator creates synthetic variants for expanded training signal

---

## Dataset

1,200 real Indian High Court bail judgments across 10+ states.  
736 Granted / 464 Rejected. Pre-labeled with `bias_flag` and `parity_argument_used`.

---

## Citation

```
@misc{undertrial-ai-2025,
  title  = {UndertriAI: Bail Decision Support Environment for LLM Training},
  year   = {2025},
  note   = {OpenEnv-compatible environment, Meta PyTorch Hackathon}
}
```
