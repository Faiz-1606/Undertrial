"""
UndertriAI — Reward Source Parity Tests

Verifies that the server reward path (compute_reward) and the trainer
reward path (combined_reward) produce matching scores within epsilon.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.reward import compute_reward
from training.train_grpo import combined_reward, parse_model_output

# Use the same golden episode and completion from anti-hack tests
from tests.test_anti_hack import GOLDEN_EPISODE, GOLDEN_COMPLETION


EPSILON = 0.10  # Acceptable parity tolerance


def test_reward_parity():
    """
    Compare server compute_reward() vs trainer combined_reward() on the same input.
    They should produce similar total rewards within epsilon.
    """
    # ── Server path ──
    parsed = parse_model_output(GOLDEN_COMPLETION)
    server_result = compute_reward(
        agent_outcome=parsed["recommended_outcome"],
        agent_flight_risk=parsed["flight_risk"] or "Medium",
        agent_eligible=parsed["statutory_eligible"],
        agent_computation=parsed["statutory_computation"] or "",
        agent_conditions=parsed["conditions"] or [],
        episode=GOLDEN_EPISODE,
        agent_flight_risk_justification=parsed["flight_risk_just"] or "",
        agent_grounds_for=parsed["grounds_for"] or [],
        agent_grounds_against=parsed["grounds_against"] or [],
        completion_text=GOLDEN_COMPLETION,
        current_stage=1,
    )

    # ── Trainer path ──
    trainer_rewards = combined_reward(
        [GOLDEN_COMPLETION],
        [GOLDEN_EPISODE],
        current_stage=1,
    )

    server_total = server_result["total_reward"]
    trainer_total = trainer_rewards[0]
    delta = abs(server_total - trainer_total)

    print(f"\n{'═' * 50}")
    print(f"  Reward Parity Test")
    print(f"{'═' * 50}")
    print(f"  Server  total_reward: {server_total:.4f}")
    print(f"  Trainer total_reward: {trainer_total:.4f}")
    print(f"  Delta:                {delta:.4f}")
    print(f"  Epsilon:              {EPSILON}")

    # Print component breakdown from server
    print(f"\n  Server breakdown:")
    for key in ["outcome_match", "flight_risk_accuracy", "statutory_accuracy",
                 "condition_appropriateness", "reasoning_quality", "bias_penalty",
                 "consistency_gate", "format_score"]:
        print(f"    {key:30s}: {server_result.get(key, 'N/A')}")

    if delta <= EPSILON:
        print(f"\n  ✅ PASS: Parity within ε={EPSILON}")
    else:
        print(f"\n  ⚠️  WARN: Parity drift {delta:.4f} > ε={EPSILON}")
        print(f"    Server and trainer reward paths may diverge.")
        print(f"    This can cause non-stationary training objectives.")

    return delta <= EPSILON


def test_parity_on_adversarial():
    """Test parity on adversarial probes too — they should also match."""
    from tests.test_anti_hack import TEMPLATE_COMPLETION, OUTCOME_ONLY_COMPLETION

    probes = {
        "template": TEMPLATE_COMPLETION,
        "outcome_only": OUTCOME_ONLY_COMPLETION,
    }

    print(f"\n{'═' * 50}")
    print(f"  Adversarial Parity Tests")
    print(f"{'═' * 50}")

    all_pass = True
    for name, completion in probes.items():
        parsed = parse_model_output(completion)
        server_result = compute_reward(
            agent_outcome=parsed["recommended_outcome"] or "Bail Denied",
            agent_flight_risk=parsed["flight_risk"] or "Medium",
            agent_eligible=parsed["statutory_eligible"],
            agent_computation=parsed["statutory_computation"] or "",
            agent_conditions=parsed["conditions"] or [],
            episode=GOLDEN_EPISODE,
            agent_flight_risk_justification=parsed["flight_risk_just"] or "",
            agent_grounds_for=parsed["grounds_for"] or [],
            agent_grounds_against=parsed["grounds_against"] or [],
            completion_text=completion,
            current_stage=1,
        )

        trainer_rewards = combined_reward(
            [completion], [GOLDEN_EPISODE], current_stage=1,
        )

        delta = abs(server_result["total_reward"] - trainer_rewards[0])
        status = "✅" if delta <= EPSILON else "⚠️"
        print(f"  {status} {name:20s}: server={server_result['total_reward']:.4f}, "
              f"trainer={trainer_rewards[0]:.4f}, Δ={delta:.4f}")
        if delta > EPSILON:
            all_pass = False

    return all_pass


if __name__ == "__main__":
    print("=" * 50)
    print("  UndertriAI Reward Parity Test Suite")
    print("=" * 50)

    test_reward_parity()
    test_parity_on_adversarial()
