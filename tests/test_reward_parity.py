"""
UndertriAI reward source parity tests.

These tests verify that the trainer reward path and the server reward contract
score the same parsed completion without silent defaults.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.reward import compute_reward
from training.train_grpo import combined_reward, parse_model_output

from tests.test_anti_hack import GOLDEN_COMPLETION, GOLDEN_EPISODE


EPSILON = 1e-4


def _process_bonus_proxy(parsed: dict, episode: dict) -> bool:
    custody_mo = episode.get("custody_months") or 0.0
    max_sent = episode.get("max_sentence_years", 5.0)
    comp_text = (parsed.get("statutory_computation") or "").lower()
    if custody_mo <= 0:
        return False
    threshold_mo = (max_sent * 12) / 2.0
    return str(int(custody_mo)) in comp_text and str(int(threshold_mo)) in comp_text


def _server_total(completion: str, episode: dict) -> float:
    parsed = parse_model_output(completion)
    result = compute_reward(
        agent_outcome=parsed["recommended_outcome"],
        agent_flight_risk=parsed["flight_risk"],
        agent_eligible=parsed["statutory_eligible"],
        agent_computation=parsed["statutory_computation"] or "",
        agent_conditions=parsed["conditions"] or [],
        episode=episode,
        step_count=0,
        max_steps=10,
        statutory_tool_used=_process_bonus_proxy(parsed, episode),
        agent_flight_risk_justification=parsed["flight_risk_just"] or "",
        agent_grounds_for=parsed["grounds_for"] or [],
        agent_grounds_against=parsed["grounds_against"] or [],
        completion_text=completion,
        current_stage=episode.get("curriculum_stage", 1),
    )
    return result["total_reward"]


def _assert_parity(name: str, completion: str, episode: dict) -> None:
    server_total = _server_total(completion, episode)
    trainer_total = combined_reward(
        [completion],
        [episode],
        current_stage=episode.get("curriculum_stage", 1),
    )[0]
    delta = abs(server_total - trainer_total)
    print(
        f"{name}: server={server_total:.4f}, "
        f"trainer={trainer_total:.4f}, delta={delta:.6f}"
    )
    assert delta <= EPSILON, (
        f"{name} reward parity failed: delta={delta:.6f} exceeds {EPSILON}"
    )


def test_reward_parity():
    _assert_parity("golden", GOLDEN_COMPLETION, GOLDEN_EPISODE)


def test_parity_on_adversarial():
    from tests.test_anti_hack import (
        LEGALESE_COMPLETION,
        NUMBER_COPY_COMPLETION,
        OUTCOME_ONLY_COMPLETION,
        TEMPLATE_COMPLETION,
    )

    _assert_parity("template", TEMPLATE_COMPLETION, GOLDEN_EPISODE)
    _assert_parity("legalese", LEGALESE_COMPLETION, GOLDEN_EPISODE)
    _assert_parity("number_copy", NUMBER_COPY_COMPLETION, GOLDEN_EPISODE)
    _assert_parity("outcome_only", OUTCOME_ONLY_COMPLETION, GOLDEN_EPISODE)
