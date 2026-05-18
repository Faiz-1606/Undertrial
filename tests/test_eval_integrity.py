"""
Evaluation integrity tests for difficulty-mode curriculum reporting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_grpo import load_episodes, select_evaluation_episodes


EPISODES_DIR = str(Path(__file__).parent.parent / "data" / "episodes")


def _ids(episodes):
    return {ep.get("case_id") for ep in episodes}


def test_difficulty_splits_are_disjoint():
    for difficulty in ("easy", "medium", "hard"):
        train = load_episodes(EPISODES_DIR, difficulty=difficulty, split="train")
        val = load_episodes(EPISODES_DIR, difficulty=difficulty, split="val")
        test = load_episodes(EPISODES_DIR, difficulty=difficulty, split="test")

        train_ids = _ids(train)
        val_ids = _ids(val)
        test_ids = _ids(test)

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)


def test_eval_split_order_is_deterministic():
    first = load_episodes(EPISODES_DIR, difficulty="hard", split="val")
    second = load_episodes(EPISODES_DIR, difficulty="hard", split="val")

    assert [ep.get("case_id") for ep in first] == [ep.get("case_id") for ep in second]


def test_hard_eval_split_covers_stage_3_and_4():
    hard_val = load_episodes(EPISODES_DIR, difficulty="hard", split="val")
    stages = {ep.get("curriculum_stage") for ep in hard_val}

    assert 3 in stages
    assert 4 in stages


def test_hard_eval_sample_covers_stage_3_and_4():
    hard_val = load_episodes(EPISODES_DIR, difficulty="hard", split="val")
    selected = select_evaluation_episodes(
        hard_val,
        n_samples=24,
        stratify_by_stage=True,
    )
    stages = {ep.get("curriculum_stage") for ep in selected}

    assert 3 in stages
    assert 4 in stages
