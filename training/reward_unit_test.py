from training.train_grpo import combined_reward, parse_model_output


def main() -> None:
    perfect_completion = """
<think>
IPC Section 420, max 7 years. BNSS threshold: 28 months. Custody: 8 months.
8 < 28, so not eligible for default bail. However, permanent resident,
no prior record, two children dependent. Flight risk: Low.
Recommend bail granted with conditions.
</think>

<memo>
<flight_risk>Low</flight_risk>
<flight_risk_justification>Permanent Delhi resident, no prior cases, family ties</flight_risk_justification>
<statutory_eligible>false</statutory_eligible>
<statutory_computation>IPC 420 max 7 years → threshold 28 months → custody 8 months < 28 → not eligible</statutory_computation>
<grounds_for_bail>
  <ground>No prior criminal record</ground>
  <ground>Permanent resident with family</ground>
</grounds_for_bail>
<grounds_against_bail>
  <ground>Investigation pending</ground>
</grounds_against_bail>
<recommended_outcome>Bail Granted</recommended_outcome>
<recommended_conditions>
  <condition>Surety Rs 25000</condition>
  <condition>Weekly reporting</condition>
</recommended_conditions>
</memo>
"""

    test_episode = {
        "case_id": "TEST001",
        "ipc_sections": ["420"],
        "max_sentence_years": 7.0,
        "custody_months": 8.0,
        "crime_type": "Fraud",
        "ground_truth": {
            "outcome": "Bail Granted",
            "implicit_flight_risk": "Low",
            "bias_flag": False,
        },
    }

    reward = combined_reward([perfect_completion], [test_episode], current_stage=1)[0]
    print(f"Perfect completion reward: {reward:.4f}")

    parsed = parse_model_output(perfect_completion)
    print("\nParsed fields:")
    for k, v in parsed.items():
        print(f"  {k}: {v}")

    minimal_completion = "Bail granted."
    reward_min = combined_reward([minimal_completion], [test_episode], current_stage=1)[0]
    print(f"\nMinimal completion reward: {reward_min:.4f}")

    wrong_completion = perfect_completion.replace("Bail Granted", "Bail Denied")
    reward_wrong = combined_reward([wrong_completion], [test_episode], current_stage=1)[0]
    print(f"\nWrong outcome reward: {reward_wrong:.4f}")

    assert reward > reward_min, "Perfect should beat minimal!"
    assert reward > reward_wrong, "Correct outcome should beat wrong outcome!"
    print("\n✓ Reward function sanity checks pass")


if __name__ == "__main__":
    main()

