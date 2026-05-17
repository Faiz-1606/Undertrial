# /// script
# dependencies = [
#     "unsloth",
#     "trl",
#     "datasets",
#     "matplotlib",
#     "huggingface_hub",
#     "numpy",
# ]
# ///
"""
UndertriAI — HF Jobs training launcher.

Usage:
  hf jobs uv run --flavor a10g-small --timeout 20h --secrets HF_TOKEN run_hf_job.py
"""
import os
import subprocess
import sys

# Clone the repo
REPO = "https://huggingface.co/spaces/Draken1606/undertrial-ai"
if not os.path.exists("undertrial-ai"):
    subprocess.run(["git", "clone", REPO, "undertrial-ai"], check=True)
os.chdir("undertrial-ai")
sys.path.insert(0, ".")

# Run training
from training.train_grpo import train_curriculum

results = train_curriculum(
    episodes_dir="./data/episodes",
    output_dir="./output/undertrial_grpo",
    difficulties=["easy", "medium", "hard"],
    model_name="unsloth/Qwen2.5-7B-Instruct",
    wandb_disabled=True,
    hf_save_repo="Draken1606/undertrialg",
)

print("\n✅ Training complete! Results uploaded to Draken1606/undertrial-grpo-v2")
