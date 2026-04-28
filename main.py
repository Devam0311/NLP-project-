"""
Runs the main training pipeline.

The pipeline has separate stages for preparing data, pre-training,
and supervised fine-tuning. Each stage can be run independently
from the command line.
"""

import os
import sys
import argparse

import config


# Data preparation stage

def stage_data():
    """Prepare and save the tokenized dataset."""

    train_done = os.path.join(
        config.DATA_CACHE_DIR,
        "train",
        "dataset_info.json"
    )

    if os.path.exists(train_done):
        print(f"[data] Dataset already exists at {config.DATA_CACHE_DIR} — skipping.")
        print("[data] Delete the directory to build it again.")
        return

    print("\n" + "=" * 60)
    print(" STAGE 1: DATA PIPELINE")
    print("=" * 60 + "\n")

    from data_pipeline import build_dataset

    build_dataset()


# Main training stage

def stage_train():
    """Run pre-training."""

    cache_ok = os.path.exists(
        os.path.join(
            config.DATA_CACHE_DIR,
            "train",
            "dataset_info.json"
        )
    )

    if not cache_ok:
        print("[train] ERROR: Dataset not found. Run data stage first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" STAGE 2: PRE-TRAINING")
    print("=" * 60 + "\n")

    from train import train

    train()


# Supervised fine-tuning stage

def stage_sft(base_checkpoint: "str | None"):
    """Run SFT using a pre-trained checkpoint."""

    import glob

    if base_checkpoint is None:
        ckpts = sorted(
            glob.glob(
                os.path.join(
                    config.CHECKPOINT_DIR,
                    "step_*"
                )
            ),
            key=lambda d: int(d.split("_")[-1])
        )

        if not ckpts:
            print("[sft] ERROR: No pre-training checkpoints found.")
            print("      Run training before starting SFT.")
            sys.exit(1)

        base_checkpoint = ckpts[-1]

        print(f"[sft] Using latest checkpoint: {base_checkpoint}")

    print("\n" + "=" * 60)
    print(" STAGE 3: SFT FINE-TUNING")
    print("=" * 60 + "\n")

    from sft_train import train_sft

    train_sft(base_checkpoint)


# Command line entry point

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CodeGen-150M training pipeline"
    )

    parser.add_argument(
        "--stage",
        choices=["data", "train", "sft", "all"],
        default="all",
        help="Stage to run"
    )

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for SFT"
    )

    args = parser.parse_args()

    if args.stage in ("data", "all"):
        stage_data()

    if args.stage in ("train", "all"):
        stage_train()

    if args.stage == "sft":
        stage_sft(args.base_checkpoint)

    print("\nPipeline complete.")
