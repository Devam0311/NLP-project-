"""
main.py — Orchestrator for the full 150M CodeGen training pipeline.

Stages:
  1. data  — Stream, filter, tokenise, and cache 3B tokens to disk.
  2. train — Train the 150M model to 45,776 steps (~3B tokens).

Each stage is idempotent:
  - Data stage is skipped if DATA_CACHE_DIR/train already exists.
  - Training resumes automatically from the latest checkpoint.

Usage:
  python main.py                  # run all stages (data -> train)
  python main.py --stage data     # only build dataset
  python main.py --stage train    # only train (dataset must exist first)
  python main.py --stage sft      # SFT fine-tuning after pre-training
  python main.py --stage sft --base-checkpoint PATH
"""

import os
import sys
import argparse

import config


# ---------------------------------------------------------------------------
# Stage: data
# ---------------------------------------------------------------------------

def stage_data():
    """Build and cache the tokenised dataset to disk."""
    train_done = os.path.join(config.DATA_CACHE_DIR, "train", "dataset_info.json")
    if os.path.exists(train_done):
        print(f"[data] Dataset already exists at {config.DATA_CACHE_DIR} — skipping.")
        print("[data] Delete the directory to re-build from scratch.")
        return

    print("\n" + "=" * 60)
    print(" STAGE 1: DATA PIPELINE")
    print("=" * 60 + "\n")
    from data_pipeline import build_dataset
    build_dataset()


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------

def stage_train():
    """Run the main pre-training loop."""
    cache_ok = os.path.exists(
        os.path.join(config.DATA_CACHE_DIR, "train", "dataset_info.json")
    )
    if not cache_ok:
        print("[train] ERROR: No dataset found. Run 'python main.py --stage data' first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" STAGE 2: PRE-TRAINING")
    print("=" * 60 + "\n")
    from train import train
    train()


# ---------------------------------------------------------------------------
# Stage: sft
# ---------------------------------------------------------------------------

def stage_sft(base_checkpoint: "str | None"):
    """Run supervised fine-tuning after pre-training."""
    import glob
    if base_checkpoint is None:
        ckpts = sorted(
            glob.glob(os.path.join(config.CHECKPOINT_DIR, "step_*")),
            key=lambda d: int(d.split("_")[-1])
        )
        if not ckpts:
            print("[sft] ERROR: No pre-training checkpoints found.")
            print("       Run 'python main.py --stage train' first.")
            sys.exit(1)
        base_checkpoint = ckpts[-1]
        print(f"[sft] Using latest checkpoint: {base_checkpoint}")

    print("\n" + "=" * 60)
    print(" STAGE 3: SFT FINE-TUNING")
    print("=" * 60 + "\n")
    from sft_train import train_sft
    train_sft(base_checkpoint)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CodeGen-150M: full pipeline (data -> pre-train -> SFT)"
    )
    parser.add_argument(
        "--stage",
        choices=["data", "train", "sft", "all"],
        default="all",
        help="Which stage to run (default: all = data + train)",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="(SFT only) Path to pre-trained checkpoint. Defaults to latest.",
    )
    args = parser.parse_args()

    if args.stage in ("data", "all"):
        stage_data()

    if args.stage in ("train", "all"):
        stage_train()

    if args.stage == "sft":
        stage_sft(args.base_checkpoint)

    print("\nPipeline complete.")
