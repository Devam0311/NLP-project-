"""
train.py — Full custom training loop for the 150M CodeGen model.

"""

import os
import sys
import glob
import math
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
import wandb

import config
from model import build_model
from training_logger import TrainingLogger


# ---------------------------------------------------------------------------
# Mixed-precision helpers
# ---------------------------------------------------------------------------

def get_amp_dtype() -> torch.dtype:
    """Return the best autocast dtype for the current hardware."""
    if config.DTYPE == "bfloat16":
        return torch.bfloat16
    return torch.float16


def make_scaler():
    """Create a GradScaler for FP16. Returns None for BF16 (not needed)."""
    if config.DTYPE == "float16":
        return torch.amp.GradScaler("cuda")
    return None


# ---------------------------------------------------------------------------
# LR scheduler — cosine with linear warmup
# ---------------------------------------------------------------------------

def get_lr(step: int) -> float:
    """
    Linear warmup for WARMUP_STEPS, then cosine decay to LR_MIN.
    Returns a multiplier so actual LR = LR_INIT * get_lr(step).
    """
    if step < config.WARMUP_STEPS:
        return step / max(1, config.WARMUP_STEPS)

    progress  = (step - config.WARMUP_STEPS) / max(1, config.MAX_STEPS - config.WARMUP_STEPS)
    cosine    = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = config.LR_MIN / config.LR_INIT
    return min_ratio + (1.0 - min_ratio) * cosine


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def latest_checkpoint() -> "str | None":
    """Return path to the highest-step checkpoint, or None."""
    pattern = os.path.join(config.CHECKPOINT_DIR, "step_*")
    dirs    = sorted(
        glob.glob(pattern),
        key=lambda d: int(d.split("_")[-1])
    )
    return dirs[-1] if dirs else None


def save_checkpoint(model, optimizer, step: int, tokens_seen: int, scaler=None):
    """Save model + optimizer + scaler state to checkpoints/step_{N}/."""
    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, f"step_{step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(ckpt_dir)

    state = {
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "tokens_seen":     tokens_seen,
    }
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    torch.save(state, os.path.join(ckpt_dir, "train_state.pt"))

    meta = {"step": step, "tokens_seen": tokens_seen, "lr": get_lr(step)}
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [ckpt] Saved: {ckpt_dir}")


    all_ckpts = sorted(
        glob.glob(os.path.join(config.CHECKPOINT_DIR, "step_*")),
        key=lambda d: int(d.split("_")[-1])
    )
    for old in all_ckpts[: -config.KEEP_LAST_N_CHECKPOINTS]:
        old_step = int(Path(old).name.split("_")[-1])
        if old_step not in config.PERMANENT_CHECKPOINTS:
            shutil.rmtree(old)
            print(f"  [ckpt] Removed old checkpoint: {old}")


def load_checkpoint(model, optimizer, ckpt_dir: str, scaler=None):
    """Load weights + optimizer state. Returns (step, tokens_seen)."""
    print(f"Resuming from checkpoint: {ckpt_dir}")
    from model import GPT
    state = GPT.from_pretrained(ckpt_dir).state_dict()
    model.load_state_dict(state)

    train_state = torch.load(
        os.path.join(ckpt_dir, "train_state.pt"), weights_only=True
    )
    optimizer.load_state_dict(train_state["optimizer_state"])
    step        = train_state["step"]
    tokens_seen = train_state["tokens_seen"]

    if scaler is not None and "scaler_state" in train_state:
        scaler.load_state_dict(train_state["scaler_state"])

    print(f"  Resumed at step {step:,}, tokens seen {tokens_seen:,}")
    return step, tokens_seen


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(split: str = "train") -> DataLoader:
    path    = os.path.join(config.DATA_CACHE_DIR, split)
    dataset = load_from_disk(path)
    dataset.set_format(type="torch", columns=["input_ids"])

    return DataLoader(
        dataset,
        batch_size         = config.PER_DEVICE_BATCH,
        shuffle            = (split == "train"),
        num_workers        = 2,
        persistent_workers = True,
        prefetch_factor    = 2,
        pin_memory         = True,
        drop_last          = True,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"DTYPE   : {config.DTYPE}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    print("\nBuilding model ...")
    model = build_model(device)

    torch.set_float32_matmul_precision("high")

    # -----------------------------------------------------------------------
    # Mixed-precision setup
    # -----------------------------------------------------------------------
    amp_dtype = get_amp_dtype()
    scaler    = make_scaler()
    print(f"  AMP dtype  : {amp_dtype}")
    print(f"  GradScaler : {'enabled' if scaler else 'disabled (BF16 mode)'}")

    # -----------------------------------------------------------------------
    # Optimizer — AdamW with selective weight decay
    # Decay only 2D+ params, exclude LayerNorm weights and embeddings
    # -----------------------------------------------------------------------
    decay_params   = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.dim() >= 2
        and "ln" not in n and "wte" not in n and "wpe" not in n
    ]
    nodecay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and (
            p.dim() < 2 or "ln" in n or "wte" in n or "wpe" in n
        )
    ]

    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": config.WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr    = config.LR_INIT,
        betas = config.BETAS,
        eps   = config.EPS,
        fused = use_fused,
    )

    # -----------------------------------------------------------------------
    # LR scheduler
    # -----------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # -----------------------------------------------------------------------
    # Checkpoint resumption
    # -----------------------------------------------------------------------
    start_step  = 0
    tokens_seen = 0

    ckpt = latest_checkpoint()
    if ckpt:
        start_step, tokens_seen = load_checkpoint(model, optimizer, ckpt, scaler)
        # Fast-forward the LR scheduler to the correct step
        for _ in range(start_step):
            scheduler.step()
    else:
        print("No checkpoint found — starting from scratch.")

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------
    print("\nLoading dataset ...")
    train_loader = make_dataloader("train")

    # -----------------------------------------------------------------------
    # WandB login + init
    # -----------------------------------------------------------------------
    if config.WANDB_API_KEY:
        wandb.login(key=config.WANDB_API_KEY, relogin=True)
        print(f"  [wandb] Logged in with API key from environment.")
    else:
        # Fall back to interactive login or WANDB_API_KEY env var already set
        print("  [wandb] No WANDB_API_KEY in config — using existing login or anonymous mode.")
        print("  [wandb] To set key: export WANDB_API_KEY=your_key_here")

    wandb.init(
        project = config.WANDB_PROJECT,
        name    = config.WANDB_RUN_NAME,
        config  = {
            "n_layer":          config.N_LAYER,
            "n_embd":           config.N_EMBD,
            "n_head":           config.N_HEAD,
            "n_inner":          config.N_INNER,
            "vocab_size":       config.VOCAB_SIZE,
            "context_length":   config.CONTEXT_LENGTH,
            "max_steps":        config.MAX_STEPS,
            "effective_batch":  config.EFFECTIVE_BATCH,
            "per_device_batch": config.PER_DEVICE_BATCH,
            "grad_accum_steps": config.GRAD_ACCUM_STEPS,
            "lr_init":          config.LR_INIT,
            "lr_min":           config.LR_MIN,
            "warmup_steps":     config.WARMUP_STEPS,
            "weight_decay":     config.WEIGHT_DECAY,
            "grad_clip":        config.GRAD_CLIP,
            "dtype":            config.DTYPE,
            "target_tokens":    config.TARGET_TOKENS,
            "save_every_steps": config.SAVE_EVERY_STEPS,
        },
        resume  = "allow",
    )
    print(f"  [wandb] Run URL: {wandb.run.get_url()}")

    # -----------------------------------------------------------------------
    # Lazy import of evaluate to avoid circular imports
    # -----------------------------------------------------------------------
    from evaluate import run_eval

    # -----------------------------------------------------------------------
    # CSV logger
    # -----------------------------------------------------------------------
    csv_logger = TrainingLogger(
        log_dir=config.CHECKPOINT_DIR, filename="training_metrics.csv"
    )
    print(f"  CSV log: {csv_logger.log_path}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    step         = start_step
    running_loss = 0.0

    def cycle(loader, initial_epoch: int = 0):
        """Infinite generator that cycles over the dataloader."""
        epoch = initial_epoch
        while True:
            for batch in loader:
                yield batch
            epoch += 1

    initial_epoch = 0
    if len(train_loader) > 0:
        initial_epoch = (start_step * config.GRAD_ACCUM_STEPS) // len(train_loader)
    data_iter = cycle(train_loader, initial_epoch=initial_epoch)

    # Skip micro-batches already consumed in the previous session
    batches_to_skip = (start_step * config.GRAD_ACCUM_STEPS) % max(len(train_loader), 1)
    if batches_to_skip:
        print(f"Skipping {batches_to_skip:,} micro-batches to resume ...")
        for _ in range(batches_to_skip):
            next(data_iter)

    # Training schedule banner
    print("\n" + "=" * 65)
    print(" TRAINING SCHEDULE")
    print("=" * 65)
    print(f"  Steps        : {start_step:,} -> {config.MAX_STEPS:,}")
    print(f"  Warmup       : {config.WARMUP_STEPS:,} steps (linear)")
    print(f"  Log every    : {config.LOG_EVERY_STEPS} steps  -> WandB + CSV")
    print(f"  Checkpoint   : every {config.SAVE_EVERY_STEPS} steps -> {config.CHECKPOINT_DIR}/step_NNNNNNN/")
    print(f"  Evaluation   : every {config.SAVE_EVERY_STEPS} steps (val loss + generation + syntax rate)")
    print(f"  WandB project: {config.WANDB_PROJECT}  run: {config.WANDB_RUN_NAME}")
    print("=" * 65 + "\n")
    optimizer.zero_grad()

    for micro_step, batch in enumerate(data_iter):
        if step >= config.MAX_STEPS:
            break

        input_ids = batch["input_ids"].to(device).long()
        labels    = input_ids.clone()

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out  = model(input_ids=input_ids, labels=labels)
            loss = out.loss / config.GRAD_ACCUM_STEPS

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss.item()

        # Optimizer step every GRAD_ACCUM_STEPS micro-batches
        if (micro_step + 1) % config.GRAD_ACCUM_STEPS == 0:
            step        += 1
            tokens_seen += config.EFFECTIVE_BATCH * config.CONTEXT_LENGTH

            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            # --- Gradient skip safety net ---
            
            GRAD_SKIP_THRESHOLD = 100.0
            _raw_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            if _raw_norm > GRAD_SKIP_THRESHOLD:
                print(f"  ⚠ SKIPPING step {step} — grad_norm {_raw_norm:.1f} > {GRAD_SKIP_THRESHOLD}")
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.update()
                scheduler.step()
                running_loss = 0.0
                continue

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            current_lr   = scheduler.get_last_lr()[0]
            avg_loss     = running_loss
            running_loss = 0.0

            # -----------------------------------------------------------
            # Logging
            # -----------------------------------------------------------
            if step % config.LOG_EVERY_STEPS == 0:
                print(
                    f"step {step:6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"grad_norm {grad_norm:.3f} | "
                    f"tokens {tokens_seen:,}"
                )

            _grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            wandb.log({
                "train_loss":    avg_loss,
                "learning_rate": current_lr,
                "grad_norm":     _grad_norm,
                "tokens_seen":   tokens_seen,
                "step":          step,
            }, step=step)
            csv_logger.log({
                "step":          step,
                "train_loss":    avg_loss,
                "learning_rate": current_lr,
                "grad_norm":     _grad_norm,
                "tokens_seen":   tokens_seen,
            })

            # -----------------------------------------------------------
            # Checkpoint + evaluation  (every SAVE_EVERY_STEPS = 5000)
            # -----------------------------------------------------------
            if step % config.SAVE_EVERY_STEPS == 0:
                ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"step_{step:07d}")
                print(f"\n{'='*65}")
                print(f"  CHECKPOINT @ step {step:,}  ({tokens_seen:,} tokens seen)")
                print(f"  Saving to : {ckpt_path}")
                print(f"{'='*65}")
                save_checkpoint(model, optimizer, step, tokens_seen, scaler)

                print(f"\n  EVALUATION @ step {step:,}")
                run_eval(model, device, step)

                # Log checkpoint event to WandB
                wandb.log({"checkpoint_step": step}, step=step)

    # -----------------------------------------------------------------------
    # Final checkpoint
    # -----------------------------------------------------------------------
    save_checkpoint(model, optimizer, step, tokens_seen, scaler)
    print(f"\nTraining complete — {step:,} steps, {tokens_seen:,} tokens seen.")
    wandb.finish()


if __name__ == "__main__":
    train()
