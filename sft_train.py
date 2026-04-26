"""
sft_train.py — Supervised Fine-Tuning for the 150M CodeGen model.

Takes the best pre-trained checkpoint and fine-tunes it on
18,612 Python instruction->code pairs from:
  iamtarun/python_code_instructions_18k_alpaca

Key design decisions:
  - Response-only loss masking: instruction/input tokens are masked with -100.
    Only the code response tokens contribute to the loss, so the model learns
    to *generate* code, not repeat instructions.
  - Low LR (1e-5): 10x lower than pre-training to preserve pre-trained
    language modelling capabilities.
  - FP16/BF16: consistent with pre-training (auto-detected from config).
  - 3 epochs: standard for small instruction datasets.
  - Effective batch = 2 x 16 = 32 sequences (150M needs smaller micro-batch).

Format:
  ### Instruction:
  {instruction}

  ### Input:          <- only if input field is non-empty
  {input}

  ### Response:
  {output}

Usage:
  python sft_train.py                          # uses latest pre-training ckpt
  python sft_train.py --base-checkpoint PATH  # use a specific checkpoint
"""

import os
import sys
import math
import json
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb

from model import GPT
import config
from training_logger import TrainingLogger

amp_dtype = torch.bfloat16 if config.DTYPE == "bfloat16" else torch.float16

# ---------------------------------------------------------------------------
# SFT paths (self-contained inside the project directory)
# ---------------------------------------------------------------------------
PROJECT_DIR        = os.path.dirname(os.path.abspath(__file__))
SFT_CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "sft_checkpoints")
SFT_EVAL_DIR       = os.path.join(PROJECT_DIR, "sft_eval_outputs")
SFT_FINAL_DIR      = os.path.join(PROJECT_DIR, "sft_final_model")


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def format_sample(instruction: str, inp: str, output: str) -> str:
    """Format one sample into the instruction-tuning text format."""
    if inp and inp.strip():
        return (
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{inp.strip()}\n\n"
            f"### Response:\n{output.strip()}"
        )
    return (
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response:\n{output.strip()}"
    )


# ---------------------------------------------------------------------------
# SFT Dataset with response-only loss masking
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Tokenised instruction-following dataset.

    For each sample:
      1. Format full text (instruction + optional input + response).
      2. Tokenise and truncate to SFT_MAX_SEQ_LEN.
      3. Build labels = [-100, ..., -100, <response tokens>]
         so only the response contributes to the loss.
    """

    def __init__(self, samples: list, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.data      = []

        response_marker = "### Response:\n"
        skipped = 0

        for s in samples:
            text = format_sample(
                s["instruction"], s.get("input", ""), s["output"]
            )

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids.append(tokenizer.eos_token_id)

            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]

            # Find prefix length (everything before and including "### Response:\n")
            marker_pos = text.find(response_marker)
            if marker_pos == -1:
                skipped += 1
                continue
            prefix_text = text[: marker_pos + len(response_marker)]
            prefix_ids  = tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len  = min(len(prefix_ids), len(token_ids))

            # Labels: -100 for instruction, real token ids for response
            labels = [-100] * prefix_len + token_ids[prefix_len:]
            labels = labels[:len(token_ids)]   # align length

            if len(token_ids) < 10:            # skip degenerate samples
                skipped += 1
                continue

            self.data.append({"input_ids": token_ids, "labels": labels})

        print(f"  SFTDataset: {len(self.data):,} samples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Pad a batch to the length of the longest sequence."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_padded  = []
    labels_padded     = []
    attention_mask    = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids_padded.append(item["input_ids"]  + [0]    * pad_len)
        labels_padded.append(   item["labels"]     + [-100] * pad_len)
        attention_mask.append(  [1] * len(item["input_ids"]) + [0] * pad_len)

    return {
        "input_ids":      torch.tensor(input_ids_padded, dtype=torch.long),
        "labels":         torch.tensor(labels_padded,    dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask,   dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def get_sft_lr(step: int, total_steps: int) -> float:
    """Cosine decay with linear warmup for SFT."""
    if step < config.SFT_WARMUP_STEPS:
        return step / max(1, config.SFT_WARMUP_STEPS)
    progress  = (step - config.SFT_WARMUP_STEPS) / max(1, total_steps - config.SFT_WARMUP_STEPS)
    cosine    = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = config.SFT_LR_MIN / config.SFT_LR
    return min_ratio + (1.0 - min_ratio) * cosine


# ---------------------------------------------------------------------------
# SFT Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sft_eval(model, tokenizer, device, step: int, val_loader=None):
    """Generate from code + instruction prompts; compute syntax pass rate."""
    model.eval()
    results = []

    all_prompts    = config.EVAL_PROMPTS + config.SFT_INSTRUCTION_PROMPTS
    prompt_labels  = (
        [f"Code Prompt {i+1}"        for i in range(len(config.EVAL_PROMPTS))] +
        [f"Instruction Prompt {i+1}" for i in range(len(config.SFT_INSTRUCTION_PROMPTS))]
    )

    for label, prompt in zip(prompt_labels, all_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            gen_ids = model.generate(
                input_ids,
                max_new_tokens     = 300,
                temperature        = 0.2,
                top_p              = 0.95,
                top_k              = 50,
                do_sample          = True,
                repetition_penalty = 1.2,
                pad_token_id       = tokenizer.eos_token_id,
                eos_token_id       = tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # Fix drift: cut off if the model starts hallucinating a new instruction
        if "### Instruction:" in text[len(prompt):]:
            text = text[:len(prompt)] + text[len(prompt):].split("### Instruction:")[0].strip()
        results.append((label, prompt, text))

    # Syntax pass rate on code prompts only
    code_outputs = [r[2] for r in results[:len(config.EVAL_PROMPTS)]]
    syntax_rate  = sum(1 for t in code_outputs if _is_valid_python(t)) / max(len(code_outputs), 1)

    # Validation loss
    val_loss = 0.0
    if val_loader:
        total_loss, n_batches = 0.0, 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(input_ids=input_ids, labels=labels)
            total_loss += out.loss.item()
            n_batches  += 1
        val_loss = total_loss / max(n_batches, 1)

    print(f"\n{'='*60}")
    print(f"  SFT Evaluation @ step {step:,}")
    print(f"{'='*60}")
    if val_loader:
        print(f"  sft_val_loss     : {val_loss:.4f}")
        print(f"  sft_val_ppl      : {math.exp(val_loss):.2f}")
    print(f"  syntax_pass_rate : {syntax_rate:.2%}")
    for label, _, text in results:
        print(f"\n--- {label} ---")
        print(text[:600])
    print(f"{'='*60}\n")

    # Save to file
    os.makedirs(SFT_EVAL_DIR, exist_ok=True)
    out_path = os.path.join(SFT_EVAL_DIR, f"sft_step_{step:06d}.txt")
    with open(out_path, "w") as f:
        f.write(f"SFT Step {step}\n{'='*60}\n\n")
        if val_loader:
            f.write(f"sft_val_loss     : {val_loss:.4f}\n")
            f.write(f"sft_val_ppl      : {math.exp(val_loss):.2f}\n")
        f.write(f"syntax_pass_rate : {syntax_rate:.2%}\n\n")
        for label, _, text in results:
            f.write(f"\n--- {label} ---\n{text}\n{'*'*40}\n")

    log_dict = {"sft_syntax_pass_rate": syntax_rate, "sft_step": step}
    if val_loader:
        log_dict["sft_val_loss"] = val_loss
        log_dict["sft_val_ppl"]  = math.exp(val_loss)
    wandb.log(log_dict, step=step)

    model.train()
    return syntax_rate, val_loss


def _is_valid_python(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_sft_checkpoint(model, optimizer, step: int, epoch: int, train_loss: float):
    """Save SFT model + optimizer state."""
    ckpt_dir = os.path.join(SFT_CHECKPOINT_DIR, f"sft_step_{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    torch.save({
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "epoch":           epoch,
        "train_loss":      train_loss,
    }, os.path.join(ckpt_dir, "sft_train_state.pt"))

    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump({"step": step, "epoch": epoch, "train_loss": train_loss}, f, indent=2)

    print(f"  [sft-ckpt] Saved: {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main SFT training
# ---------------------------------------------------------------------------

def train_sft(base_checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"DTYPE  : {config.DTYPE}")
    os.makedirs(SFT_CHECKPOINT_DIR, exist_ok=True)

    # -------------------------------------------------------------------
    # Load pre-trained model
    # -------------------------------------------------------------------
    print(f"\nLoading base model from {base_checkpoint} ...")
    model = GPT(
        vocab_size  = config.VOCAB_SIZE,
        n_positions = config.N_POSITIONS,
        n_embd      = config.N_EMBD,
        n_layer     = config.N_LAYER,
        n_head      = config.N_HEAD,
        n_inner     = config.N_INNER,
    ).to(device)
    state = GPT.from_pretrained(base_checkpoint).state_dict()
    model.load_state_dict(state)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # -------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer: {config.TOKENIZER_ID} (vocab={tokenizer.vocab_size})")

    # -------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------
    print(f"\nLoading SFT dataset: {config.SFT_DATASET_ID} ...")
    raw_ds = load_dataset(config.SFT_DATASET_ID, split="train")
    print(f"  Raw samples: {len(raw_ds):,}")

    split         = raw_ds.train_test_split(test_size=config.SFT_VAL_FRACTION, seed=42)
    train_samples = list(split["train"])
    val_samples   = list(split["test"])
    print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}")

    print("\nTokenising train set ...")
    train_ds = SFTDataset(train_samples, tokenizer, max_len=config.SFT_MAX_SEQ_LEN)
    print("Tokenising val set ...")
    val_ds   = SFTDataset(val_samples,   tokenizer, max_len=config.SFT_MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.SFT_BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.SFT_BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 2,
        pin_memory  = True,
    )

    steps_per_epoch = len(train_loader) // config.SFT_GRAD_ACCUM
    total_steps     = steps_per_epoch * config.SFT_EPOCHS
    print(f"\n  Steps per epoch : {steps_per_epoch}")
    print(f"  Total steps     : {total_steps}")
    print(f"  Effective batch : {config.SFT_BATCH_SIZE * config.SFT_GRAD_ACCUM}")

    # -------------------------------------------------------------------
    # Optimizer + scaler
    # -------------------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda") if config.DTYPE == "float16" else None

    decay_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2
                      and "ln" not in n and "wte" not in n and "wpe" not in n]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and (
                          p.dim() < 2 or "ln" in n or "wte" in n or "wpe" in n
                      )]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": config.SFT_WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr    = config.SFT_LR,
        betas = config.SFT_BETAS,
        eps   = config.SFT_EPS,
        fused = torch.cuda.is_available(),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: get_sft_lr(s, total_steps)
    )

    # -------------------------------------------------------------------
    # WandB
    # -------------------------------------------------------------------
    wandb.init(
        project = config.WANDB_PROJECT,
        name    = f"sft-{config.WANDB_RUN_NAME}",
        config  = {
            "base_checkpoint": base_checkpoint,
            "dataset":         config.SFT_DATASET_ID,
            "lr":              config.SFT_LR,
            "epochs":          config.SFT_EPOCHS,
            "batch_size":      config.SFT_BATCH_SIZE * config.SFT_GRAD_ACCUM,
            "max_seq_len":     config.SFT_MAX_SEQ_LEN,
        },
    )

    # -------------------------------------------------------------------
    # Baseline evaluation (pre-SFT)
    # -------------------------------------------------------------------
    print("\n--- Pre-SFT Baseline Evaluation ---")
    run_sft_eval(model, tokenizer, device, step=0, val_loader=val_loader)

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    csv_logger = TrainingLogger(log_dir=SFT_CHECKPOINT_DIR, filename="sft_metrics.csv")
    print(f"  CSV log: {csv_logger.log_path}")

    global_step  = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    avg_loss     = 0.0

    print(f"\nStarting SFT for {config.SFT_EPOCHS} epochs ({total_steps} steps) ...\n")

    for epoch in range(config.SFT_EPOCHS):
        print(f"--- Epoch {epoch + 1}/{config.SFT_EPOCHS} ---")
        optimizer.zero_grad()
        model.train()

        for micro_step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            labels         = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out  = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = out.loss / config.SFT_GRAD_ACCUM

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item()

            if (micro_step + 1) % config.SFT_GRAD_ACCUM == 0:
                global_step += 1

                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), config.SFT_GRAD_CLIP
                )

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

                if global_step % config.SFT_LOG_EVERY == 0:
                    print(
                        f"  step {global_step:5d}/{total_steps} | "
                        f"epoch {epoch+1} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {current_lr:.2e} | "
                        f"grad_norm {grad_norm:.3f}"
                    )

                _gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                wandb.log({
                    "sft_train_loss": avg_loss,
                    "sft_lr":         current_lr,
                    "sft_grad_norm":  _gn,
                    "sft_epoch":      epoch + 1,
                    "sft_step":       global_step,
                }, step=global_step)
                csv_logger.log({
                    "step":       global_step,
                    "epoch":      epoch + 1,
                    "train_loss": avg_loss,
                    "lr":         current_lr,
                    "grad_norm":  _gn,
                })

                # Checkpoint + eval
                if global_step % config.SFT_SAVE_EVERY == 0:
                    save_sft_checkpoint(model, optimizer, global_step, epoch + 1, avg_loss)
                    syntax_rate, v_loss = run_sft_eval(
                        model, tokenizer, device, global_step, val_loader
                    )
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_dir = os.path.join(SFT_CHECKPOINT_DIR, "best")
                        os.makedirs(best_dir, exist_ok=True)
                        model.save_pretrained(best_dir)
                        print(f"  ★ New best model saved (val_loss={v_loss:.4f})")

        # --- Flush leftover micro-batches at end of epoch ---
        # If the number of batches doesn't divide evenly by SFT_GRAD_ACCUM,
        # the last few micro-batches have accumulated gradients that haven't
        # been stepped yet. Flush them so no training signal is lost.
        leftover = (micro_step + 1) % config.SFT_GRAD_ACCUM
        if leftover > 0:
            global_step += 1

            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.SFT_GRAD_CLIP
            )

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

            print(
                f"  step {global_step:5d}/{total_steps} | "
                f"epoch {epoch+1} [flush] | "
                f"loss {avg_loss:.4f} | "
                f"lr {current_lr:.2e} | "
                f"grad_norm {grad_norm:.3f}"
            )

    # -------------------------------------------------------------------
    # Final checkpoint + evaluation
    # -------------------------------------------------------------------
    save_sft_checkpoint(model, optimizer, global_step, config.SFT_EPOCHS, avg_loss)
    run_sft_eval(model, tokenizer, device, global_step, val_loader)

    # -------------------------------------------------------------------
    # Export final model
    # -------------------------------------------------------------------
    os.makedirs(SFT_FINAL_DIR, exist_ok=True)
    model.save_pretrained(SFT_FINAL_DIR)
    tokenizer.save_pretrained(SFT_FINAL_DIR)
    print(f"\n  Final SFT model saved: {SFT_FINAL_DIR}")

    print(f"\n{'='*60}")
    print(f"  SFT Training Complete")
    print(f"  Total steps   : {global_step:,}")
    print(f"  Best val_loss : {best_val_loss:.4f}")
    print(f"{'='*60}")
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT fine-tuning for CodeGen-150M")
    parser.add_argument(
        "--base-checkpoint", type=str, default=None,
        help="Path to the pre-trained checkpoint. Defaults to latest in ./checkpoints/"
    )
    args = parser.parse_args()

    if args.base_checkpoint is None:
        ckpts = sorted(
            glob.glob(os.path.join(config.CHECKPOINT_DIR, "step_*")),
            key=lambda d: int(d.split("_")[-1])
        )
        if not ckpts:
            print("ERROR: No pre-training checkpoints found!")
            sys.exit(1)
        args.base_checkpoint = ckpts[-1]
        print(f"Using latest checkpoint: {args.base_checkpoint}")

    train_sft(args.base_checkpoint)
