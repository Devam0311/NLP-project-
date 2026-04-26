"""
evaluate.py — Validation loss, qualitative generation, and syntax pass rate.

Called from train.py at every checkpoint, or standalone:
  python evaluate.py --checkpoint ./checkpoints/step_0005000
"""

import os
import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
import wandb

import config
from model import GPT

amp_dtype = torch.bfloat16 if config.DTYPE == "bfloat16" else torch.float16


# ---------------------------------------------------------------------------
# A. Validation loss + perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_loss(model: GPT, device: torch.device) -> "tuple[float, float]":
    """Compute val loss and perplexity (capped at 50 batches for speed)."""
    val_path = os.path.join(config.DATA_CACHE_DIR, "val")
    val_ds   = load_from_disk(val_path)
    val_ds.set_format(type="torch", columns=["input_ids"])

    val_loader = DataLoader(
        val_ds,
        batch_size  = config.PER_DEVICE_BATCH,
        shuffle     = False,
        num_workers = 1,
        pin_memory  = True,
    )

    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    max_batches = 200

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device).long()
        labels    = input_ids.clone()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(input_ids=input_ids, labels=labels)
        total_loss += out.loss.item()
        n_batches  += 1

    model.train()
    avg_loss   = total_loss / max(n_batches, 1)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# ---------------------------------------------------------------------------
# B. Qualitative generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model: GPT, tokenizer, device: torch.device) -> "list[str]":
    """Run generation on all EVAL_PROMPTS and return decoded strings."""
    model.eval()
    outputs = []

    for prompt in config.EVAL_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            gen_ids = model.generate(
                input_ids,
                max_new_tokens     = config.GEN_MAX_NEW_TOKENS,
                temperature        = config.GEN_TEMPERATURE,
                top_p              = config.GEN_TOP_P,
                top_k              = config.GEN_TOP_K,
                do_sample          = config.GEN_DO_SAMPLE,
                repetition_penalty = config.GEN_REPETITION_PEN,
                pad_token_id       = tokenizer.eos_token_id,
                eos_token_id       = tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        if "### Instruction:" in text[len(prompt):]:
            text = text[:len(prompt)] + text[len(prompt):].split("### Instruction:")[0].strip()
        outputs.append(text)

    model.train()
    return outputs


# ---------------------------------------------------------------------------
# C. Syntax pass rate
# ---------------------------------------------------------------------------

def syntax_pass_rate(texts: "list[str]") -> float:
    """Fraction of generated strings that are valid Python."""
    if not texts:
        return 0.0
    passed = sum(1 for t in texts if _is_valid_python(t))
    return passed / len(texts)


def _is_valid_python(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Combined evaluation entry-point
# ---------------------------------------------------------------------------

def run_eval(model: GPT, device: torch.device, step: int):
    """Run the full evaluation suite and log results to WandB + file."""
    print(f"\n{'='*60}")
    print(f"  Evaluation @ step {step:,}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)

    # A. Validation loss
    val_loss, val_ppl = compute_val_loss(model, device)
    print(f"  val_loss       : {val_loss:.4f}")
    print(f"  val_perplexity : {val_ppl:.2f}")

    # B. Generation
    samples = generate_samples(model, tokenizer, device)
    for i, (prompt, out) in enumerate(zip(config.EVAL_PROMPTS, samples)):
        print(f"\n--- Prompt {i+1} ---")
        print(out[:500])

    # C. Syntax pass rate
    spr = syntax_pass_rate(samples)
    print(f"\n  syntax_pass_rate : {spr:.2%}")
    print(f"{'='*60}\n")

    # Save to file
    os.makedirs(config.EVAL_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.EVAL_OUTPUT_DIR, f"step_{step:07d}.txt")
    with open(out_path, "w") as f:
        f.write(f"Step {step}\n{'='*60}\n\n")
        f.write(f"val_loss        : {val_loss:.4f}\n")
        f.write(f"val_perplexity  : {val_ppl:.2f}\n")
        f.write(f"syntax_pass_rate: {spr:.2%}\n\n")
        for i, (prompt, out_text) in enumerate(zip(config.EVAL_PROMPTS, samples)):
            f.write(f"\n--- Prompt {i+1} ---\n{out_text}\n{'*'*40}\n")

    # Log to WandB
    wandb.log({
        "val_loss":         val_loss,
        "val_perplexity":   val_ppl,
        "syntax_pass_rate": spr,
        "step":             step,
    }, step=step)


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def standalone_eval(checkpoint_dir: str):
    """Evaluate a saved checkpoint without a running training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint_dir} ...")
    model = GPT.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    wandb.init(project=config.WANDB_PROJECT, name="standalone-eval", mode="offline")
    try:
        step = int(Path(checkpoint_dir).name.split("_")[-1])
    except ValueError:
        step = 0
    run_eval(model, device, step)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a checkpoint directory (e.g. ./checkpoints/step_0005000)"
    )
    args = parser.parse_args()
    standalone_eval(args.checkpoint)
