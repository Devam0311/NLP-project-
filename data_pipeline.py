"""
data_pipeline.py — Stream, filter, tokenise, and chunk Python code from
bigcode/the-stack-dedup, then save as a HuggingFace Arrow dataset to disk.

Design choices:
  - Streaming: avoids downloading the full ~3 TB corpus.
  - Quality filters: remove noise (auto-generated, junk, invalid syntax).
  - Tight packing: no padding — every token in a chunk is real code.
  - compile() filter: removes syntactically invalid files.
  - 99.5 / 0.5 train/val split.
  - Dataset.from_generator(): writes Arrow rows incrementally — RAM stays low.

Usage:
  HF_TOKEN=<token> python data_pipeline.py
"""

import os
import math
import warnings
from typing import Iterator, List

from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from huggingface_hub import login

import config

# Authenticate with HuggingFace (required for gated bigcode/the-stack-dedup)
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass  # Already logged in or token not needed


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def passes_quality_filter(content: str) -> bool:
    """
    Return True if the Python file passes all quality gates.

    Checks are ordered cheapest-first to fail fast:
      1. Length gate (200 – 80,000 chars)
      2. Minimum 8 lines
      3. Average line length 10 – 200 chars
      4. Alphanumeric ratio >= 0.20
      5. Auto-generated marker check in first 1 KB
      6. Must contain 'def ' or 'class '
      7. No null bytes
      8. Valid Python syntax via compile()  (most expensive, done last)
    """
    n = len(content)
    if n < config.FILTER_MIN_CHARS or n > config.FILTER_MAX_CHARS:
        return False

    lines = content.splitlines()
    if len(lines) < config.FILTER_MIN_LINES:
        return False

    avg_line = n / max(len(lines), 1)
    if not (config.FILTER_MIN_AVG_LINE <= avg_line <= config.FILTER_MAX_AVG_LINE):
        return False

    alnum = sum(c.isalnum() for c in content)
    if alnum / n < config.FILTER_MIN_ALPHA_RATIO:
        return False

    header = content[:1024].lower()
    for marker in config.FILTER_AUTOGEN_MARKERS:
        if marker in header:
            return False

    if "def " not in content and "class " not in content:
        return False

    if "\x00" in content:
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(content, "<string>", "exec")
    except (SyntaxError, ValueError):
        return False

    return True


# ---------------------------------------------------------------------------
# Token stream
# ---------------------------------------------------------------------------

def token_stream(tokenizer) -> Iterator[List[int]]:
    """
    Yield token-ID lists, one per file, from the streaming dataset.
    Appends EOS at each file boundary so the model learns where files end.
    """
    ds = load_dataset(
        config.HF_DATASET_ID,
        data_dir=config.HF_DATA_DIR,
        split=config.HF_SPLIT,
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    eos = tokenizer.eos_token_id

    for example in ds:
        content = example.get("content", "")
        if not passes_quality_filter(content):
            continue
        ids = tokenizer.encode(content, add_special_tokens=False)
        ids.append(eos)
        yield ids


# ---------------------------------------------------------------------------
# Pack tokens into fixed-length chunks
# ---------------------------------------------------------------------------

def chunk_generator(tokenizer) -> Iterator[dict]:
    """
    Tightly pack tokens into CONTEXT_LENGTH blocks with no padding.
    Labels == input_ids (standard causal LM objective; no masking needed).
    """
    block  = config.CONTEXT_LENGTH
    buffer: List[int] = []

    for ids in token_stream(tokenizer):
        buffer.extend(ids)
        while len(buffer) >= block:
            chunk  = buffer[:block]
            buffer = buffer[block:]
            yield {"input_ids": chunk}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset():
    """
    Stream -> filter -> tokenise -> chunk -> write to disk.

    Memory footprint: ~one chunk (1024 x 4 bytes = 4 KB) at a time.
    Arrow files are written incrementally by Dataset.from_generator().

    Output directory structure:
      DATA_CACHE_DIR/
        train/    (99.5% of chunks)
        val/      (0.5% of chunks)
    """
    os.makedirs(config.DATA_CACHE_DIR, exist_ok=True)

    print("Loading tokeniser ...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)
    print(f"  vocab size : {tokenizer.vocab_size}")
    print(f"  eos token  : {tokenizer.eos_token_id}")

    target_chunks = math.ceil(config.TARGET_TOKENS / config.CONTEXT_LENGTH)
    print(
        f"\nTarget: {config.TARGET_TOKENS:,} tokens"
        f" ~ {target_chunks:,} chunks of {config.CONTEXT_LENGTH} tokens"
    )

    total_tokens_seen = [0]

    def chunk_gen():
        for i, chunk in enumerate(chunk_generator(tokenizer)):
            yield chunk
            total_tokens_seen[0] += config.CONTEXT_LENGTH

            if (i + 1) % config.PROGRESS_EVERY_CHUNKS == 0:
                pct = 100 * total_tokens_seen[0] / config.TARGET_TOKENS
                print(
                    f"  [{i+1:>9,} chunks | "
                    f"{total_tokens_seen[0]:>14,} tokens | {pct:.1f}%]"
                )

            if i + 1 >= target_chunks:
                break

    print("\nStreaming chunks to disk (no RAM accumulation) ...")
    features = Features({"input_ids": Sequence(Value("int32"))})
    full_ds  = Dataset.from_generator(chunk_gen, features=features)

    n_chunks = len(full_ds)
    print(f"\nFull dataset: {n_chunks:,} chunks ({n_chunks * config.CONTEXT_LENGTH:,} tokens)")

    print("Shuffling ...")
    full_ds = full_ds.shuffle(seed=config.DATA_SEED)

    val_n   = max(1, int(n_chunks * config.VAL_FRACTION))
    train_n = n_chunks - val_n
    print(f"Train chunks : {train_n:,}")
    print(f"Val   chunks : {val_n:,}")

    train_path = os.path.join(config.DATA_CACHE_DIR, "train")
    val_path   = os.path.join(config.DATA_CACHE_DIR, "val")

    print("Saving train split ...")
    full_ds.select(range(train_n)).save_to_disk(train_path)
    print("Saving val split ...")
    full_ds.select(range(train_n, n_chunks)).save_to_disk(val_path)

    print("Cleaning up generator cache ...")
    full_ds.cleanup_cache_files()

    print(f"\nDone. Data saved to: {config.DATA_CACHE_DIR}")
    return train_path, val_path


if __name__ == "__main__":
    build_dataset()
