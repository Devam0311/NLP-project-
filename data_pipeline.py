"""
Builds the training dataset by streaming Python code from
bigcode/the-stack-dedup, filtering it, tokenizing it, and
packing everything into fixed-length chunks saved to disk.

Uses streaming so the full corpus never needs to be downloaded.
"""

import os
import math
import warnings
from typing import Iterator, List

from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from huggingface_hub import login

import config


# HuggingFace login if token is provided
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass


# Quality filtering

def passes_quality_filter(content: str) -> bool:
    """
    Basic filtering to remove noisy or unusable files.

    Checks include:
    - file size limits
    - enough lines
    - reasonable line lengths
    - enough alphanumeric content
    - skip generated files
    - require code structure (def/class)
    - ignore broken syntax
    """
    n = len(content)

    if n < config.FILTER_MIN_CHARS or n > config.FILTER_MAX_CHARS:
        return False

    lines = content.splitlines()

    if len(lines) < config.FILTER_MIN_LINES:
        return False

    avg_line = n / max(len(lines), 1)

    if not (
        config.FILTER_MIN_AVG_LINE
        <= avg_line
        <= config.FILTER_MAX_AVG_LINE
    ):
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


# Streaming tokenized files

def token_stream(tokenizer) -> Iterator[List[int]]:
    """
    Streams files from dataset and yields one tokenized file at a time.
    Adds EOS between files so model can learn file boundaries.
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

        ids = tokenizer.encode(
            content,
            add_special_tokens=False
        )

        ids.append(eos)

        yield ids


# Pack tokens into training chunks

def chunk_generator(tokenizer) -> Iterator[dict]:
    """
    Packs incoming token stream into fixed-size blocks.
    No padding — every token is real training data.
    """

    block = config.CONTEXT_LENGTH
    buffer: List[int] = []

    for ids in token_stream(tokenizer):

        buffer.extend(ids)

        while len(buffer) >= block:
            chunk = buffer[:block]
            buffer = buffer[block:]

            yield {
                "input_ids": chunk
            }


# Main dataset pipeline

def build_dataset():
    """
    Full pipeline:
    stream -> filter -> tokenize -> chunk -> save train/val datasets
    """

    os.makedirs(
        config.DATA_CACHE_DIR,
        exist_ok=True
    )

    print("Loading tokenizer ...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_ID
    )

    print(f"vocab size : {tokenizer.vocab_size}")
    print(f"eos token  : {tokenizer.eos_token_id}")

    target_chunks = math.ceil(
        config.TARGET_TOKENS /
        config.CONTEXT_LENGTH
    )

    print(
        f"\nTarget: {config.TARGET_TOKENS:,} tokens"
        f" ~ {target_chunks:,} chunks"
    )

    total_tokens_seen = [0]

    def chunk_gen():

        for i, chunk in enumerate(
            chunk_generator(tokenizer)
        ):

            yield chunk

            total_tokens_seen[0] += config.CONTEXT_LENGTH

            if (i + 1) % config.PROGRESS_EVERY_CHUNKS == 0:

                pct = (
                    100 *
                    total_tokens_seen[0] /
                    config.TARGET_TOKENS
                )

                print(
                    f"[{i+1:>9,} chunks | "
                    f"{total_tokens_seen[0]:>14,} tokens | "
                    f"{pct:.1f}%]"
                )

            if i + 1 >= target_chunks:
                break


    print("\nStreaming chunks to disk ...")

    features = Features(
        {
            "input_ids": Sequence(
                Value("int32")
            )
        }
    )

    full_ds = Dataset.from_generator(
        chunk_gen,
        features=features
    )

    n_chunks = len(full_ds)

    print(
        f"\nFull dataset: {n_chunks:,} chunks "
        f"({n_chunks * config.CONTEXT_LENGTH:,} tokens)"
    )

    print("Shuffling dataset ...")

    full_ds = full_ds.shuffle(
        seed=config.DATA_SEED
    )

    val_n = max(
        1,
        int(
            n_chunks *
            config.VAL_FRACTION
        )
    )

    train_n = n_chunks - val_n

    print(f"Train chunks : {train_n:,}")
    print(f"Val chunks   : {val_n:,}")

    train_path = os.path.join(
        config.DATA_CACHE_DIR,
        "train"
    )

    val_path = os.path.join(
        config.DATA_CACHE_DIR,
        "val"
    )

    print("Saving train split ...")
    full_ds.select(
        range(train_n)
    ).save_to_disk(
        train_path
    )

    print("Saving val split ...")
    full_ds.select(
        range(train_n, n_chunks)
    ).save_to_disk(
        val_path
    )

    print("Cleaning cache ...")
    full_ds.cleanup_cache_files()

    print(
        f"\nDone. Data saved to: "
        f"{config.DATA_CACHE_DIR}"
    )

    return train_path, val_path


if __name__ == "__main__":
    build_dataset()
