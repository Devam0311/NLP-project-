"""
Supervised fine-tuning script for the CodeGen model.

Loads a pretrained checkpoint, fine-tunes it on instruction-code
pairs, runs periodic evaluation, and saves checkpoints.
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


amp_dtype = (
    torch.bfloat16
    if config.DTYPE=="bfloat16"
    else torch.float16
)


# Project paths for SFT artifacts
PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SFT_CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR,
    "sft_checkpoints"
)

SFT_EVAL_DIR = os.path.join(
    PROJECT_DIR,
    "sft_eval_outputs"
)

SFT_FINAL_DIR = os.path.join(
    PROJECT_DIR,
    "sft_final_model"
)



# Formatting helper

def format_sample(
    instruction,
    inp,
    output
):
    """
    Formats one instruction tuning example.
    """

    if inp and inp.strip():
        return (
            f"### Instruction:\n"
            f"{instruction.strip()}\n\n"
            f"### Input:\n"
            f"{inp.strip()}\n\n"
            f"### Response:\n"
            f"{output.strip()}"
        )

    return (
        f"### Instruction:\n"
        f"{instruction.strip()}\n\n"
        f"### Response:\n"
        f"{output.strip()}"
    )



class SFTDataset(Dataset):
    """
    Tokenized instruction dataset with response-only loss masking.
    """

    def __init__(
        self,
        samples,
        tokenizer,
        max_len
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        response_marker = "### Response:\n"

        skipped = 0

        for s in samples:

            text = format_sample(
                s["instruction"],
                s.get("input",""),
                s["output"]
            )

            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False
            )

            token_ids.append(
                tokenizer.eos_token_id
            )


            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]


            marker_pos = text.find(
                response_marker
            )

            if marker_pos == -1:
                skipped += 1
                continue


            prefix_text = text[
                :
                marker_pos
                +
                len(response_marker)
            ]

            prefix_ids = tokenizer.encode(
                prefix_text,
                add_special_tokens=False
            )

            prefix_len = min(
                len(prefix_ids),
                len(token_ids)
            )


            labels = (
                [-100]*prefix_len
                +
                token_ids[prefix_len:]
            )

            labels = labels[:len(token_ids)]


            if len(token_ids) < 10:
                skipped += 1
                continue


            self.data.append(
                {
                    "input_ids":token_ids,
                    "labels":labels
                }
            )

        print(
            f"SFTDataset: "
            f"{len(self.data):,} loaded, "
            f"{skipped} skipped"
        )


    def __len__(self):
        return len(
            self.data
        )


    def __getitem__(
        self,
        idx
    ):
        return self.data[idx]



def collate_fn(batch):
    """
    Pads a batch to equal sequence length.
    """

    max_len = max(
        len(
            item["input_ids"]
        )
        for item in batch
    )

    input_ids_padded=[]
    labels_padded=[]
    attention_mask=[]


    for item in batch:

        pad_len = (
            max_len
            -
            len(item["input_ids"])
        )

        input_ids_padded.append(
            item["input_ids"]
            +
            [0]*pad_len
        )

        labels_padded.append(
            item["labels"]
            +
            [-100]*pad_len
        )

        attention_mask.append(
            [1]*len(
                item["input_ids"]
            )
            +
            [0]*pad_len
        )


    return {
        "input_ids":
            torch.tensor(
                input_ids_padded,
                dtype=torch.long
            ),

        "labels":
            torch.tensor(
                labels_padded,
                dtype=torch.long
            ),

        "attention_mask":
            torch.tensor(
                attention_mask,
                dtype=torch.long
            )
    }



def get_sft_lr(
    step,
    total_steps
):
    """
    Warmup + cosine decay schedule.
    """

    if step < config.SFT_WARMUP_STEPS:
        return (
            step /
            max(
                1,
                config.SFT_WARMUP_STEPS
            )
        )

    progress = (
        step
        -
        config.SFT_WARMUP_STEPS
    ) / max(
        1,
        total_steps
        -
        config.SFT_WARMUP_STEPS
    )

    cosine = (
        0.5
        *
        (
            1.0
            +
            math.cos(
                math.pi*progress
            )
        )
    )

    min_ratio = (
        config.SFT_LR_MIN
        /
        config.SFT_LR
    )

    return (
        min_ratio
        +
        (
            1.0-min_ratio
        )*cosine
    )



@torch.no_grad()
def run_sft_eval(
    model,
    tokenizer,
    device,
    step,
    val_loader=None
):
    """
    Runs evaluation prompts and computes validation metrics.
    """

    model.eval()

    results=[]

    all_prompts=(
        config.EVAL_PROMPTS
        +
        config.SFT_INSTRUCTION_PROMPTS
    )


    prompt_labels=(
        [
            f"Code Prompt {i+1}"
            for i in range(
                len(config.EVAL_PROMPTS)
            )
        ]
        +
        [
            f"Instruction Prompt {i+1}"
            for i in range(
                len(
                    config.SFT_INSTRUCTION_PROMPTS
                )
            )
        ]
    )


    for label,prompt in zip(
        prompt_labels,
        all_prompts
    ):

        input_ids=tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(device)


        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype
        ):

            gen_ids=model.generate(
                input_ids,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )


        text=tokenizer.decode(
            gen_ids[0],
            skip_special_tokens=True
        )


        if "### Instruction:" in text[len(prompt):]:
            text=(
                text[:len(prompt)]
                +
                text[len(prompt):]
                .split(
                    "### Instruction:"
                )[0]
                .strip()
            )

        results.append(
            (
                label,
                prompt,
                text
            )
        )


    code_outputs=[
        r[2]
        for r in results[
            :
            len(
                config.EVAL_PROMPTS
            )
        ]
    ]

    syntax_rate=(
        sum(
            1
            for t in code_outputs
            if _is_valid_python(t)
        )
        /
        max(
            len(code_outputs),
            1
        )
    )


    val_loss=0.0

    if val_loader:

        total_loss=0.0
        n_batches=0

        for batch in val_loader:

            input_ids=batch[
                "input_ids"
            ].to(device)

            labels=batch[
                "labels"
            ].to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype
            ):
                out=model(
                    input_ids=input_ids,
                    labels=labels
                )

            total_loss+=out.loss.item()
            n_batches+=1


        val_loss=(
            total_loss
            /
            max(
                n_batches,
                1
            )
        )


    print(
        f"\nSFT Eval @ step {step}"
    )

    print(
        f"syntax_pass_rate:"
        f"{syntax_rate:.2%}"
    )


    wandb.log(
        {
            "sft_syntax_pass_rate":
                syntax_rate,
            "sft_step":
                step
        },
        step=step
    )

    model.train()

    return syntax_rate,val_loss



def _is_valid_python(
    code
):
    try:
        compile(
            code,
            "<string>",
            "exec"
        )
        return True
    except SyntaxError:
        return False



def save_sft_checkpoint(
    model,
    optimizer,
    step,
    epoch,
    train_loss
):
    """
    Saves model checkpoint.
    """

    ckpt_dir=os.path.join(
        SFT_CHECKPOINT_DIR,
        f"sft_step_{step:06d}"
    )

    os.makedirs(
        ckpt_dir,
        exist_ok=True
    )

    model.save_pretrained(
        ckpt_dir
    )

    torch.save(
        {
            "optimizer_state":
                optimizer.state_dict(),

            "step":step,
            "epoch":epoch,
            "train_loss":train_loss
        },
        os.path.join(
            ckpt_dir,
            "sft_train_state.pt"
        )
    )

    print(
        f"Saved checkpoint:"
        f"{ckpt_dir}"
    )



def train_sft(
    base_checkpoint
):

    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )


    os.makedirs(
        SFT_CHECKPOINT_DIR,
        exist_ok=True
    )


    print(
        f"\nLoading model "
        f"from {base_checkpoint}"
    )

    model=GPT(
        vocab_size=config.VOCAB_SIZE,
        n_positions=config.N_POSITIONS,
        n_embd=config.N_EMBD,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_inner=config.N_INNER
    ).to(device)


    state=GPT.from_pretrained(
        base_checkpoint
    ).state_dict()

    model.load_state_dict(
        state
    )

    model.train()


    tokenizer=AutoTokenizer.from_pretrained(
        config.TOKENIZER_ID
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token


    print(
        f"\nLoading SFT dataset..."
    )

    raw_ds=load_dataset(
        config.SFT_DATASET_ID,
        split="train"
    )

    split=raw_ds.train_test_split(
        test_size=config.SFT_VAL_FRACTION,
        seed=42
    )


    train_ds=SFTDataset(
        list(split["train"]),
        tokenizer,
        config.SFT_MAX_SEQ_LEN
    )

    val_ds=SFTDataset(
        list(split["test"]),
        tokenizer,
        config.SFT_MAX_SEQ_LEN
    )


    train_loader=DataLoader(
        train_ds,
        batch_size=config.SFT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )


    val_loader=DataLoader(
        val_ds,
        batch_size=config.SFT_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )


    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=config.SFT_LR
    )


    wandb.init(
        project=config.WANDB_PROJECT,
        name=f"sft-{config.WANDB_RUN_NAME}"
    )


    print(
        "\nStarting SFT training...\n"
    )


    global_step=0

    for epoch in range(
        config.SFT_EPOCHS
    ):

        print(
            f"Epoch "
            f"{epoch+1}/"
            f"{config.SFT_EPOCHS}"
        )

        optimizer.zero_grad()

        for micro_step,batch in enumerate(
            train_loader
        ):

            input_ids=batch[
                "input_ids"
            ].to(device)

            labels=batch[
                "labels"
            ].to(device)

            attention_mask=batch[
                "attention_mask"
            ].to(device)


            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype
            ):
                out=model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )

                loss=(
                    out.loss
                    /
                    config.SFT_GRAD_ACCUM
                )


            loss.backward()


            if (
                micro_step+1
            ) % config.SFT_GRAD_ACCUM==0:

                global_step+=1

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.SFT_GRAD_CLIP
                )

                optimizer.step()
                optimizer.zero_grad()


                if (
                    global_step
                    %
                    config.SFT_LOG_EVERY
                    ==0
                ):
                    print(
                        f"step "
                        f"{global_step} "
                        f"loss "
                        f"{loss.item():.4f}"
                    )


                if (
                    global_step
                    %
                    config.SFT_SAVE_EVERY
                    ==0
                ):
                    save_sft_checkpoint(
                        model,
                        optimizer,
                        global_step,
                        epoch+1,
                        loss.item()
                    )

                    run_sft_eval(
                        model,
                        tokenizer,
                        device,
                        global_step,
                        val_loader
                    )


    save_sft_checkpoint(
        model,
        optimizer,
        global_step,
        config.SFT_EPOCHS,
        loss.item()
    )

    run_sft_eval(
        model,
        tokenizer,
        device,
        global_step,
        val_loader
    )


    os.makedirs(
        SFT_FINAL_DIR,
        exist_ok=True
    )

    model.save_pretrained(
        SFT_FINAL_DIR
    )

    tokenizer.save_pretrained(
        SFT_FINAL_DIR
    )

    print(
        f"\nFinal model saved:"
        f"{SFT_FINAL_DIR}"
    )

    wandb.finish()



if __name__=="__main__":

    parser=argparse.ArgumentParser(
        description="SFT training"
    )

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None
    )

    args=parser.parse_args()


    if args.base_checkpoint is None:

        ckpts=sorted(
            glob.glob(
                os.path.join(
                    config.CHECKPOINT_DIR,
                    "step_*"
                )
            ),
            key=lambda d:
                int(
                    d.split("_")[-1]
                )
        )

        if not ckpts:
            print(
                "No checkpoints found"
            )
            sys.exit(1)

        args.base_checkpoint=ckpts[-1]


    train_sft(
        args.base_checkpoint
    )
```
