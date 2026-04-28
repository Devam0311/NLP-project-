"""
Model setup for the CodeGen training run.

This file builds the GPT-style language model using HuggingFace's
GPT2LMHeadModel, applies the initialization we want, ties the
embedding/output weights, and runs a small forward-pass check.
"""

import math
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

import config


def build_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=config.VOCAB_SIZE,
        n_positions=config.N_POSITIONS,
        n_embd=config.N_EMBD,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_inner=config.N_INNER,
        resid_pdrop=config.RESID_PDROP,
        embd_pdrop=config.EMBD_PDROP,
        attn_pdrop=config.ATTN_PDROP,
        activation_function=config.ACTIVATION,

        # keep token type embeddings out of this setup
        bos_token_id=0,
        eos_token_id=0,
    )


def _init_weights(module: nn.Module, n_layer: int) -> None:
    """
    Initialize model weights in the GPT-2 style.

    Linear and embedding weights use a normal distribution.
    Residual projection layers get an extra scale factor.
    """

    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    if isinstance(module, nn.Linear):
        name = getattr(module, "_is_residual_proj", False)

        if name:
            scale = 1.0 / math.sqrt(2 * n_layer)

            with torch.no_grad():
                module.weight.mul_(scale)


def _tag_residual_projections(model: GPT2LMHeadModel) -> None:
    """
    Mark the attention and MLP projection layers so they can
    receive the extra residual scaling during initialization.
    """

    for block in model.transformer.h:
        block.attn.c_proj._is_residual_proj = True
        block.mlp.c_proj._is_residual_proj = True


def build_model(device: torch.device) -> GPT2LMHeadModel:
    """
    Build, initialize, check, and return the model.
    """

    cfg = build_config()
    model = GPT2LMHeadModel(cfg)

    _tag_residual_projections(model)

    for module in model.modules():
        _init_weights(module, cfg.n_layer)

    # share input embedding and output projection weights
    model.tie_weights()

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    model = model.to(device)

    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(
            1,
            config.CONTEXT_LENGTH,
            dtype=torch.long,
            device=device
        )

        out = model(
            input_ids=dummy,
            labels=dummy
        )

        assert out.loss is not None, "Forward pass returned no loss!"

        print(f"Forward pass OK — dummy loss: {out.loss.item():.4f}")

    model.train()

    return model


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = build_model(device)
