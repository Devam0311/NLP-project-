"""
model.py — GPT-2 style 150 M-parameter code-generation model.

Key design decisions:
  - GPT2LMHeadModel from HuggingFace gives a battle-tested forward/backward
    pass without reinventing attention.
  - Weight tying: input embedding = output projection (saves ~47M params,
    also improves optimisation as done in the original GPT-2).
  - Initialisation follows the GPT-2 paper:
      * Std 0.02 for all linear & embedding weights.
      * Residual projection weights are additionally scaled by
        1/sqrt(2 × n_layer) to keep the residual stream variance
        stable at initialisation depth.
  - Biases are zero-initialised (HuggingFace default).
"""

import math
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

import config


def build_config() -> GPT2Config:
    return GPT2Config(
        vocab_size          = config.VOCAB_SIZE,
        n_positions         = config.N_POSITIONS,
        n_embd              = config.N_EMBD,
        n_layer             = config.N_LAYER,
        n_head              = config.N_HEAD,
        n_inner             = config.N_INNER,
        resid_pdrop         = config.RESID_PDROP,
        embd_pdrop          = config.EMBD_PDROP,
        attn_pdrop          = config.ATTN_PDROP,
        activation_function = config.ACTIVATION,
        # Needed so HF doesn't add extra token-type embeddings
        bos_token_id        = 0,
        eos_token_id        = 0,
    )


def _init_weights(module: nn.Module, n_layer: int) -> None:
    """
    GPT-2 initialisation scheme.
    Linear & Embedding → N(0, 0.02).
    Residual projections (c_proj inside attention & MLP) additionally
    scaled by 1/sqrt(2 × n_layer) to keep activation variance bounded.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    # Identify residual projection layers by name convention used in
    # HuggingFace's GPT-2 implementation: attn.c_proj and mlp.c_proj.
    # We apply the extra scale *after* the standard init above.
    if isinstance(module, nn.Linear):
        name = getattr(module, "_is_residual_proj", False)
        if name:
            scale = 1.0 / math.sqrt(2 * n_layer)
            with torch.no_grad():
                module.weight.mul_(scale)


def _tag_residual_projections(model: GPT2LMHeadModel) -> None:
    """
    Tag the residual projection Linear layers with a marker attribute so
    _init_weights can identify them for the extra scale factor.
    HuggingFace GPT-2 names them as `attn.c_proj` and `mlp.c_proj`
    inside each transformer block.
    """
    for block in model.transformer.h:
        block.attn.c_proj._is_residual_proj = True
        block.mlp.c_proj._is_residual_proj  = True


def build_model(device: torch.device) -> GPT2LMHeadModel:
    """
    Instantiate, initialise, tie weights, verify, and return the model.
    """
    cfg   = build_config()
    model = GPT2LMHeadModel(cfg)

    # Tag residual projections before init so _init_weights can find them
    _tag_residual_projections(model)

    # Apply custom init to every submodule
    for module in model.modules():
        _init_weights(module, cfg.n_layer)

    # Weight tying: lm_head.weight ← wte.weight
    # This both reduces parameters and stabilises training.
    model.tie_weights()

    # Count parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    # Tied weights are shared, so unique storage is less than total count.

    # Verify a single forward pass before returning
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, config.CONTEXT_LENGTH, dtype=torch.long, device=device)
        out   = model(input_ids=dummy, labels=dummy)
        assert out.loss is not None, "Forward pass returned no loss!"
        print(f"Forward pass OK — dummy loss: {out.loss.item():.4f}")
    model.train()

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(device)
