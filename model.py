"""
model.py — GPT-2 style decoder-only transformer written from scratch in PyTorch.

Architecture (~150M params):
  VOCAB_SIZE  = 49152
  N_POSITIONS = 1024
  N_EMBD      = 768
  N_LAYER     = 12
  N_HEAD      = 12
  N_INNER     = 3072   (4 x n_embd)

Design decisions:
  - Pre-norm: LayerNorm before each sub-layer (GPT-2 style)
  - Weight tying: lm_head.weight = wte.weight  (saves ~38M params)
  - Flash attention via PyTorch 2.0 F.scaled_dot_product_attention (is_causal=True)
    with a manual masked fallback for older PyTorch versions
  - NewGELU activation: tanh approximation matching GPT-2's gelu_new
  - Residual projections scaled by 1/sqrt(2 * n_layer) for stable depth
  - Normal(0, 0.02) weight init; zeros for biases
  - No dropout (0.0 everywhere — modern practice for small models)

Public interface:
  forward(input_ids, labels, attention_mask) -> ModelOutput(.loss, .logits)
  save_pretrained(path) / from_pretrained(path)
  generate(input_ids, ...)
  tie_weights()  (no-op: already tied in __init__)
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Output container (mirrors HuggingFace CausalLMOutput interface)
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    loss:   Optional[torch.Tensor]
    logits: torch.Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses PyTorch 2.0+ F.scaled_dot_product_attention with is_causal=True
    when available (triggers Flash Attention / Memory-Efficient kernels).
    Falls back to explicit O(T^2) masked attention on older PyTorch.
    """

    def __init__(self, n_embd: int, n_head: int, n_positions: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head   = n_head
        self.n_embd   = n_embd
        self.head_dim = n_embd // n_head

        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")

        # Fused Q, K, V projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        # Output projection — tagged for residual-depth scaling
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.c_proj._is_residual_proj = True

        # Causal mask only needed for manual fallback
        if not self.use_sdpa:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(n_positions, n_positions))
                      .view(1, 1, n_positions, n_positions),
            )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Build the 4D attention mask:
        #   - Always causal (lower-triangular)
        #   - If attention_mask is provided (2D: B,T), also mask padded positions
        if attention_mask is not None:
            # Causal mask: (1, 1, T, T) — True where attention is ALLOWED
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            # Padding mask: (B, 1, 1, T) — True where token is REAL (not pad)
            pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            # Combined: (B, 1, T, T) — attend only to non-pad, causal positions
            combined = causal & pad_mask  # broadcasts to (B, 1, T, T)
            # Convert to float mask for SDPA: 0.0 = attend, -inf = mask
            attn_mask_4d = torch.zeros_like(combined, dtype=q.dtype)
            attn_mask_4d.masked_fill_(~combined, float("-inf"))
        else:
            attn_mask_4d = None

        if self.use_sdpa:
            if attn_mask_4d is not None:
                # Use explicit mask (not is_causal) when we have padding
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask_4d,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                # No padding — use fast is_causal path (Flash Attention)
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
        else:
            # Manual masked attention fallback
            scale = 1.0 / math.sqrt(self.head_dim)
            att   = (q @ k.transpose(-2, -1)) * scale
            if attn_mask_4d is not None:
                att = att + attn_mask_4d
            else:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att   = F.softmax(att, dim=-1)
            y     = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class NewGELU(nn.Module):
    """
    GPT-2 gelu_new activation (tanh approximation).
    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Max absolute error vs exact GELU ~ 1.8e-4 per element.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            ))
        )


class MLP(nn.Module):
    """Position-wise feed-forward: Linear -> NewGELU -> Linear."""

    def __init__(self, n_embd: int, n_inner: int):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, n_inner, bias=True)
        self.c_proj = nn.Linear(n_inner, n_embd, bias=True)
        self.c_proj._is_residual_proj = True
        self.act    = NewGELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    """Single transformer block: pre-norm attention + pre-norm MLP."""

    def __init__(self, n_embd: int, n_head: int, n_inner: int, n_positions: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, n_positions)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, n_inner)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)   # residual + attention
        x = x + self.mlp(self.ln_2(x))    # residual + MLP
        return x


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT-2 style decoder-only transformer (~150M parameters).

    Approximate parameter count (N_EMBD=768, N_LAYER=12, VOCAB_SIZE=49152):
      wte  (token emb)      : 49152 x 768  = 37,748,736
      wpe  (pos emb)        : 1024  x 768  =    786,432
      12 x Block:
        ln_1 + ln_2         :   12 x 2 x 768 =   18,432
        c_attn (QKV)        :   12 x 768x2304 = 21,233,664  + bias
        c_proj (attn out)   :   12 x 768x768  =  7,077,888  + bias
        c_fc   (MLP up)     :   12 x 768x3072 = 28,311,552  + bias
        c_proj (MLP down)   :   12 x 3072x768 = 28,311,552  + bias
      ln_f                  : 768
      lm_head               : shares wte weights -> 0 extra params
      ---------------------------------------------------------------
      Total (unique params) : ~123.5M non-embedding + 37.75M shared wte
      Total (all params)    : ~161M  (wte counted once due to weight tying)
    """

    def __init__(
        self,
        vocab_size:  int,
        n_positions: int,
        n_embd:      int,
        n_layer:     int,
        n_head:      int,
        n_inner:     int,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.n_positions = n_positions
        self.n_embd      = n_embd
        self.n_layer     = n_layer
        self.n_head      = n_head
        self.n_inner     = n_inner

        self.wte     = nn.Embedding(vocab_size, n_embd)
        self.wpe     = nn.Embedding(n_positions, n_embd)
        self.h       = nn.ModuleList([
            Block(n_embd, n_head, n_inner, n_positions)
            for _ in range(n_layer)
        ])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: lm_head and wte share the same parameter matrix.
        # Saves ~38M params and improves optimisation (both ends of the
        # embedding space are trained with every forward pass).
        self.lm_head.weight = self.wte.weight

        # Initialise all weights
        self.apply(self._init_weights)

        # Scale residual projections to keep residual stream variance stable
        # at initialisation: each layer adds variance, so we scale down by
        # 1/sqrt(2 * n_layer) as described in the GPT-2 paper (section 2.3).
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_is_residual_proj", False):
                with torch.no_grad():
                    module.weight.mul_(1.0 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 weight init: Normal(0, 0.02) for weights, zeros for biases."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def tie_weights(self):
        """No-op: weights are already tied in __init__."""
        pass

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:      torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,   # accepted but unused
    ) -> ModelOutput:
        B, T = input_ids.shape
        assert T <= self.n_positions, f"Sequence length {T} > max {self.n_positions}"

        pos = torch.arange(T, device=input_ids.device)
        x   = self.wte(input_ids) + self.wpe(pos)

        for block in self.h:
            x = block(x, attention_mask=attention_mask)

        x      = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # Causal LM loss: logits[t] predicts token[t+1]
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
            shift_labels = labels[..., 1:].contiguous()        # (B, T-1)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        """Save weights (model.pt) and architecture config (config.json)."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        cfg = {
            "vocab_size":  self.vocab_size,
            "n_positions": self.n_positions,
            "n_embd":      self.n_embd,
            "n_layer":     self.n_layer,
            "n_head":      self.n_head,
            "n_inner":     self.n_inner,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "GPT":
        """Load a model saved with save_pretrained()."""
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        model = cls(**cfg)
        state = torch.load(
            os.path.join(path, "model.pt"),
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state)
        return model

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids:          torch.Tensor,
        max_new_tokens:     int   = 200,
        temperature:        float = 1.0,
        top_k:              int   = 50,
        top_p:              float = 0.95,
        do_sample:          bool  = True,
        repetition_penalty: float = 1.0,
        pad_token_id:       Optional[int] = None,
        eos_token_id:       Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressive generation with temperature / top-k / top-p sampling.

        Args:
            input_ids:          (1, T) prompt token ids on the correct device
            max_new_tokens:     number of new tokens to generate
            temperature:        softmax temperature (<1 = sharper, >1 = flatter)
            top_k:              keep only top-k logits before sampling
            top_p:              nucleus sampling: keep tokens until cumulative
                                prob exceeds top_p
            do_sample:          True = sample, False = greedy argmax
            repetition_penalty: penalise tokens already in the context (>1.0)
            pad_token_id:       accepted for API compatibility; not applied
            eos_token_id:       stop early when this token is generated

        Returns:
            (1, T + n_new) token id tensor
        """
        for _ in range(max_new_tokens):
            ctx    = input_ids[:, -self.n_positions:]
            logits = self.forward(ctx).logits[:, -1, :]  # (B, vocab)

            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            if do_sample:
                logits = logits / max(temperature, 1e-8)

                if top_k > 0:
                    kth_val = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits  = logits.masked_fill(logits < kth_val, float("-inf"))

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs   = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove_mask] = float("-inf")
                    logits.scatter_(1, sorted_idx, sorted_logits)

                probs      = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


# ---------------------------------------------------------------------------
# Factory function  (called by train.py, evaluate.py, sft_train.py)
# ---------------------------------------------------------------------------

def build_model(device: torch.device) -> GPT:
    """
    Instantiate, verify parameter count, run forward-pass sanity check,
    and return the model ready for training.
    """
    model = GPT(
        vocab_size  = config.VOCAB_SIZE,
        n_positions = config.N_POSITIONS,
        n_embd      = config.N_EMBD,
        n_layer     = config.N_LAYER,
        n_head      = config.N_HEAD,
        n_inner     = config.N_INNER,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total parameters     : {total:,}")
    print(f"[model] Trainable parameters : {trainable:,}")
    print(f"[model] wte/lm_head shared   : {model.wte.weight.numel():,} params")

    if hasattr(F, "scaled_dot_product_attention"):
        print("[model] ✓ PyTorch SDPA (flash attention) available")
    else:
        print("[model] ✗ SDPA unavailable — using manual attention fallback")

    model = model.to(device)

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, config.CONTEXT_LENGTH, dtype=torch.long, device=device)
        out   = model(input_ids=dummy, labels=dummy)
        assert out.loss is not None, "Forward pass returned no loss!"
        print(f"[model] Forward pass OK — dummy loss: {out.loss.item():.4f}")
    model.train()

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    build_model(device)
