# Code Generation at Scale: Pre-Training and Supervised Fine-Tuning of GPT-2 Style Transformers

## Abstract
This project presents a systematic scaling study of autoregressive transformer models trained entirely from scratch for Python code generation. We design, implement, and train three GPT-2 style decoder-only models at increasing scales (50M, 150M, and 225M parameters) on filtered subsets of The Stack Dedup. These models are trained on approximately ~819M, ~2B, and ~3B tokens of Python code respectively.

The models were subsequently fine-tuned using Supervised Fine-Tuning (SFT) on the `iamtarun/python_code_instructions_18k_alpaca` dataset utilizing response-only loss masking. Our results demonstrate that scaling from 50M to 225M parameters yields measurable gains in syntactic correctness and instruction-following ability.

## Repository Structure
- `config.py`: Configuration classes for model architectures, training hyperparameters, and dataset settings.
- `data_pipeline.py`: Implementation of the data ingestion pipeline, including quality filtering, StarCoder2 tokenization, and tight sequence packing.
- `model.py`: Custom PyTorch implementation of the GPT-2 decoder-only architecture with features like Pre-LayerNorm and Weight Tying.
- `main.py`: The primary entry point for pre-training the base models from random initialization.
- `sft_train.py`: Script for Supervised Fine-Tuning (SFT) the pre-trained base models into interactive assistants.
- `evaluate.py`: Evaluation suite for calculating validation perplexity and syntactic pass rate.

## Model Architecture
All three models follow the GPT-2 decoder-only transformer design:
- **50M Model**: 8 layers, 512 embedding dim, 8 attention heads. Implemented in custom PyTorch.
- **150M Model**: 12 layers, 768 embedding dim, 12 attention heads. Implemented in custom PyTorch.
- **225M Model**: 16 layers, 960 embedding dim, 16 attention heads. Uses HuggingFace's `GPT2LMHeadModel` as its backbone.

### Key Architectural Decisions
- **Pre-LayerNorm**: Layer normalization is applied before each sub-layer to dramatically improve gradient flow in early training.
- **Weight Tying**: The language model head shares its weight matrix with the token embedding table, saving significant parameters and acting as a strong regularizer.
- **Flash Attention**: Uses PyTorch 2.0's `F.scaled_dot_product_attention` for near-linear memory complexity, enabling 1,024 context lengths on 16GB VRAM.
- **NewGELU Activation**: Employs the `gelu_new` tanh approximation matching the original GPT-2 activation.
- **Residual Projection Scaling**: Output projections are scaled down by `1 / sqrt(2 * N_layer)` at initialization to keep residual stream variance approximately constant.

## Dataset & Preprocessing
- **Pre-Training Data**: The Python subset of The Stack Dedup (`bigcode/the-stack-dedup`).
- **Tokenizer**: StarCoder2 BPE tokenizer (vocab size: 49,152), optimized to minimize token fragmentation on Python syntax.
- **Quality Filtering**: Files are heuristically filtered based on character counts, line counts, alphanumeric ratios, and a syntax check using Python's `compile()`.
- **Tight Sequence Packing**: Sequences are packed into fixed-length blocks of 1,024 tokens with no padding to ensure 100% compute efficiency.

## Training Setup
- **Pre-Training**: Trained on single consumer and cloud GPUs (Kaggle T4 or local NVIDIA RTX 4060 Ada). Mixed precision (Float16 + GradScaler or BFloat16) used depending on hardware. Models utilize the AdamW optimizer with a linear warmup and cosine decay learning rate schedule. Hard gradient clipping is applied as a primary defense against loss spikes.
- **Supervised Fine-Tuning (SFT)**: Formatted using the standard Alpaca prompt template applied to 18,612 Python instruction-response pairs. Uses response-only loss masking to focus cross-entropy loss solely on the generated code, preventing the model from merely memorizing prompt templates.

## Evaluation & Findings
Models are evaluated on Validation Perplexity and Syntax Pass Rate (parsing the generated code with Python's `compile()`). 

### Quantitative Results (Post-SFT)
- The **150M SFT Model** achieved a 44% reduction in validation loss over 3 epochs of SFT, dropping to a perplexity of ~2.16.
- The **225M SFT Model** achieved a best post-SFT perplexity of 2.12 and a 100% syntax pass rate.

### Key Observations
1. **Syntax vs. Reasoning**: Syntactic correctness scales rapidly with both parameter count and token budget. However, algorithmic reasoning scales far more slowly; sub-billion parameter models can learn surface-level syntax perfectly but continue to struggle with multi-step reasoning or internal state tracking.
2. **Impact of SFT**: Supervised Fine-Tuning is highly transformative. A single epoch of response-only fine-tuning on 18k instruction pairs successfully converts a raw base model into a coherent interactive assistant, eliminating base-model pre-training artifacts (e.g., commenting out solutions or failing to respect instruction boundaries).
3. **Training Instability**: Early/mid-training gradient norm explosions in BFloat16 models can be mitigated through targeted interventions such as doubling gradient accumulation, increasing the Adam epsilon, and implementing gradient skip blocks.

## Usage

Start by adjusting the configurations inside `config.py` based on the targeted model scale and available hardware.

**1. Pre-training**
To begin pre-training a base model from scratch:
```bash
python main.py
```

**2. Evaluation**
To evaluate the validation perplexity and syntactic pass rate of a checkpoint:
```bash
python evaluate.py
```

**3. Supervised Fine-Tuning (SFT)**
To fine-tune a pre-trained base model for instruction following:
```bash
python sft_train.py
```
