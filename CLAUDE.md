# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Stanford CS336 Spring 2025 Assignment 1: Basics. The assignment involves implementing transformer components and a BPE tokenizer from scratch using PyTorch.

## Commands

```bash
uv run pytest                          # Run all tests
uv run pytest tests/test_model.py      # Run specific test file
uv run pytest tests/test_tokenizer.py  # Run tokenizer tests
uv run <file.py>                       # Run any Python file
uv run ruff check .                    # Run linter
uv run ruff format .                   # Format code
```

## Architecture

### Implementation Pattern
Implementations go in `cs336_basics/`. Tests use adapter functions in `tests/adapters.py` that call your implementations. To connect your code to tests:

1. Implement your class/function in `cs336_basics/`
2. Fill in the corresponding stub function in `tests/adapters.py`
3. Ensure your state dict keys match what adapters expect (e.g., Linear uses "W", Embedding uses "embedding_matrix", RMSNorm uses "g")

Adapter functions instantiate your classes and use `load_state_dict()` to load test weights, then verify outputs match reference implementations.

### Development Workflow
1. Read assignment handout for specifications
2. Implement components in `cs336_basics/`
3. Connect to tests via `tests/adapters.py`
4. Run tests: `uv run pytest tests/test_<component>.py`
5. Once tests pass, train models with `train.py`

### Components to Implement
- **Tokenization**: BPE training (`train_bpe`), tokenizer class with encode/decode
- **Neural Network Layers**: Linear, Embedding, RMSNorm, SwiGLU activation
- **Attention**: Scaled dot-product attention, multi-head self-attention with RoPE
- **Model**: Transformer block, Transformer language model
- **Training**: Softmax, cross-entropy loss, gradient clipping, AdamW optimizer, cosine LR schedule, checkpointing
- **Data**: Batch sampling from tokenized datasets

### Key Files
- `cs336_basics/bpe.py` - BPE tokenizer implementation
- `cs336_basics/model.py` - Neural network layers (Linear, Embedding, RMSNorm, SwiGLU, attention, transformer blocks)
- `cs336_basics/optimizer.py` - AdamW optimizer
- `cs336_basics/utils.py` - Training utilities (loss functions, gradient clipping, batching, checkpointing)
- `cs336_basics/pretokenization.py` - File chunking for parallel BPE training
- `tests/adapters.py` - Adapter functions connecting implementations to tests
- `tests/fixtures/` - Test data and reference outputs
- `train.py` - Complete training script with checkpointing and logging

## Data Setup

Download training data to `data/`:
```bash
mkdir -p data && cd data

# TinyStories dataset
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample (optional)
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Testing

Tests use snapshot-based assertions. Reference implementations provide expected outputs in `tests/fixtures/`. Tests initially fail with `NotImplementedError` until adapters are connected.

## Training

Train a Transformer language model:
```bash
uv run train.py --vocab_size 50257 --context_length 1024 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072
```

Key arguments:
- Model hyperparameters: `--vocab_size`, `--context_length`, `--d_model`, `--num_layers`, `--num_heads`, `--d_ff`, `--rope_theta`
- Optimizer settings: `--lr`, `--weight_decay`, `--betas`, `--eps`

The script uses TensorBoard for logging and automatically saves checkpoints.
