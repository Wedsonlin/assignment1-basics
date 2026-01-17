# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Stanford CS336 Spring 2025 Assignment 1: Basics. The assignment involves implementing transformer components and a BPE tokenizer from scratch using PyTorch.

## Commands

```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_X.py    # Run specific test file
uv run <file.py>                 # Run any Python file
```

## Architecture

### Implementation Pattern
Implementations go in `cs336_basics/`. Tests use adapter functions in `tests/adapters.py` that call your implementations. To connect your code to tests, fill in the stub functions in `tests/adapters.py`.

### Components to Implement
- **Tokenization**: BPE training (`train_bpe`), tokenizer class with encode/decode
- **Neural Network Layers**: Linear, Embedding, RMSNorm, SwiGLU activation
- **Attention**: Scaled dot-product attention, multi-head self-attention with RoPE
- **Model**: Transformer block, Transformer language model
- **Training**: Softmax, cross-entropy loss, gradient clipping, AdamW optimizer, cosine LR schedule, checkpointing
- **Data**: Batch sampling from tokenized datasets

### Key Files
- `cs336_basics/bpe.py` - BPE tokenizer implementation
- `cs336_basics/pretokenization_example.py` - File chunking utilities for parallel processing
- `tests/adapters.py` - Adapter functions connecting implementations to tests
- `tests/fixtures/` - Test data and reference outputs

## Data Setup

Download training data to `data/`:
```bash
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

## Testing

Tests use snapshot-based assertions. Reference implementations provide expected outputs in `tests/fixtures/`. Tests initially fail with `NotImplementedError` until adapters are connected.
