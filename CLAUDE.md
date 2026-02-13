# CLAUDE.md - AI Assistant Guide for microgpt-php

## Project Overview

A pure PHP port of Andrej Karpathy's [microgpt](https://karpathy.ai/microgpt.html) — the most atomic way to train and inference a GPT (Generative Pretrained Transformer) with zero dependencies. The original is ~250 lines of pure Python; this is the equivalent in pure PHP.

The model trains a character-level GPT on a dataset of names and then generates new, hallucinated names via autoregressive sampling.

## Repository Structure

```
microgpt/
├── CLAUDE.md          # This file — AI assistant guide
├── microgpt.php       # The entire implementation (single file)
└── input.txt          # Training data (auto-downloaded on first run)
```

There is exactly **one source file**: `microgpt.php`. Everything — autograd, model weights, forward pass, training loop, Adam optimizer, and inference — lives in that single file.

## Tech Stack

- **Language**: PHP 8.1+ (uses union types `Value|float`)
- **Dependencies**: None. Pure PHP, no Composer, no extensions, no frameworks.
- **Runtime**: CLI only (`php microgpt.php`)

## Architecture

### Core Components (in order of appearance in `microgpt.php`)

1. **`Value` class** (~80 lines) — Tiny automatic differentiation (autograd) engine. Every scalar is wrapped in a `Value` that tracks the computational graph for backpropagation. Supports: `add`, `mul`, `pow_`, `log_`, `exp_`, `relu`, `neg`, `sub`, `div`, and `backward()`.

2. **Helper functions**:
   - `gauss()` — Box-Muller transform for Gaussian random numbers
   - `matrix()` — Creates a 2D array of `Value` objects with random init
   - `linear()` — Matrix-vector multiply (a dense layer)
   - `softmax()` — Numerically stable softmax over `Value[]`
   - `rmsnorm()` — RMS normalization (used instead of LayerNorm)
   - `weighted_choice()` — Weighted random sampling for token generation

3. **Model configuration** — Hyperparameters defined as global variables:
   - `$n_embd = 16` (embedding dimension)
   - `$n_head = 4` (attention heads)
   - `$n_layer = 1` (transformer layers)
   - `$block_size = 16` (context window)

4. **State dict** — All model weights stored in a global `$state_dict` associative array with keys like `wte`, `wpe`, `lm_head`, `layer0.attn_wq`, etc.

5. **`gpt()` function** — The full GPT forward pass: token + positional embeddings, RMS normalization, multi-head causal self-attention with KV cache, feed-forward MLP with ReLU activation, and residual connections.

6. **Training loop** — 1000 steps of next-token prediction on the names dataset using Adam optimizer with linear learning rate decay.

7. **Inference** — Generates 20 new names via temperature-scaled autoregressive sampling.

### Key Architectural Choices (same as Karpathy's original)

- **RMS Normalization** instead of Layer Normalization
- **No biases** anywhere in the model
- **ReLU** activation (not GeLU) in the MLP
- **Character-level tokenizer** (each character = one token, plus a BOS/EOS token)
- **KV cache** during both training and inference

## How to Run

```bash
php microgpt.php
```

On first run, it auto-downloads `names.txt` (~29K names) from Karpathy's makemore repo. Training runs 1000 steps and then generates 20 sample names. No GPU required.

**Requirements**: PHP 8.1+ with `allow_url_fopen=On` (default).

**Expected runtime**: This is intentionally slow — pure scalar autograd with no vectorization. Expect it to take a long time. The purpose is educational, not performant.

## Development Conventions

### Code Style
- Single file, procedural + one class
- Global variables for model config and state (mirrors the Python original)
- Functions use PHPDoc `@param`/`@return` annotations for array types
- No namespaces, no autoloading — this is intentionally minimal

### Naming
- PHP function/variable names match the Python original as closely as possible
- PHP methods that shadow built-in names use trailing underscores: `pow_()`, `log_()`, `exp_()`
- The `Value` class method names are explicit verbs (`add`, `mul`, `div`) since PHP lacks operator overloading

### Key Differences from the Python Original
- `Value` operations are method calls (`$a->add($b)`) rather than operators (`a + b`)
- `SplObjectStorage` replaces Python's `set()` for object identity tracking in `backward()`
- `gauss()` uses Box-Muller since PHP has no `random.gauss()`
- `weighted_choice()` replaces Python's `random.choices()`
- PHP arrays are used where Python uses lists; `array_slice` replaces Python slicing

### Testing
There is no test suite. Correctness can be verified by:
1. Checking that training loss decreases over steps
2. Comparing generated names to the Python version's output (with same seed)

### Extending
- To change hyperparameters, edit the config section (`$n_embd`, `$n_head`, `$n_layer`, `$block_size`)
- To use a different dataset, replace `input.txt` with any newline-separated text file
- To change the number of training steps, edit `$num_steps`
- To change inference temperature, edit `$temperature`
