# Addition Challenge

Train the smallest autoregressive transformer to add two 10-digit numbers.

Your model sees `0001234567+0009876543=` and predicts the answer one token at a time. **Smallest model with ≥99% accuracy wins.**

Inspired by [AdderBoard](https://github.com/anadim/AdderBoard) — but here, the evaluation harness controls generation, making autoregressive behavior structurally enforced rather than honor-system.

## Quick Start

```bash
git clone <repo-url>
cd addition-challenge
make install                            # uv sync
make train                              # Train starter model (~2 min)
make eval                               # Evaluate (~69K params, 100%)
```

Iterate on your own submission:

```bash
cp starter/submission.py my_submission.py
# Edit my_submission.py...
make eval SUBMISSION=my_submission.py
```

## Baselines

Converted from the [AdderBoard leaderboard](https://github.com/anadim/AdderBoard) and verified against our harness:

| Submission | Params | Accuracy | Type | Original Author |
|---|---|---|---|---|
| `baselines/wonderfall_121.py` | **121** | 100.00% | Hand-coded | [Wonderfall](https://gist.github.com/Wonderfall/7d6f49aa6703352f94d3d80b4cd31e15) |
| `baselines/cosminscn_130.py` | **130** | 100.00% | Hand-coded | [cosminscn](https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b) |
| `baselines/rezabyt_311.py` | **311** | 100.00% | Trained | [rezabyt](https://github.com/rezabyt/digit-addition-311p) |
| `baselines/yinglunz_456.py` | **456** | 100.00% | Trained | [yinglunz](https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition) |
| `baselines/anadim_1644.py` | **1,644** | 99.14% | Trained | [anadim](https://github.com/anadim/smallest-addition-transformer-codex) |
| `baselines/anadim_6080.py` | **6,080** | 100.00% | Trained | [anadim](https://github.com/anadim/smallest-addition-transformer-claude-code) |
| `starter/submission.py` | **69,184** | 100.00% | Trained | (starter) |

The frontier is **121 parameters**. Can you do better?

## What You Submit

A single `.py` file with 5 exports:

| Export | Type | Description |
|---|---|---|
| `build_model()` | `() -> (nn.Module, dict)` | Return your model and a metadata dict |
| `encode(a, b)` | `(int, int) -> list[int]` | Convert two integers to input token IDs |
| `decode(tokens)` | `(list[int]) -> int` | Convert output token IDs back to an integer |
| `VOCAB_SIZE` | `int` | Number of distinct tokens (max 256) |
| `MAX_OUTPUT_LEN` | `int` | Max output tokens to generate (max 30) |

See `starter/submission.py` for a working example.

## How Evaluation Works

**The harness controls generation, not your model.** Your model only provides `forward(token_ids) -> logits`. The harness:

1. Calls `encode(a, b)` to get input tokens
2. Feeds tokens to your model one step at a time
3. Takes `argmax` of the last position's logits
4. Appends the predicted token to the context and repeats
5. Calls `decode(output_tokens)` to get the answer

Your model never sees future output tokens — autoregressive behavior is enforced by construction. This is the key difference from AdderBoard, which lets users control the generation loop.

### Validation Checks

Before evaluation, submissions pass through layered anti-cheat validation:

- **Structural check** — model must contain self-attention layers
- **Causal check** — prefix consistency test proves causal masking
- **Encode bounds** — token count ≤25, values in `[0, VOCAB_SIZE)`
- **Encode consistency** — encode must not smuggle the answer
- **Decode honesty** — random garbage tokens must not produce valid answers
- **AST safety** — no global state mutation, dangerous imports, or eval/exec

## Scoring

- **Metric**: unique trainable parameter count (lower is better)
- **Threshold**: ≥99% accuracy on 10,010 test cases (10 edge cases + 10,000 random pairs, seed=2025)
- **Deduplication**: weight-tied parameters counted once (`data_ptr()`)
- **Frozen params**: `requires_grad=False` parameters are excluded

## Rules

1. Submission must be a single `.py` file with all 5 required exports
2. Model must be an `nn.Module` with `forward(token_ids) -> logits`
3. Model must contain at least one self-attention layer with causal masking
4. `encode` and `decode` must be pure functions — no global state, no side channels
5. `VOCAB_SIZE` ≤ 256, `MAX_OUTPUT_LEN` ≤ 30, `encode` output ≤ 25 tokens
6. Hand-coded weights are allowed — this is about representation, not just learning

## Tips

- **Reversed digit order** helps carry propagation (predict ones digit first)
- **Zero-padded fixed-length** inputs align digit positions for the model
- **Weight tying** between embedding and output projection saves params for free
- **Low-rank factorizations** (rank-1 to rank-3) dramatically reduce params
- **RoPE** can route digit information without learned positional embeddings
- **Paired tokens** (encoding both operand digits at each position) compress input
- Study the baselines — they demonstrate the state of the art

## CLI Reference

```bash
make install                            # Install dependencies (uv sync)
make validate                           # Structural + behavioral validation
make eval                               # Full evaluation (10,010 test cases)
make eval-quick                         # Quick eval (110 test cases)
make count                              # Just parameter count
make train                              # Train starter model
make test                               # Run test suite

# Custom submission:
make eval SUBMISSION=my_submission.py
```

Or use the CLI directly:

```bash
uv run addition validate starter/submission.py
uv run addition eval starter/submission.py --num-random 100 --device cpu
uv run addition count-params starter/submission.py
```

## Project Structure

```
addition-challenge/
├── README.md
├── pyproject.toml
├── Makefile
├── .gitignore
│
├── addition_challenge/           # Evaluation harness
│   ├── cli.py                    # CLI: validate, eval, count-params
│   ├── harness.py                # Autoregressive generation loop
│   ├── validator.py              # Anti-cheat validation (7 layers)
│   ├── evaluator.py              # Accuracy evaluation engine
│   ├── param_counter.py          # Unique parameter counting
│   ├── test_cases.py             # Edge + random test generation
│   └── loader.py                 # Dynamic submission import
│
├── starter/                      # Working starter (~69K params)
│   ├── submission.py             # Copy and modify this
│   ├── model.py                  # GPT-style transformer
│   ├── train.py                  # Training script
│   └── checkpoint.pt             # Pre-trained weights
│
├── baselines/                    # AdderBoard submissions (121–6080 params)
│   ├── wonderfall_121.py         # 121 params, hand-coded Qwen3+RoPE
│   ├── cosminscn_130.py          # 130 params, hand-coded rank-1 GPT
│   ├── rezabyt_311.py            # 311 params, trained low-rank
│   ├── yinglunz_456.py           # 456 params, trained low-rank
│   ├── anadim_1644.py            # 1,644 params, trained paired-token
│   ├── anadim_6080.py            # 6,080 params, trained GPT
│   ├── lowrank_model.py          # Shared architecture for rezabyt/yinglunz
│   └── checkpoints/              # Pre-trained weights
│
└── tests/
    ├── test_harness.py
    ├── test_validator.py
    └── test_param_counter.py
```
