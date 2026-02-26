"""Starter submission — copy this file and modify it.

Tokenization: digits 0-9, plus, equals, EOS → 13 tokens.
Format: zero-padded to 10 digits each.
  Input:  "0001234567+0009876543=" (22 tokens, always fixed length)
  Output: reversed digits of sum, zero-padded to 11 + EOS (12 tokens)

Zero-padding aligns digit positions so carry patterns are positionally consistent.
Reversed output order lets the model predict least-significant digit first.
"""

from pathlib import Path

import torch
import torch.nn as nn

from starter.model import AdditionTransformer

# --- Token vocabulary ---
# 0-9: digit tokens
# 10: '+'
# 11: '='
# 12: EOS
VOCAB_SIZE: int = 13
MAX_OUTPUT_LEN: int = 12  # 11-digit result (zero-padded) + EOS

TOKEN_PLUS = 10
TOKEN_EQ = 11
TOKEN_EOS = 12

NUM_DIGITS = 10  # pad inputs to this many digits
RESULT_DIGITS = 11  # max digits in sum of two 10-digit numbers


def build_model() -> tuple[nn.Module, dict]:
    """Build and return the model with metadata."""
    model = AdditionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=34,
        weight_tying=True,
    )

    # Load checkpoint if available
    ckpt_path = Path(__file__).parent / "checkpoint.pt"
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    metadata = {
        "name": "starter",
        "author": "addition-challenge",
        "architecture": "GPT-style transformer, 2 layers, 4 heads, d_model=64",
        "tricks": "weight tying, reversed output digits, zero-padded fixed-length input",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    """Encode two integers as token IDs with zero-padded fixed-length format.

    "0001234567+0009876543=" → always 22 tokens
    """
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    tokens = [int(d) for d in a_str] + [TOKEN_PLUS] + [int(d) for d in b_str] + [TOKEN_EQ]
    return tokens


def decode(tokens: list[int]) -> int:
    """Decode output tokens back to integer.

    Output is reversed digit order (least-significant first), zero-padded.
    Stops at EOS or first non-digit token.
    """
    digits = []
    for t in tokens:
        if t == TOKEN_EOS or t < 0 or t > 9:
            break
        digits.append(t)

    if not digits:
        return 0

    # Reverse because output is least-significant-digit first
    digits = digits[::-1]
    return int("".join(str(d) for d in digits))
