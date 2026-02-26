"""Baseline: rezabyt 311-param trained low-rank transformer.

Adapted from: https://github.com/rezabyt/digit-addition-311p
Original: 311 params, 99.999% accuracy, trained with grokking.
Architecture: 1-layer low-rank GPT, d=4, h=1, ff=8, rank-3 factorization,
              shared-A tied-KV, RMSNorm, weight tying.
"""

from pathlib import Path

import torch
import torch.nn as nn

from baselines.lowrank_model import ModelConfig, TinyDecoderLM

VOCAB_SIZE: int = 14  # 0-9, +, =, <PAD>, <EOS>
MAX_OUTPUT_LEN: int = 12  # 11 reversed digits + EOS

NUM_DIGITS = 10
SUM_DIGITS = 11
EOS_ID = 13


def build_model() -> tuple[nn.Module, dict]:
    cfg = ModelConfig(
        n_layer=1, d_model=4, n_head=1, d_ff=8, dropout=0.0,
        max_seq_len=33, vocab_size=14, pos_rank=3, qkv_rank=3,
        attn_out_rank=3, ffn_rank=3, use_rmsnorm=True, tie_qkv="shareA_tieKV",
    )
    model = TinyDecoderLM(cfg)
    ckpt_path = Path(__file__).parent / "checkpoints" / "rezabyt_311.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    metadata = {
        "name": "rezabyt-311",
        "author": "rezabyt",
        "architecture": "1-layer low-rank GPT, d=4, h=1, ff=8, rank-3",
        "tricks": "rank-3 factorization, shared-A tied-KV, RMSNorm, grokking, weight tying",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    """Zero-padded digits + '+' + digits + '=' (no BOS)."""
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    text = a_str + "+" + b_str + "="
    token_map = {str(i): i for i in range(10)}
    token_map["+"] = 10
    token_map["="] = 11
    return [token_map[ch] for ch in text]


def decode(tokens: list[int]) -> int:
    """Decode reversed digit tokens, stop at EOS or non-digit."""
    digits = []
    for t in tokens:
        if t == EOS_ID:
            break
        if 0 <= t <= 9:
            digits.append(str(t))
        else:
            break
    if not digits:
        return 0
    while len(digits) < SUM_DIGITS:
        digits.append("0")
    digits = digits[:SUM_DIGITS]
    return int("".join(digits)[::-1])
