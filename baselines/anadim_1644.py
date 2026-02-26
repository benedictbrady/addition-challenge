"""Baseline: anadim 1,644-param trained transformer with paired tokens.

Adapted from: https://github.com/anadim/smallest-addition-transformer-codex
Original: 1,644 params, 99.04% accuracy, trained with SGD.
Architecture: 1-layer GPT, d=8, h=2, ff=12, paired digit-column tokens.
"""

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocab: 10 digit tokens + 100 pair tokens (P00-P99) + =, <bos>, <eos>, <pad> = 114
VOCAB_SIZE: int = 114
MAX_OUTPUT_LEN: int = 12  # 11 reversed digits + EOS

NUM_DIGITS = 10
SUM_DIGITS = 11
PAIR_BASE = 10  # pair tokens start at index 10
EQUALS_ID = 110
BOS_ID = 111
EOS_ID = 112
PAD_ID = 113

MAX_SEQ_LEN = 23  # BOS + 10 pair tokens + = + 11 digits (model input during generation)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:seqlen, :seqlen], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PairedTokenTransformer(nn.Module):
    def __init__(self, vocab_size=114, d_model=8, n_head=2, n_layer=1,
                 d_ff=12, max_seq_len=23):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, d_ff, max_seq_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        if T > self.max_seq_len:
            idx = idx[:, -self.max_seq_len:]
            T = self.max_seq_len
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def build_model() -> tuple[nn.Module, dict]:
    model = PairedTokenTransformer(
        vocab_size=VOCAB_SIZE, d_model=8, n_head=2, n_layer=1,
        d_ff=12, max_seq_len=MAX_SEQ_LEN,
    )
    ckpt_path = Path(__file__).parent / "checkpoints" / "anadim_1644.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    metadata = {
        "name": "anadim-1644",
        "author": "anadim",
        "architecture": "1-layer GPT, d=8, h=2, ff=12, paired digit-column tokens",
        "tricks": "weight tying, paired tokens (100 column tokens), reversed output",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    """BOS + 10 paired digit-column tokens (LSD first) + =."""
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    # Reversed: LSD first
    a_digits = [int(c) for c in a_str[::-1]]
    b_digits = [int(c) for c in b_str[::-1]]
    tokens = [BOS_ID]
    for da, db in zip(a_digits, b_digits):
        tokens.append(PAIR_BASE + da * 10 + db)
    tokens.append(EQUALS_ID)
    return tokens


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
