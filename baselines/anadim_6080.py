"""Baseline: anadim 6,080-param trained transformer.

Adapted from: https://github.com/anadim/smallest-addition-transformer-claude-code
Original: 6,080 params, 100% accuracy, trained with SGD.
Architecture: 2-layer GPT, d=16, h=2, ff=48, weight tying.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocab: 0-9, +, =, <pad>, <bos>, <eos> = 15 tokens
VOCAB_SIZE: int = 15
MAX_OUTPUT_LEN: int = 12  # 11 reversed digits + EOS

BOS_ID = 13
EOS_ID = 14
NUM_DIGITS = 10
OUT_DIGITS = 11
FIXED_SEQ_LEN = 35  # BOS + 10 + 1(+) + 10 + 1(=) + 11 + EOS


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=False),
            nn.GELU(),
            nn.Linear(ff_dim, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class AnadimTransformer(nn.Module):
    def __init__(self, vocab_size=15, d_model=16, n_heads=2, n_layers=2,
                 ff_dim=48, max_seq_len=35):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        if T > self.max_seq_len:
            idx = idx[:, -self.max_seq_len:]
            T = self.max_seq_len
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def build_model() -> tuple[nn.Module, dict]:
    model = AnadimTransformer(
        vocab_size=VOCAB_SIZE, d_model=16, n_heads=2, n_layers=2,
        ff_dim=48, max_seq_len=FIXED_SEQ_LEN,
    )
    ckpt_path = Path(__file__).parent / "checkpoints" / "anadim_6080.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    metadata = {
        "name": "anadim-6080",
        "author": "anadim",
        "architecture": "2-layer GPT, d=16, h=2, ff=48, weight tying",
        "tricks": "weight tying, reversed output, zero-padded input",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    """BOS + zero-padded digits + '+' + zero-padded digits + '='."""
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    text = a_str + "+" + b_str + "="
    token_map = {str(i): i for i in range(10)}
    token_map["+"] = 10
    token_map["="] = 11
    return [BOS_ID] + [token_map[ch] for ch in text]


def decode(tokens: list[int]) -> int:
    """Decode reversed digit tokens, stop at EOS or non-digit."""
    digits = []
    for t in tokens:
        if t == EOS_ID:
            break
        if 0 <= t <= 9:
            digits.append(t)
        else:
            break
    if not digits:
        return 0
    while len(digits) < OUT_DIGITS:
        digits.append(0)
    digits = digits[:OUT_DIGITS]
    return int("".join(str(d) for d in reversed(digits)))
