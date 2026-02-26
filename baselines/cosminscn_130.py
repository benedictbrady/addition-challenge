"""Baseline: cosminscn 130-param hand-coded GPT adder.

Adapted from: https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b
Original: 130 params, 100% accuracy, hand-coded weights.
Architecture: 1-layer GPT, n_embd=4, 2 heads, rank-1 projections, factorized embedding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 10  # digits 0-9 only
MAX_OUTPUT_LEN: int = 11  # 11 reversed digits (no EOS token in this scheme)

NUM_DIGITS = 10


# ---- Architecture components ----

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, rank=1):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(vocab_size, rank))
        self.B = nn.Parameter(torch.zeros(rank, emb_dim))

    def forward(self, x):
        return self.A[x] @ self.B


class Rank1Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(out_features, 1))
        self.v = nn.Parameter(torch.zeros(1, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = (x @ self.v.T) @ self.u.T
        if self.bias is not None:
            out += self.bias
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, mlp_hidden, bias=True)
        self.gelu = nn.ReLU()  # replaced at init time
        self.c_proj = Rank1Linear(mlp_hidden, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, mlp_hidden):
        super().__init__()
        self.ln_1 = nn.Identity()
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.Identity()
        self.mlp = MLP(n_embd, mlp_hidden)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class CosminAdder(nn.Module):
    """130-param GPT with hand-coded weights for addition."""

    def __init__(self):
        super().__init__()
        n_embd = 4
        n_head = 2
        mlp_hidden = 4
        block_size = 35

        self.wte = FactorizedEmbedding(10, n_embd, rank=1)
        self.block = Block(n_embd, n_head, mlp_hidden)
        self.lm_head = Rank1Linear(n_embd, 10, bias=True)
        self.block_size = block_size
        self.n_embd = n_embd

    def generate_pe(self, seq_len, device):
        pe = torch.zeros(seq_len, self.n_embd, device=device)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        th = 2 * math.pi / 11
        amp = torch.where(positions <= 21, 100.0, 1.0)
        pe[:, 1] = amp * torch.sin(positions * th)
        pe[:, 2] = amp * torch.cos(positions * th)
        return pe

    def forward(self, idx):
        seq_len = idx.size(1)
        x = self.wte(idx) + self.generate_pe(seq_len, idx.device)
        x = self.block(x)
        logits = self.lm_head(x)
        return logits


def _build_adder():
    """Build the model and hand-set all weights."""
    model = CosminAdder()
    th = 2 * math.pi / 11
    S = 100.0
    n = 4

    with torch.no_grad():
        # Token embeddings
        model.wte.A.zero_()
        model.wte.B.zero_()
        for v in range(10):
            model.wte.A[v, 0] = float(v)
        model.wte.B[0, 0] = 1.0

        # Attention weights (c_attn)
        w = torch.zeros(3 * n, n)
        w[0, 1] = -math.cos(8 * th) * S
        w[0, 2] = math.sin(8 * th) * S
        w[1, 1] = math.sin(8 * th) * S
        w[1, 2] = math.cos(8 * th) * S
        w[2, 1] = -math.cos(9 * th) * S
        w[2, 2] = math.sin(9 * th) * S
        w[3, 1] = math.sin(9 * th) * S
        w[3, 2] = math.cos(9 * th) * S
        w[4, 1] = w[6, 1] = 1.0
        w[5, 2] = w[7, 2] = 1.0
        w[8, 0] = w[10, 0] = 1.0
        model.block.attn.c_attn.weight.copy_(w)

        # Projection weights (c_proj)
        w = torch.zeros(n, n)
        w[3, 0] = 2.0
        w[1, 2] = 2.0
        model.block.attn.c_proj.weight.copy_(w)

        # MLP fully-connected layer
        fw, fb = torch.zeros(4, n), torch.zeros(4)
        fw[0, 1] = 100
        fw[0, 0] = -100
        fb[0] = -50
        fw[1, 1] = 100
        fw[1, 0] = -100
        fb[1] = -150
        fw[2, 3] = 1000
        fw[2, 1] = 10
        fw[2, 0] = -10
        fb[2] = -9045
        fw[3, 3] = 1000
        fw[3, 1] = 10
        fw[3, 0] = -10
        fb[3] = -9055
        model.block.mlp.c_fc.weight.copy_(fw)
        model.block.mlp.c_fc.bias.copy_(fb)

        # MLP projection (Rank-1)
        model.block.mlp.c_proj.u.zero_()
        model.block.mlp.c_proj.v.zero_()
        model.block.mlp.c_proj.u[3, 0] = 1.0
        model.block.mlp.c_proj.v[0, :] = torch.tensor([0.01, -0.01, -1.0, 1.0])

        # Output head (Rank-1)
        model.lm_head.u.zero_()
        model.lm_head.v.zero_()
        model.lm_head.bias.zero_()
        for v in range(10):
            model.lm_head.u[v, 0] = 2.0 * v
            model.lm_head.bias[v] = -float(v * v)
        model.lm_head.v[0, 3] = 1.0

    return model


def build_model() -> tuple[nn.Module, dict]:
    model = _build_adder()
    metadata = {
        "name": "cosminscn-130",
        "author": "cosminscn",
        "architecture": "1-layer GPT, n_embd=4, 2 heads, rank-1 projections, factorized embedding",
        "tricks": "hand-coded weights, sinusoidal PE (not counted), rank-1 linear, factorized embedding",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    """Encode as "0001234567+0009876543=" with + and = mapped to 0."""
    s = f"{a:010d}+{b:010d}="
    # + and = are mapped to token 0 (same as digit '0')
    return [int(c) if c.isdigit() else 0 for c in s]


def decode(tokens: list[int]) -> int:
    """Decode 11 reversed digits back to integer."""
    digits = []
    for t in tokens:
        if 0 <= t <= 9:
            digits.append(t)
        else:
            break
        if len(digits) == 11:
            break
    if not digits:
        return 0
    # Pad to 11 digits with zeros
    while len(digits) < 11:
        digits.append(0)
    # Reverse (output is LSD-first)
    return int("".join(str(d) for d in reversed(digits)))
