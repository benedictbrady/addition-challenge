"""Baseline: cosminscn 66-param hand-coded nanoGPT adder.

Adapted from: https://gist.github.com/cosminscn/e4d028281378e16b18e61fca1163f9cb
Original: 66 params, 100% accuracy, hand-coded weights.
Architecture: 1-layer nanoGPT, d=4, 2 heads.
              Rotation Q (2 angles), sparse c_proj (2 nonzero), parabolic lm_head,
              factorized embed, sinusoidal PE (period 11).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 10
MAX_OUTPUT_LEN: int = 11

NUM_DIGITS = 10
OUTPUT_DIGITS = 11
N_EMBD = 4
N_HEAD = 2
HEAD_DIM = N_EMBD // N_HEAD
MLP_HIDDEN = 4
BLOCK_SIZE = 35


class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, rank=1):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(vocab_size, rank))
        self.B = nn.Parameter(torch.zeros(rank, emb_dim))

    def forward(self, x):
        return self.A[x] @ self.B


class Rank1Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(out_features, 1))
        self.v = nn.Parameter(torch.zeros(1, in_features))

    def forward(self, x):
        return (x @ self.v.T) @ self.u.T


class SlimLinear(nn.Module):
    """Linear reading only selected input dims."""
    def __init__(self, out_features, input_dims):
        super().__init__()
        self.input_dims = input_dims
        self.weight = nn.Parameter(torch.zeros(out_features, len(input_dims)))

    def forward(self, x):
        return x[..., self.input_dims] @ self.weight.T


class RotationQ(nn.Module):
    """Q projection via 2 rotation angles + 1 scale."""
    def __init__(self):
        super().__init__()
        self.angle_h0 = nn.Parameter(torch.zeros(1))
        self.angle_h1 = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        S, a0, a1 = self.scale, self.angle_h0, self.angle_h1
        W = x.new_zeros(4, 4)
        W[0, 1] = -torch.cos(a0) * S; W[0, 2] = torch.sin(a0) * S
        W[1, 1] = torch.sin(a0) * S; W[1, 2] = torch.cos(a0) * S
        W[2, 1] = -torch.cos(a1) * S; W[2, 2] = torch.sin(a1) * S
        W[3, 1] = torch.sin(a1) * S; W[3, 2] = torch.cos(a1) * S
        return x @ W.T


class SparseProj(nn.Module):
    """Projection with exactly 2 non-zero entries at fixed positions."""
    def __init__(self):
        super().__init__()
        self.val0 = nn.Parameter(torch.zeros(1))
        self.val1 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = torch.zeros_like(x)
        out[..., 3] = self.val0 * x[..., 0]
        out[..., 1] = self.val1 * x[..., 2]
        return out


class ParabolicHead(nn.Module):
    """LM head: logit[v] = lin * v * (x[...,3] * dim_w) + quad * v**2."""
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.lin = nn.Parameter(torch.zeros(1))
        self.quad = nn.Parameter(torch.zeros(1))
        self.dim_w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        scalar = x[..., 3:4] * self.dim_w
        d = torch.arange(self.vocab_size, device=x.device, dtype=x.dtype).view(1, 1, -1)
        return self.lin * d * scalar + self.quad * d ** 2


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = N_HEAD
        self.head_dim = HEAD_DIM
        self.q_proj = RotationQ()
        self.k_proj = SlimLinear(N_EMBD, [1, 2])
        self.v_proj = Rank1Linear(N_EMBD, N_EMBD)
        self.c_proj = SparseProj()

    def forward(self, x):
        B, T, C = x.size()
        hd = self.head_dim
        q = self.q_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, MLP_HIDDEN, bias=True)
        self.c_proj = Rank1Linear(MLP_HIDDEN, N_EMBD)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CausalSelfAttention()
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class CosminAdder66(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = FactorizedEmbedding(VOCAB_SIZE, N_EMBD)
        self.block = Block()
        self.lm_head = ParabolicHead(VOCAB_SIZE)

    def generate_pe(self, seq_len, device):
        pe = torch.zeros(seq_len, N_EMBD, device=device)
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        th = 2 * math.pi / 11
        amp = torch.where(pos <= 21, 100.0, 1.0)
        pe[:, 1] = amp * torch.sin(pos * th)
        pe[:, 2] = amp * torch.cos(pos * th)
        return pe

    def forward(self, idx):
        seq_len = idx.size(1)
        x = self.wte(idx) + self.generate_pe(seq_len, idx.device)
        x = self.block(x)
        return self.lm_head(x)


def _set_weights(model: CosminAdder66) -> None:
    th = 2 * math.pi / 11
    S = 100.0

    with torch.no_grad():
        for v in range(10):
            model.wte.A[v, 0] = float(v)
        model.wte.B[0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])

        attn = model.block.attn
        attn.q_proj.angle_h0.fill_(8 * th)
        attn.q_proj.angle_h1.fill_(9 * th)
        attn.q_proj.scale.fill_(S)

        attn.k_proj.weight.copy_(torch.tensor([
            [1.0, 0.0], [0.0, 1.0],
            [1.0, 0.0], [0.0, 1.0],
        ]))

        attn.v_proj.u.copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]]))
        attn.v_proj.v.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

        attn.c_proj.val0.fill_(2.0)
        attn.c_proj.val1.fill_(2.0)

        mlp = model.block.mlp
        fw = torch.zeros(4, 4)
        fb = torch.zeros(4)
        fw[0, 1] = 100; fw[0, 0] = -100; fb[0] = -50
        fw[1, 1] = 100; fw[1, 0] = -100; fb[1] = -150
        fw[2, 3] = 1000; fw[2, 1] = 10; fw[2, 0] = -10; fb[2] = -9045
        fw[3, 3] = 1000; fw[3, 1] = 10; fw[3, 0] = -10; fb[3] = -9055
        mlp.c_fc.weight.copy_(fw)
        mlp.c_fc.bias.copy_(fb)

        mlp.c_proj.u.zero_()
        mlp.c_proj.v.zero_()
        mlp.c_proj.u[3, 0] = 1.0
        mlp.c_proj.v[0, :] = torch.tensor([0.01, -0.01, -1.0, 1.0])

        model.lm_head.lin.fill_(2.0)
        model.lm_head.quad.fill_(-1.0)
        model.lm_head.dim_w.fill_(1.0)


def build_model() -> tuple[nn.Module, dict]:
    model = CosminAdder66()
    _set_weights(model)
    metadata = {
        "name": "cosminscn-66",
        "author": "cosminscn",
        "architecture": "1-layer nanoGPT, d=4, 2h, hd=2",
        "tricks": "hand-coded weights, rotation Q, sparse c_proj, parabolic lm_head, factorized embed, sinusoidal PE (period 11)",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    s = f"{a:010d}+{b:010d}="
    return [int(c) if c.isdigit() else 0 for c in s]


def decode(tokens: list[int]) -> int:
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
    while len(digits) < 11:
        digits.append(0)
    return int("".join(str(d) for d in reversed(digits)))
