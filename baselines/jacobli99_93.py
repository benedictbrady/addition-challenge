"""Baseline: jacobli99 (SeuperHakkerJa) 93-param hand-coded decoder adder.

Adapted from: https://gist.github.com/SeuperHakkerJa/9d615964d2284a9a699b5a24cf19e69d
Original: 93 params, 100% accuracy, hand-coded weights.
Architecture: 1-layer decoder, d=2, 5 heads (MQA), head_dim=2, ff=4.
              Tied parabolic decode, RoPE digit routing, ReLU carry detection.

All parameters are frozen (requires_grad=False). Our param counter counts
all nn.Parameter tensors regardless of requires_grad.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 10
MAX_OUTPUT_LEN: int = 11

NUM_DIGITS = 10
OUTPUT_DIGITS = 11

HIDDEN_SIZE = 2
NUM_HEADS = 5
HEAD_DIM = 2
MLP_SIZE = 4

EMBED_CONST = 1000.0
DECODE_EPS = 5e-4
QK_SCALE = 256.0
CAUSAL_MASK_NEG = -1e4
ROPE_OFFSETS = (0.0, 23.0, 11.0, 22.0, 10.0)

MAX_SEQ_LEN = 36


def rope_2d(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    c = torch.cos(pos).view(1, 1, -1, 1)
    s = torch.sin(pos).view(1, 1, -1, 1)
    x0, x1 = x[..., 0:1], x[..., 1:2]
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)


class JacobliAdder(nn.Module):
    def __init__(self):
        super().__init__()
        C = EMBED_CONST
        eps = DECODE_EPS
        qk = QK_SCALE
        quad = eps / 2.0

        # Embedding (tied with output)
        emb = torch.tensor([
            [C - quad * (d * d), float(d)] for d in range(10)
        ])
        self.embed_tokens = nn.Parameter(emb, requires_grad=False)

        # Output scaling for tied decode
        self.out_scale = nn.Parameter(
            torch.tensor([1.0 / C, eps]), requires_grad=False
        )

        # K projection
        self.k_proj = nn.Parameter(
            torch.tensor([[qk, 0.0], [0.0, 0.0]]), requires_grad=False
        )

        # V projection
        self.v_proj = nn.Parameter(
            torch.tensor([[0.0, 1.0], [0.0, 0.0]]), requires_grad=False
        )

        # Q projection (5 heads * 2 = 10 rows)
        Q = torch.zeros(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)
        for h, off in enumerate(ROPE_OFFSETS):
            c, s = math.cos(off), -math.sin(off)
            Q[2 * h, 0] = c * qk
            Q[2 * h + 1, 0] = s * qk
        self.q_proj = nn.Parameter(Q, requires_grad=False)

        # O projection
        O = torch.zeros(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM)
        O[0, 0] = +1.0; O[0, 2] = -1.0; O[0, 4] = -1.0
        O[1, 0] = -1.0; O[1, 6] = +1.0; O[1, 8] = +1.0
        self.o_proj = nn.Parameter(O, requires_grad=False)

        # Causal mask
        mask = torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        # MLP
        w1 = torch.tensor([
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 20.0],
            [-2.0, 20.0],
        ])
        self.w1 = nn.Parameter(w1, requires_grad=False)

        b1 = torch.tensor([C - 8.0, C - 9.0, 2 * C - 188.0, 2 * C - 189.0])
        self.b1 = nn.Parameter(b1, requires_grad=False)

        w2 = torch.zeros(HIDDEN_SIZE, MLP_SIZE)
        w2[1, 0] = +1.0; w2[1, 1] = -1.0
        w2[1, 2] = -10.0; w2[1, 3] = +10.0
        self.w2 = nn.Parameter(w2, requires_grad=False)

        self.b2 = nn.Parameter(torch.zeros(HIDDEN_SIZE), requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = F.embedding(input_ids, self.embed_tokens)
        B, T, _ = x.shape

        pos = torch.arange(T, device=x.device, dtype=x.dtype)

        q = (x @ self.q_proj.t()).view(B, T, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k = (x @ self.k_proj.t()).view(B, T, 1, HEAD_DIM).permute(0, 2, 1, 3)
        k = k.expand(-1, NUM_HEADS, -1, -1)
        v = (x @ self.v_proj.t()).view(B, T, 1, HEAD_DIM).permute(0, 2, 1, 3)
        v = v.expand(-1, NUM_HEADS, -1, -1)

        q = rope_2d(q, pos)
        k = rope_2d(k, pos)

        scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(HEAD_DIM)
        scores = scores.masked_fill(self.causal_mask[:T, :T].view(1, 1, T, T), float("-inf"))
        w = F.softmax(scores, dim=-1)
        att = torch.einsum("bhts,bhsd->bhtd", w, v)
        att = att.permute(0, 2, 1, 3).contiguous().view(B, T, NUM_HEADS * HEAD_DIM)

        x = x + (att @ self.o_proj.t())

        h = F.relu(x @ self.w1.t() + self.b1)
        x = x + (h @ self.w2.t() + self.b2)

        y = x * self.out_scale
        return y @ self.embed_tokens.t()


def build_model() -> tuple[nn.Module, dict]:
    model = JacobliAdder()
    model.eval()
    metadata = {
        "name": "jacobli99-93",
        "author": "jacobli99 (SeuperHakkerJa)",
        "architecture": "1-layer decoder, d=2, 5h (MQA), hd=2, ff=4",
        "tricks": "hand-coded weights, tied parabolic decode, RoPE digit routing, ReLU carry detection",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    a_digits = [int(c) for c in a_str[::-1]]
    b_digits = [int(c) for c in b_str[::-1]]
    return [0] + a_digits + [0, 0] + b_digits + [0]


def decode(tokens: list[int]) -> int:
    digits = []
    for t in tokens:
        if 0 <= t <= 9:
            digits.append(t)
        else:
            break
        if len(digits) == OUTPUT_DIGITS:
            break
    if not digits:
        return 0
    while len(digits) < OUTPUT_DIGITS:
        digits.append(0)
    return int("".join(str(d) for d in reversed(digits)))
