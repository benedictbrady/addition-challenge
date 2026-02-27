"""Baseline: alexlitz 36-param hand-coded 2-layer ALiBi decoder adder.

Adapted from: https://gist.github.com/alexlitz/0d5efbccf443fb0e8136b8f5bd85140a
Original: 36 params, 100% accuracy, hand-coded weights.
Architecture: 2-layer decoder, d=5, 5h+1h.
              ALiBi slope=log(10) for base-10 weighting, sparse embed,
              gated ReLU FFN, float64.

Note: This model requires float64 precision for correct operation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 14  # 0-9, =, <bos>, <eos>, +
MAX_OUTPUT_LEN: int = 11

NUM_DIGITS = 10
OUTPUT_DIGITS = 11
EMBEDDING_DIM = 5
LAYER0_HEADS = 5
LAYER1_D_MODEL = 16

# Token constants
TOK_EQUAL = 10
TOK_BOS = 11
TOK_EOS = 12
TOK_PLUS = 13

# Embedding constants
DIGIT_EMBED_SCALE = 10
V_SCALE = 1e4
DIGIT_SCALE = 1e10
FINAL_SCALE = 100
DIGIT_OFFSET = 0.5
GATE_BIAS_SHIFT = 15.0
ALIBI_CONSTANT = math.log(10)

# Dimension indices
EQ_DIM, SPECIAL_DIM, DIGIT_DIM, COUNT_DIM, SCALE_DIM = 0, 1, 2, 3, 4
ADJUSTMENT_HEAD = 3
SCALE_HEAD = 4
CANDIDATES_START = 5
DIGIT_POS_DIM = 15

K_DIGIT_SCORE = -1000.0
K_SPECIAL_SCORE = -40.0
V_PROJ_SPECIAL = 0.1
V_PROJ_NEG_DOUBLE = -1.1
V_PROJ_SCALE = math.exp(K_SPECIAL_SCORE - math.log(10))


def softmax1(x, dim=-1):
    """Softmax with denominator = 1 + sum(exp(x))."""
    exp_x = x.exp()
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


def apply_alibi(seq_len, n_heads):
    pos = torch.arange(seq_len, dtype=torch.float64)
    rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
    slopes = torch.zeros(n_heads, dtype=torch.float64)
    slopes[ADJUSTMENT_HEAD] = ALIBI_CONSTANT
    return slopes.unsqueeze(1).unsqueeze(2) * rel_pos.unsqueeze(0)


def pad_to(x, d):
    if x.size(-1) >= d:
        return x[..., :d]
    return torch.cat([x, torch.zeros(*x.shape[:-1], d - x.size(-1), dtype=x.dtype)], dim=-1)


class AlexlitzAttentionL0(nn.Module):
    """Layer 0: 5-head ALiBi attention with sparse projections."""

    def __init__(self):
        super().__init__()
        # K: weight + bias (2 params)
        self.k_weight = nn.Parameter(torch.tensor(
            K_SPECIAL_SCORE - K_DIGIT_SCORE, dtype=torch.float64
        ))
        self.k_bias = nn.Parameter(torch.tensor(K_DIGIT_SCORE, dtype=torch.float64))
        # V: 3 weights (3 params)
        self.v_w1 = nn.Parameter(torch.tensor(V_PROJ_SPECIAL / V_PROJ_SCALE, dtype=torch.float64))
        self.v_w2 = nn.Parameter(torch.tensor(V_PROJ_NEG_DOUBLE / V_PROJ_SCALE, dtype=torch.float64))
        self.v_w3 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        # Q: bias=1 broadcast (1 param)
        self.q_bias = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, h):
        B, T, _ = h.shape
        q = self.q_bias.expand(B, T, LAYER0_HEADS).view(B, T, LAYER0_HEADS, 1).transpose(1, 2)

        k = torch.zeros(B, T, LAYER0_HEADS, dtype=torch.float64, device=h.device)
        k[..., ADJUSTMENT_HEAD] = h[..., SPECIAL_DIM] * self.k_weight + self.k_bias
        k = k.view(B, T, LAYER0_HEADS, 1).transpose(1, 2)

        v = torch.zeros(B, T, LAYER0_HEADS, dtype=torch.float64, device=h.device)
        v[..., ADJUSTMENT_HEAD] = h[..., SPECIAL_DIM] * self.v_w1 + h[..., EQ_DIM] * self.v_w2
        v[..., SCALE_HEAD] = h[..., EQ_DIM] * self.v_w3
        v = v.view(B, T, LAYER0_HEADS, 1).transpose(1, 2)

        alibi = apply_alibi(T, LAYER0_HEADS).unsqueeze(0).to(h.device)
        scores = torch.matmul(q, k.transpose(-2, -1)) + alibi
        scores = scores.masked_fill(
            torch.triu(torch.ones(T, T, device=h.device), 1).bool(), float('-inf')
        )
        attn = softmax1(scores, dim=-1).double()
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return out


class AlexlitzAttentionL1(nn.Module):
    """Layer 1: 1-head uniform causal attention."""

    def __init__(self):
        super().__init__()
        # V: weight + bias (2 params)
        self.v_weight = nn.Parameter(torch.tensor(FINAL_SCALE, dtype=torch.float64))
        self.v_bias = nn.Parameter(torch.tensor(GATE_BIAS_SHIFT, dtype=torch.float64))

    def forward(self, h):
        B, T, D = h.shape
        v = (h[..., DIGIT_POS_DIM] * self.v_weight + self.v_bias).unsqueeze(-1)
        v = v.view(B, T, 1, 1).transpose(1, 2)

        q = torch.zeros(B, T, 1, 1, dtype=torch.float64, device=h.device).transpose(1, 2)
        k = torch.zeros(B, T, 1, 1, dtype=torch.float64, device=h.device).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores.masked_fill(
            torch.triu(torch.ones(T, T, device=h.device), 1).bool(), float('-inf')
        )
        attn = softmax1(scores, dim=-1).double()
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return out


class AlexlitzAdder(nn.Module):
    """36-parameter transformer for 10-digit addition (float64).

    Embedding is factorized: only 13 non-zero values stored as params.
    Full 14x5 embedding table is reconstructed in forward().
    """

    def __init__(self):
        super().__init__()
        # Embedding (13 params): 9 digit values in dim 2, 4 flag values
        # Digits 1-9: [0, 0, digit_vals[i], 0, 0]
        self.digit_vals = nn.Parameter(torch.tensor(
            [float(i * DIGIT_EMBED_SCALE) for i in range(1, 10)], dtype=torch.float64
        ))
        # Flag values: eq_dim0, eq_special, bos_special, plus_special (all 1.0)
        self.flag_vals = nn.Parameter(torch.tensor(
            [1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        ))

        # Layer 0 attention (6 params)
        self.attn0 = AlexlitzAttentionL0()

        # Layer 0 FFN (12 params): gate=1 broadcast + up=11 values
        pv = [(i + DIGIT_OFFSET) * DIGIT_SCALE * FINAL_SCALE for i in range(NUM_DIGITS)]
        self.ffn0_up = nn.Parameter(torch.tensor(pv + [DIGIT_SCALE], dtype=torch.float64))
        self.ffn0_gate_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        # Layer 1 attention (2 params)
        self.attn1 = AlexlitzAttentionL1()

        # Layer 1 FFN (3 params): +V_SCALE, -V_SCALE, FINAL_SCALE broadcast
        self.ffn1_v_scale_pos = nn.Parameter(torch.tensor(V_SCALE, dtype=torch.float64))
        self.ffn1_v_scale_neg = nn.Parameter(torch.tensor(-V_SCALE, dtype=torch.float64))
        self.ffn1_up_scale = nn.Parameter(torch.tensor(FINAL_SCALE, dtype=torch.float64))

    def _build_embedding(self) -> torch.Tensor:
        """Reconstruct full 14x5 embedding table from sparse params."""
        emb = torch.zeros(14, 5, dtype=torch.float64, device=self.digit_vals.device)
        emb[1:10, DIGIT_DIM] = self.digit_vals
        emb[TOK_EQUAL, EQ_DIM] = self.flag_vals[0]
        emb[TOK_EQUAL, SPECIAL_DIM] = self.flag_vals[1]
        emb[TOK_BOS, SPECIAL_DIM] = self.flag_vals[2]
        emb[TOK_PLUS, SPECIAL_DIM] = self.flag_vals[3]
        return emb

    @torch.inference_mode()
    def forward(self, x):
        B, T = x.shape
        h = F.embedding(x, self._build_embedding())

        # === Layer 0 ===
        h = pad_to(h, EMBEDDING_DIM)
        attn_out = self.attn0(h)
        h = h + attn_out

        # FFN0: gated ReLU
        gate_in = torch.zeros(B, T, 11, dtype=torch.float64, device=x.device)
        gate_in[..., :NUM_DIGITS] = h[..., SCALE_DIM:SCALE_DIM + 1] * self.ffn0_gate_scale
        gate_in[..., NUM_DIGITS] = h[..., DIGIT_DIM]
        gate_out = F.relu(gate_in)
        up_out = h[..., COUNT_DIM:COUNT_DIM + 1] * self.ffn0_up
        ffn_hidden = gate_out * up_out

        h = pad_to(h, LAYER1_D_MODEL)
        h[..., 5:16] = h[..., 5:16] + ffn_hidden

        # === Layer 1 ===
        attn1_out = self.attn1(h)
        h = h + attn1_out

        # FFN1: V-shape via relu(+x) + relu(-x)
        candidates = h[..., CANDIDATES_START:CANDIDATES_START + NUM_DIGITS]
        gate_pos = F.relu(candidates * self.ffn1_v_scale_pos)
        gate_neg = F.relu(candidates * self.ffn1_v_scale_neg)
        ffn_out = (gate_pos + gate_neg) * self.ffn1_up_scale

        h = pad_to(h, NUM_DIGITS)
        h = h + ffn_out

        # Return negated logits (model uses argmin, harness uses argmax)
        return -h


def build_model() -> tuple[nn.Module, dict]:
    model = AlexlitzAdder()
    model.eval()
    metadata = {
        "name": "alexlitz-36",
        "author": "alexlitz",
        "architecture": "2-layer decoder, d=5, 5h+1h, ALiBi, gated ReLU FFN, float64",
        "tricks": "hand-coded weights, ALiBi slope=log(10), softmax1, identity/broadcast (0 params), sparse embed",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    """Encode as <bos> + digits_a + + + digits_b + =."""
    s = f"{a:010d}+{b:010d}="
    token_map = {str(i): i for i in range(10)}
    token_map["+"] = TOK_PLUS
    token_map["="] = TOK_EQUAL
    return [TOK_BOS] + [token_map[c] for c in s]


def decode(tokens: list[int]) -> int:
    """Decode digit tokens (MSB-first, NOT reversed)."""
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
    # MSB-first: read digits left to right
    return int("".join(str(d) for d in digits))
