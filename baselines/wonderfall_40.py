"""Baseline: Wonderfall 40-param hand-coded 1-layer decoder adder.

Adapted from: https://gist.github.com/Wonderfall/373460ba8cec6cd143c8b0e9ebcd1294
Original: 40 params, 100% accuracy, hand-coded weights using MLX.
Architecture: 1-layer decoder, d=2, 1h, head_dim=2,
              tied Q/K + V/O projections, RoPE period-19,
              parabolic tied-embed decode, two-hinge ReLU MLP,
              parameterless RMSNorm.

Reimplemented in pure PyTorch (original used Apple MLX).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 10  # digits 0-9 only
MAX_OUTPUT_LEN: int = 11  # 11 reversed digits (no EOS)

# Model constants (from original)
MODEL_DIM = 2
HEAD_DIM = 2
NUM_DIGITS = 10
OUTPUT_DIGITS = 11

EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)  # sqrt(2)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
DECODE_QUAD = 1e-3
DECODE_CURVATURE = 0.1
ROPE_PERIOD = 19.0
ROPE_FACTOR = ROPE_PERIOD / (2.0 * math.pi)
ROPE_SCALE = 1.0 / ROPE_FACTOR
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)
TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
CARRY_ALPHA = 256.0 / CONST_NORM

MAX_SEQ_LEN = 65  # 31 input + 11 output + margin


# ---- Building blocks ----

def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply RoPE with period-19 to tensor of shape (..., seq_len, 2).

    With custom theta = ROPE_FACTOR, angle = position * ROPE_SCALE.
    GPT-NeoX style: split at midpoint.
    """
    seq_len = x.shape[-2]
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    angles = positions * ROPE_SCALE
    cos_a = torch.cos(angles).unsqueeze(-1)  # (T, 1)
    sin_a = torch.sin(angles).unsqueeze(-1)  # (T, 1)
    x0 = x[..., :1]
    x1 = x[..., 1:]
    return torch.cat([x0 * cos_a - x1 * sin_a, x0 * sin_a + x1 * cos_a], dim=-1)


class Wonderfall40Attention(nn.Module):
    """Single-head attention with tied Q/K and V/O projections, RoPE period-19."""

    def __init__(self):
        super().__init__()
        # Tied Q/K projection (4 params) + Q phase offset (1 param)
        self.w_qk = nn.Linear(MODEL_DIM, HEAD_DIM, bias=False)
        self.q_phase = nn.Parameter(torch.zeros(1))
        # Tied V/O projection (4 params) + V scale (1 param)
        self.w_vo = nn.Linear(MODEL_DIM, HEAD_DIM, bias=False)
        self.v_scale = nn.Parameter(torch.zeros(1))

        mask = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Q and K share the same projection
        k = self.w_qk(x)  # (B, T, 2)

        # Q gets an additional phase rotation
        phase = self.q_phase
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        q0 = k[..., :1]
        q1 = k[..., 1:]
        q = torch.cat([q0 * cos_p - q1 * sin_p, q0 * sin_p + q1 * cos_p], dim=-1)

        # Apply RoPE to both Q and K
        q = _apply_rope(q)
        k = _apply_rope(k)

        # V projection with scale
        v = self.w_vo(x) * self.v_scale  # (B, T, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(HEAD_DIM)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v  # (B, T, 2)

        # O projection shares weights with V (transposed)
        out = F.linear(out, self.w_vo.weight.t())  # (B, T, 2)
        return out


class Wonderfall40MLP(nn.Module):
    """Two-hinge ReLU MLP: relu(w1 @ x) then w2 @ h."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.w2 = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


class RMSNorm(nn.Module):
    """RMSNorm with learnable weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


def _parameterless_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype)


class Wonderfall40Adder(nn.Module):
    """40-param 1-layer decoder reimplemented in PyTorch.

    Parameters:
      - embed_tokens: 10 x 2 = 20 params
      - attn.w_qk: 2 x 2 = 4 params
      - attn.q_phase: 1 param
      - attn.w_vo: 2 x 2 = 4 params
      - attn.v_scale: 1 param
      - mlp.w1: 2 x 2 = 4 params
      - mlp.w2: 2 x 2 = 4 params
      - final_norm: 2 params
      Total: 40 params
    """

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.attn = Wonderfall40Attention()
        self.mlp = Wonderfall40MLP()
        self.final_norm = RMSNorm(MODEL_DIM)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(idx)          # (B, T, 2)
        # Pre-attention parameterless RMSNorm
        h = _parameterless_rmsnorm(x)
        x = x + self.attn(h)               # (B, T, 2)
        # Post-attention parameterless RMSNorm
        h = _parameterless_rmsnorm(x)
        x = x + self.mlp(h)                # (B, T, 2)
        x = self.final_norm(x)             # (B, T, 2)
        # Tied embedding decode
        logits = x @ self.embed_tokens.weight.t()  # (B, T, 10)
        return logits


# ---- Hand-coded weight initialization ----

def _set_weights(model: Wonderfall40Adder) -> None:
    """Hand-set all 40 weights to implement addition."""
    with torch.no_grad():
        # Embedding: digit d -> [EMBED_CONST - DECODE_QUAD * d^2, -d]
        for d in range(10):
            model.embed_tokens.weight[d] = torch.tensor([
                EMBED_CONST - DECODE_QUAD * (d * d),
                -float(d),
            ])

        # Final norm
        model.final_norm.weight.copy_(torch.tensor([
            (DECODE_CURVATURE / DECODE_QUAD) / CONST_NORM,
            -(DIGIT_SCALE / 50.0),
        ]))

        # Attention: tied Q/K projection
        model.attn.w_qk.weight.copy_(torch.tensor([
            [QK_SCALE, 0.0],
            [0.0, 0.0],
        ]))

        # Q phase offset
        model.attn.q_phase.fill_(-PHI)

        # Attention: tied V/O projection
        model.attn.w_vo.weight.copy_(torch.tensor([
            [0.0, 1.0],
            [0.0, 0.0],
        ]))

        # V scale
        model.attn.v_scale.fill_(-22.0 * DIGIT_SCALE)

        # MLP w1: carry detection
        model.mlp.w1.weight.copy_(torch.tensor([
            [CARRY_ALPHA * (-94.0) / CONST_NORM, CARRY_ALPHA * DIGIT_SCALE],
            [CARRY_ALPHA * (-95.0) / CONST_NORM, CARRY_ALPHA * DIGIT_SCALE],
        ]))

        # MLP w2: carry application
        model.mlp.w2.weight.copy_(torch.tensor([
            [0.0, 0.0],
            [-100.0 / CARRY_ALPHA, 100.0 / CARRY_ALPHA],
        ]))


# ---- Submission interface ----

def build_model() -> tuple[nn.Module, dict]:
    model = Wonderfall40Adder()
    _set_weights(model)
    metadata = {
        "name": "wonderfall-40",
        "author": "Wonderfall",
        "architecture": "1-layer decoder, d=2, 1h, head_dim=2, RoPE period-19, parameterless RMSNorm",
        "tricks": "hand-coded weights, tied Q/K + V/O projections, parabolic tied-embed decode, two-hinge ReLU MLP",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    """Encode: [0] + reversed(a_digits) + [0]*9 + reversed(b_digits) + [0].

    31 tokens total. The 9-zero spacer between operands creates a gap of 19
    positions (10 a-digits + 9 zeros) which matches the RoPE period, enabling
    digit routing: positions 19 apart share the same RoPE encoding.
    """
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    a_digits = [int(c) for c in a_str[::-1]]  # reversed: LSD first
    b_digits = [int(c) for c in b_str[::-1]]
    return [0] + a_digits + [0] * 9 + b_digits + [0]


def decode(tokens: list[int]) -> int:
    """Decode 11 reversed digits back to integer."""
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
