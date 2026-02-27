"""Baseline: corbensorenson 111-param hand-coded Qwen3-style adder.

Adapted from: https://github.com/corbensorenson/adderboard-submissions
Original: 111 params, 100% accuracy, hand-coded weights using Codex.
Architecture: 1-layer decoder (GQA 4q/1kv, head_dim=2, d=3, ff=2),
              tied embed, RoPE, SwiGLU, GQA.

Reimplemented in pure PyTorch. Uses parameter-free pre-attention RMS norms
(ones vectors, not counted) and a final RMSNorm with learned weights.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 10
MAX_OUTPUT_LEN: int = 11

MODEL_DIM = 3
NUM_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 2
FF_DIM = 2
NUM_DIGITS = 10
OUTPUT_DIGITS = 11

EMBED_CONST = 1000.0
DIGIT_SCALE = EMBED_CONST / math.sqrt(MODEL_DIM)
CONST_NORM = math.sqrt(MODEL_DIM)
ALPHA = 20.0
QK_NORM_SCALE = 256.0
DECODE_LINEAR_EPS = 5e-4
DECODE_QUAD = DECODE_LINEAR_EPS / 2.0
CARRY_SLOPE = -0.1
EPS = 1e-6

MAX_SEQ_LEN = 36


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    """RoPE for shape (B, T, H, D) where D=2 and theta=10000."""
    bsz, seqlen, _, hdim = x.shape
    half = hdim // 2
    inv_freq = torch.exp(
        -math.log(10000.0) * (torch.arange(half, dtype=x.dtype, device=x.device) / half)
    )
    pos = torch.arange(seqlen, dtype=x.dtype, device=x.device)
    ang = pos[:, None] * inv_freq[None, :]
    cs = torch.cos(ang)[None, :, None, :]
    sn = torch.sin(ang)[None, :, None, :]
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * cs - xo * sn
    out[..., 1::2] = xe * sn + xo * cs
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight


class CorbenAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Parameter(torch.zeros(NUM_HEADS * HEAD_DIM, MODEL_DIM))
        self.k_proj = nn.Parameter(torch.zeros(NUM_KV_HEADS * HEAD_DIM, MODEL_DIM))
        self.v_proj = nn.Parameter(torch.zeros(NUM_KV_HEADS * HEAD_DIM, MODEL_DIM))
        self.o_proj = nn.Parameter(torch.zeros(MODEL_DIM, NUM_HEADS * HEAD_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = F.linear(x, self.q_proj).view(bsz, seqlen, NUM_HEADS, HEAD_DIM)
        k = F.linear(x, self.k_proj).view(bsz, seqlen, NUM_KV_HEADS, HEAD_DIM)
        v = F.linear(x, self.v_proj).view(bsz, seqlen, NUM_KV_HEADS, HEAD_DIM)

        qk_scale = torch.full((HEAD_DIM,), QK_NORM_SCALE, dtype=q.dtype, device=q.device)
        q = _apply_rope(_rms_norm(q, qk_scale))
        k = _apply_rope(_rms_norm(k, qk_scale))

        k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=2)
        v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(
            torch.full((seqlen, seqlen), float("-inf"), dtype=scores.dtype, device=scores.device),
            diagonal=1,
        )
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, NUM_HEADS * HEAD_DIM)
        return F.linear(out, self.o_proj)


class CorbenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Parameter(torch.zeros(FF_DIM, MODEL_DIM))
        self.up_proj = nn.Parameter(torch.zeros(FF_DIM, MODEL_DIM))
        self.down_proj = nn.Parameter(torch.zeros(MODEL_DIM, FF_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            F.silu(F.linear(x, self.gate_proj)) * F.linear(x, self.up_proj),
            self.down_proj,
        )


class CorbenAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Parameter(torch.zeros(VOCAB_SIZE, MODEL_DIM))
        self.attn = CorbenAttention()
        self.mlp = CorbenMLP()
        self.final_norm = RMSNorm(MODEL_DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.embedding(tokens, self.embed_tokens)
        ones = torch.ones(MODEL_DIM, dtype=x.dtype, device=x.device)
        x = x + self.attn(_rms_norm(x, ones))
        x = x + self.mlp(_rms_norm(x, ones))
        x = self.final_norm(x)
        return torch.matmul(x, self.embed_tokens.t())


def _qvec(offset: float) -> tuple[float, float]:
    return (math.cos(offset), -math.sin(offset))


def _set_weights(model: CorbenAdder) -> None:
    with torch.no_grad():
        emb = torch.tensor([
            [EMBED_CONST - DECODE_QUAD * (d * d), float(d), DECODE_LINEAR_EPS * float(d)]
            for d in range(10)
        ])
        model.embed_tokens.copy_(emb)

        model.final_norm.weight.copy_(torch.tensor([
            1.0 / CONST_NORM,
            CARRY_SLOPE * DIGIT_SCALE * DECODE_LINEAR_EPS,
            DIGIT_SCALE,
        ]))

        model.attn.k_proj.zero_()
        model.attn.k_proj[0, 0] = 1.0
        model.attn.v_proj.zero_()
        model.attn.v_proj[0, 1] = DIGIT_SCALE

        q_prev_a = _qvec(23.0)
        q_prev_b = _qvec(11.0)
        q_cur_a = _qvec(22.0)
        q_cur_b = _qvec(10.0)
        model.attn.q_proj.copy_(torch.tensor([
            [q_prev_a[0], 0.0, 0.0], [q_prev_a[1], 0.0, 0.0],
            [q_prev_b[0], 0.0, 0.0], [q_prev_b[1], 0.0, 0.0],
            [q_cur_a[0], 0.0, 0.0], [q_cur_a[1], 0.0, 0.0],
            [q_cur_b[0], 0.0, 0.0], [q_cur_b[1], 0.0, 0.0],
        ]))

        model.attn.o_proj.zero_()
        model.attn.o_proj[1, 0] = -1.0
        model.attn.o_proj[1, 2] = -1.0
        model.attn.o_proj[2, 4] = 1.0
        model.attn.o_proj[2, 6] = 1.0

        gate = torch.zeros(FF_DIM, MODEL_DIM)
        gate[0, 0] = ALPHA * (-188.0) / CONST_NORM
        gate[0, 1] = ALPHA * (-2.0) * DIGIT_SCALE
        gate[0, 2] = ALPHA * 20.0 * DIGIT_SCALE
        gate[1, 0] = ALPHA * (-189.0) / CONST_NORM
        gate[1, 1] = ALPHA * (-2.0) * DIGIT_SCALE
        gate[1, 2] = ALPHA * 20.0 * DIGIT_SCALE
        model.mlp.gate_proj.copy_(gate)

        model.mlp.up_proj.zero_()
        model.mlp.up_proj[0, 0] = 1.0
        model.mlp.up_proj[1, 0] = 1.0

        model.mlp.down_proj.zero_()
        scale = 1.0 / (ALPHA * CONST_NORM)
        model.mlp.down_proj[2, 0] = -10.0 * scale
        model.mlp.down_proj[2, 1] = 10.0 * scale


def build_model() -> tuple[nn.Module, dict]:
    model = CorbenAdder()
    _set_weights(model)
    model.eval()
    metadata = {
        "name": "corbensorenson-111",
        "author": "corbensorenson",
        "architecture": "1-layer decoder (GQA 4q/1kv, head_dim=2, d=3, ff=2), RoPE, SwiGLU",
        "tricks": "hand-coded weights, tied embed, parameter-free pre-attn RMS norms, RoPE",
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
