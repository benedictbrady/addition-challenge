"""Baseline: nino (prasannakotyal) 116-param hand-coded Qwen3-style adder.

Adapted from: https://gist.github.com/prasannakotyal/467d4c54564beba34d9d7edbd41c33dc
Original: 116 params, 100% accuracy, hand-coded weights.
Architecture: 1-layer Qwen3 (GQA 4q/1kv, head_dim=2, d=3),
              tied embed, shared RMSNorm vectors, RoPE (hd=2).

Reimplemented in pure PyTorch.
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
INTERMEDIATE_SIZE = 2
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

MAX_SEQ_LEN = 36


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[-2]
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    cos_p = torch.cos(positions).unsqueeze(-1)
    sin_p = torch.sin(positions).unsqueeze(-1)
    x0 = x[..., :1]
    x1 = x[..., 1:]
    return torch.cat([x0 * cos_p - x1 * sin_p, x0 * sin_p + x1 * cos_p], dim=-1)


class NinoAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(MODEL_DIM, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(MODEL_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(MODEL_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, MODEL_DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        # Weight tying: k_norm shares weights with q_norm
        self.k_norm.weight = self.q_norm.weight

        mask = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM)
        k = self.k_proj(x).view(B, T, NUM_KV_HEADS, HEAD_DIM)
        v = self.v_proj(x).view(B, T, NUM_KV_HEADS, HEAD_DIM)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)
        k = k.expand(B, NUM_HEADS, T, HEAD_DIM)
        v = v.expand(B, NUM_HEADS, T, HEAD_DIM)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, NUM_HEADS * HEAD_DIM)
        return self.o_proj(out)


class NinoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(MODEL_DIM, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(MODEL_DIM, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, MODEL_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NinoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(MODEL_DIM)
        self.self_attn = NinoAttention()
        self.post_attention_layernorm = RMSNorm(MODEL_DIM)
        self.mlp = NinoMLP()
        # Weight tying: post_attention_layernorm shares weights with input_layernorm
        self.post_attention_layernorm.weight = self.input_layernorm.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class NinoAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.block = NinoBlock()
        self.norm = RMSNorm(MODEL_DIM)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(idx)
        x = self.block(x)
        x = self.norm(x)
        return torch.matmul(x, self.embed_tokens.weight.t())


def _qvec(offset: int) -> tuple[float, float]:
    return (math.cos(offset), -math.sin(offset))


def _set_weights(model: NinoAdder) -> None:
    with torch.no_grad():
        for d in range(10):
            model.embed_tokens.weight[d] = torch.tensor([
                EMBED_CONST - DECODE_QUAD * (d * d),
                float(d),
                DECODE_LINEAR_EPS * float(d),
            ])

        model.norm.weight.copy_(torch.tensor([
            1.0 / CONST_NORM,
            CARRY_SLOPE * DIGIT_SCALE * DECODE_LINEAR_EPS,
            DIGIT_SCALE,
        ]))

        model.block.input_layernorm.weight.fill_(1.0)
        model.block.self_attn.q_norm.weight.fill_(QK_NORM_SCALE)

        model.block.self_attn.k_proj.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]))
        model.block.self_attn.v_proj.weight.copy_(torch.tensor([
            [0.0, DIGIT_SCALE, 0.0],
            [0.0, 0.0, 0.0],
        ]))

        q_prev_a = _qvec(23)
        q_prev_b = _qvec(11)
        q_cur_a = _qvec(22)
        q_cur_b = _qvec(10)
        q_w = torch.zeros(NUM_HEADS * HEAD_DIM, MODEL_DIM)
        q_w[0, 0] = q_prev_a[0]; q_w[1, 0] = q_prev_a[1]
        q_w[2, 0] = q_prev_b[0]; q_w[3, 0] = q_prev_b[1]
        q_w[4, 0] = q_cur_a[0]; q_w[5, 0] = q_cur_a[1]
        q_w[6, 0] = q_cur_b[0]; q_w[7, 0] = q_cur_b[1]
        model.block.self_attn.q_proj.weight.copy_(q_w)

        o_w = torch.zeros(MODEL_DIM, NUM_HEADS * HEAD_DIM)
        o_w[1, 0] = -1.0; o_w[1, 2] = -1.0
        o_w[2, 4] = 1.0; o_w[2, 6] = 1.0
        model.block.self_attn.o_proj.weight.copy_(o_w)

        gate_w = torch.zeros(INTERMEDIATE_SIZE, MODEL_DIM)
        gate_w[0, 0] = ALPHA * (-188.0) / CONST_NORM
        gate_w[0, 1] = ALPHA * (-2.0) * DIGIT_SCALE
        gate_w[0, 2] = ALPHA * 20.0 * DIGIT_SCALE
        gate_w[1, 0] = ALPHA * (-189.0) / CONST_NORM
        gate_w[1, 1] = ALPHA * (-2.0) * DIGIT_SCALE
        gate_w[1, 2] = ALPHA * 20.0 * DIGIT_SCALE
        model.block.mlp.gate_proj.weight.copy_(gate_w)

        up_w = torch.zeros(INTERMEDIATE_SIZE, MODEL_DIM)
        up_w[0, 0] = 1.0; up_w[1, 0] = 1.0
        model.block.mlp.up_proj.weight.copy_(up_w)

        scale = 1.0 / (ALPHA * CONST_NORM)
        down_w = torch.zeros(MODEL_DIM, INTERMEDIATE_SIZE)
        down_w[2, 0] = -10.0 * scale
        down_w[2, 1] = 10.0 * scale
        model.block.mlp.down_proj.weight.copy_(down_w)


def build_model() -> tuple[nn.Module, dict]:
    model = NinoAdder()
    _set_weights(model)
    metadata = {
        "name": "nino-116",
        "author": "nino (prasannakotyal)",
        "architecture": "1-layer Qwen3 (GQA 4q/1kv, head_dim=2, d=3), RoPE, RMSNorm",
        "tricks": "hand-coded weights, shared RMSNorm vectors, RoPE digit routing, tied embedding",
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
