"""Baseline: Wonderfall 121-param hand-coded Qwen3-style adder.

Adapted from: https://gist.github.com/Wonderfall/7d6f49aa6703352f94d3d80b4cd31e15
Original: 121 params, 100% accuracy, hand-coded weights using MLX Qwen3.
Architecture: 1-layer Qwen3 (GQA 4q/1kv, head_dim=2, d=3, ff=2),
              RoPE digit routing, SiLU carry logic, RMSNorm.

Reimplemented in pure PyTorch (original used Apple MLX).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 10  # digits 0-9 only
MAX_OUTPUT_LEN: int = 11  # 11 reversed digits (no EOS)

# Model constants (from original)
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

MAX_SEQ_LEN = 36  # 24 input + 11 output + 1 margin


# ---- Building blocks ----

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


def apply_rope(x: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Apply RoPE to tensor of shape (..., seq_len, 2).

    With head_dim=2 and theta=10000, inv_freq=1.0, so angle = position.
    Non-traditional (GPT-NeoX) style: split at midpoint.
    """
    seq_len = x.shape[-2]
    positions = torch.arange(offset, offset + seq_len, device=x.device, dtype=x.dtype)
    # Shape: (seq_len, 1) for broadcasting
    cos_p = torch.cos(positions).unsqueeze(-1)  # (T, 1)
    sin_p = torch.sin(positions).unsqueeze(-1)  # (T, 1)

    x0 = x[..., :1]  # first half
    x1 = x[..., 1:]  # second half
    return torch.cat([x0 * cos_p - x1 * sin_p, x0 * sin_p + x1 * cos_p], dim=-1)


class WonderfallAttention(nn.Module):
    """GQA attention with 4 query heads, 1 KV head, head_dim=2, QK-norm, RoPE."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(MODEL_DIM, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(MODEL_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(MODEL_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, MODEL_DIM, bias=False)

        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

        mask = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)  # (B, T, 8)
        k = self.k_proj(x)  # (B, T, 2)
        v = self.v_proj(x)  # (B, T, 2)

        # Reshape to per-head view
        q = q.view(B, T, NUM_HEADS, HEAD_DIM)   # (B, T, 4, 2)
        k = k.view(B, T, NUM_KV_HEADS, HEAD_DIM)  # (B, T, 1, 2)
        v = v.view(B, T, NUM_KV_HEADS, HEAD_DIM)  # (B, T, 1, 2)

        # QK-norm BEFORE RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, heads, T, head_dim) for RoPE and attention
        q = q.transpose(1, 2)  # (B, 4, T, 2)
        k = k.transpose(1, 2)  # (B, 1, T, 2)
        v = v.transpose(1, 2)  # (B, 1, T, 2)

        # Apply RoPE
        q = apply_rope(q)
        k = apply_rope(k)

        # GQA: expand KV to match query heads
        k = k.expand(B, NUM_HEADS, T, HEAD_DIM)  # (B, 4, T, 2)
        v = v.expand(B, NUM_HEADS, T, HEAD_DIM)  # (B, 4, T, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(HEAD_DIM)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, 4, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v  # (B, 4, T, 2)
        out = out.transpose(1, 2).contiguous().view(B, T, NUM_HEADS * HEAD_DIM)  # (B, T, 8)
        return self.o_proj(out)  # (B, T, 3)


class WonderfallMLP(nn.Module):
    """SwiGLU MLP: silu(gate) * up -> down."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(MODEL_DIM, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(MODEL_DIM, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, MODEL_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class WonderfallBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(MODEL_DIM)
        self.self_attn = WonderfallAttention()
        self.post_attention_layernorm = RMSNorm(MODEL_DIM)
        self.mlp = WonderfallMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class WonderfallAdder(nn.Module):
    """121-param Qwen3-style transformer reimplemented in PyTorch."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.block = WonderfallBlock()
        self.norm = RMSNorm(MODEL_DIM)
        self.lm_head = nn.Linear(MODEL_DIM, VOCAB_SIZE, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(idx)       # (B, T, 3)
        x = self.block(x)                # (B, T, 3)
        x = self.norm(x)                 # (B, T, 3)
        logits = self.lm_head(x)         # (B, T, 10)
        return logits


# ---- Hand-coded weight initialization ----

def _qvec(offset: int) -> tuple[float, float]:
    return (math.cos(offset), -math.sin(offset))


def _set_weights(model: WonderfallAdder) -> None:
    """Hand-set all weights to implement addition."""
    with torch.no_grad():
        # Embedding: digit d -> [EMBED_CONST - DECODE_QUAD*d², d, DECODE_LINEAR_EPS*d]
        for d in range(10):
            model.embed_tokens.weight[d] = torch.tensor([
                EMBED_CONST - DECODE_QUAD * (d * d),
                float(d),
                DECODE_LINEAR_EPS * float(d),
            ])

        # Final norm
        model.norm.weight.copy_(torch.tensor([
            1.0 / CONST_NORM,
            CARRY_SLOPE * DIGIT_SCALE * DECODE_LINEAR_EPS,
            DIGIT_SCALE,
        ]))

        # Layer norms (identity)
        model.block.input_layernorm.weight.fill_(1.0)
        model.block.post_attention_layernorm.weight.fill_(1.0)

        # QK norm scales
        model.block.self_attn.q_norm.weight.fill_(QK_NORM_SCALE)
        model.block.self_attn.k_norm.weight.fill_(QK_NORM_SCALE)

        # K projection: picks up first component of hidden state
        model.block.self_attn.k_proj.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]))

        # V projection: picks up second component (digit value) scaled
        model.block.self_attn.v_proj.weight.copy_(torch.tensor([
            [0.0, DIGIT_SCALE, 0.0],
            [0.0, 0.0, 0.0],
        ]))

        # Q projection: 4 heads with different RoPE offsets for digit routing
        # Head 0 (rows 0-1): offset 23 → attends to current a digit
        # Head 1 (rows 2-3): offset 11 → attends to current b digit
        # Head 2 (rows 4-5): offset 22 → attends to previous a digit
        # Head 3 (rows 6-7): offset 10 → attends to previous b digit
        q_prev_a = _qvec(23)
        q_prev_b = _qvec(11)
        q_cur_a = _qvec(22)
        q_cur_b = _qvec(10)

        q_w = torch.zeros(NUM_HEADS * HEAD_DIM, MODEL_DIM)
        q_w[0, 0] = q_prev_a[0]
        q_w[1, 0] = q_prev_a[1]
        q_w[2, 0] = q_prev_b[0]
        q_w[3, 0] = q_prev_b[1]
        q_w[4, 0] = q_cur_a[0]
        q_w[5, 0] = q_cur_a[1]
        q_w[6, 0] = q_cur_b[0]
        q_w[7, 0] = q_cur_b[1]
        model.block.self_attn.q_proj.weight.copy_(q_w)

        # O projection: routes attention outputs to hidden dimensions
        import numpy as np
        o_w = np.zeros((MODEL_DIM, NUM_HEADS * HEAD_DIM), dtype=np.float32)
        o_w[1, 0] = -1.0   # head 0 output → dim 1 (negative)
        o_w[1, 2] = -1.0   # head 1 output → dim 1 (negative)
        o_w[2, 4] = 1.0    # head 2 output → dim 2
        o_w[2, 6] = 1.0    # head 3 output → dim 2
        model.block.self_attn.o_proj.weight.copy_(torch.tensor(o_w))

        # MLP gate projection: carry detection logic
        gate_w = torch.zeros(INTERMEDIATE_SIZE, MODEL_DIM)
        gate_b_vals = torch.zeros(INTERMEDIATE_SIZE)  # stored in weight since no bias
        gate_w[0, 0] = ALPHA * (-188.0) / CONST_NORM
        gate_w[0, 1] = ALPHA * (-2.0) * DIGIT_SCALE
        gate_w[0, 2] = ALPHA * 20.0 * DIGIT_SCALE
        gate_w[1, 0] = ALPHA * (-189.0) / CONST_NORM
        gate_w[1, 1] = ALPHA * (-2.0) * DIGIT_SCALE
        gate_w[1, 2] = ALPHA * 20.0 * DIGIT_SCALE
        model.block.mlp.gate_proj.weight.copy_(gate_w)

        # MLP up projection
        up_w = torch.zeros(INTERMEDIATE_SIZE, MODEL_DIM)
        up_w[0, 0] = 1.0
        up_w[1, 0] = 1.0
        model.block.mlp.up_proj.weight.copy_(up_w)

        # MLP down projection
        scale = 1.0 / (ALPHA * CONST_NORM)
        down_w = torch.zeros(MODEL_DIM, INTERMEDIATE_SIZE)
        down_w[2, 0] = -10.0 * scale
        down_w[2, 1] = 10.0 * scale
        model.block.mlp.down_proj.weight.copy_(down_w)


# ---- Submission interface ----

def build_model() -> tuple[nn.Module, dict]:
    model = WonderfallAdder()
    _set_weights(model)
    metadata = {
        "name": "wonderfall-121",
        "author": "Wonderfall",
        "architecture": "1-layer Qwen3 (GQA 4q/1kv, head_dim=2, d=3, ff=2), RoPE, RMSNorm",
        "tricks": "hand-coded weights, RoPE digit routing, SiLU carry logic, tied embedding",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    """Encode: [0] + reversed(a_digits) + [0,0] + reversed(b_digits) + [0].

    24 tokens total. Reversed digit order (LSD first) with 0-delimiters.
    """
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    a_digits = [int(c) for c in a_str[::-1]]  # reversed: LSD first
    b_digits = [int(c) for c in b_str[::-1]]
    return [0] + a_digits + [0, 0] + b_digits + [0]


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
