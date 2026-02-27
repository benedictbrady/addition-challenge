"""Baseline: bingbangboom-lab 87-param hand-coded 2-layer Qwen3 adder.

Adapted from: https://gist.github.com/bingbangboom-lab/ec367a6078e9ac2c5748dbbb78eae2a1
Original: 87 params, 100% accuracy, hand-coded weights.
Architecture: 2-layer Qwen3, d=5, 2h/1kv, hd=2, ff=3.
              Cross-layer sharing, rank-1 projections, sparse gate,
              low-rank head, frozen scaling params.

Reimplemented in pure PyTorch.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 10
MAX_OUTPUT_LEN: int = 11

MODEL_LAYERS = 2
MODEL_DIM = 5
ATTENTION_HEADS = 2
KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 3
NUM_DIGITS = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1

LM_HEAD_WEIGHT = np.array([
    [5.5779090e00, 3.1322198e00, -4.0438358e02, 6.2589108e01, 9.9358273e-01],
    [5.0814748e00, 2.4687927e00, -3.1444955e02, 4.8671352e01, 7.7272820e-01],
    [3.6916721e00, 1.7657869e00, -2.2455742e02, 3.4757641e01, 5.5075526e-01],
    [1.4084998e00, 1.0232025e00, -1.3470717e02, 2.0847967e01, 3.2766387e-01],
    [-1.7680415e00, 2.4103954e-01, -4.4898785e01, 6.9423370e00, 1.0345399e-01],
    [-5.8379521e00, -5.8070201e-01, 4.4867714e01, -6.9592528e00, -1.2187435e-01],
    [-1.0801232e01, -1.4420221e00, 1.3459233e02, -2.0856800e01, -3.4832114e-01],
    [-1.6657881e01, -2.3429208e00, 2.2427509e02, -3.4750309e01, -5.7588643e-01],
    [-2.3407900e01, -3.2833982e00, 3.1391595e02, -4.8639774e01, -8.0457014e-01],
    [-3.1051287e01, -4.2634540e00, 4.0351492e02, -6.2525200e01, -1.0343723e00],
], dtype=np.float32)


def _factorize_lowrank(weight, rank):
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    a = u[:, :rank] * s[:rank]
    b = vt[:rank, :]
    return a.astype(np.float32), b.astype(np.float32)


class Rank1Linear(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(out_features, 1))
        self.v = nn.Parameter(torch.zeros(1, in_features))

    def forward(self, x):
        return x @ self.v.T @ self.u.T


class ParameterlessEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(100.0), requires_grad=False)

    def forward(self, ids):
        out = torch.zeros((*ids.shape, MODEL_DIM), dtype=torch.float32, device=ids.device)
        out[..., 0] = self.base
        out[..., 1] = ids.float()
        return out


class LowRankLMHead(nn.Module):
    def __init__(self, vocab_size, dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(vocab_size, rank))
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x):
        return (x @ self.B.T) @ self.A.T


class SparseGateProj0(nn.Module):
    def __init__(self):
        super().__init__()
        self.W23 = nn.Parameter(torch.zeros(2, 3))

    def forward(self, x):
        x3 = x[..., :3]
        y2 = x3 @ self.W23.T
        pad = torch.zeros((*y2.shape[:-1], 1), dtype=y2.dtype, device=y2.device)
        return torch.cat([y2, pad], dim=-1)


class RMSNormNoWeight(nn.Module):
    def __init__(self, eps=1e-6, scale=1.0, count_scale_as_frozen=False):
        super().__init__()
        self.eps = eps
        if count_scale_as_frozen:
            self.scale_param = nn.Parameter(
                torch.tensor(float(scale)), requires_grad=False
            )
            self.scale = None
        else:
            self.scale_param = None
            self.scale = scale

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        s = self.scale_param if self.scale_param is not None else self.scale
        return x * torch.rsqrt(variance + self.eps) * s


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class QProj(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x[..., 0:1] * self.u


class KProj(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x[..., 0:1] * self.u


class ParameterlessVProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.copy_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        out = torch.zeros((*x.shape[:-1], 2), device=x.device, dtype=x.dtype)
        out[..., 0] = self.copy_scale * x[..., 1]
        return out


class OProjL0(nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        out = torch.zeros((*x.shape[:-1], 5), device=x.device, dtype=x.dtype)
        out[..., 2] = self.w0 * x[..., 0] + self.w2 * x[..., 2]
        return out


class OProjL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        out = torch.zeros((*x.shape[:-1], 5), device=x.device, dtype=x.dtype)
        out[..., 4] = self.w0 * x[..., 0] + self.w2 * x[..., 2]
        return out


class UpProjL0(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        s = (x[..., :3] * self.v).sum(-1, keepdim=True)
        out = torch.zeros((*x.shape[:-1], 3), device=x.device, dtype=x.dtype)
        out[..., 0:2] = s.expand(*s.shape[:-1], 2)
        return out


class DownProjL0(nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.w1 = nn.Parameter(torch.tensor(-1.0), requires_grad=False)

    def forward(self, x):
        out = torch.zeros((*x.shape[:-1], 5), device=x.device, dtype=x.dtype)
        out[..., 3] = self.w0 * x[..., 0] + self.w1 * x[..., 1]
        return out


class UpProjL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(5))

    def forward(self, x):
        s = (x * self.v).sum(-1, keepdim=True)
        return s.expand(*s.shape[:-1], 3)


class DownProjL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.w1 = nn.Parameter(torch.tensor(-10.0), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(10.0), requires_grad=False)

    def forward(self, x):
        out = torch.zeros((*x.shape[:-1], 5), device=x.device, dtype=x.dtype)
        out[..., 2] = self.w0 * x[..., 0] + self.w1 * x[..., 1] + self.w2 * x[..., 2]
        return out


class CompressedAttention(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.num_heads = ATTENTION_HEADS
        self.num_kv_heads = KEY_VALUE_HEADS
        self.head_dim = HEAD_DIM
        self.q_proj = QProj(self.num_heads * self.head_dim)
        self.k_proj = KProj(self.num_kv_heads * self.head_dim)
        self.v_proj = ParameterlessVProj()
        self.o_proj = OProjL0() if layer_idx == 0 else OProjL1()
        self.q_norm = RMSNormNoWeight(scale=16.0, count_scale_as_frozen=True)
        self.k_norm = RMSNormNoWeight(scale=16.0, count_scale_as_frozen=True)

    def forward(self, x, freqs_cis, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_norm(q.view(B, S, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(B, S, self.num_kv_heads, self.head_dim))
        v = v.view(B, S, self.num_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(output)


class CompressedLayer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNormNoWeight(scale=1.0)
        self.self_attn = CompressedAttention(layer_idx)
        self.post_attention_layernorm = RMSNormNoWeight(scale=1.0)
        if layer_idx == 0:
            self.gate_proj = SparseGateProj0()
            self.up_proj = UpProjL0()
            self.down_proj = DownProjL0()
        else:
            self.gate_proj = nn.Linear(MODEL_DIM, INTERMEDIATE_SIZE, bias=False)
            self.up_proj = UpProjL1()
            self.down_proj = DownProjL1()

    def forward(self, x, freqs_cis, mask=None):
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask)
        mlp_in = self.post_attention_layernorm(h)
        out = h + self.down_proj(F.silu(self.gate_proj(mlp_in)) * self.up_proj(mlp_in))
        return out


class BingbangboomAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = ParameterlessEmbedding()
        self.layers = nn.ModuleList([CompressedLayer(i) for i in range(MODEL_LAYERS)])
        self.norm = RMSNormNoWeight(scale=1.0)
        self.lm_head = LowRankLMHead(VOCAB_SIZE, MODEL_DIM, rank=2)
        self.freqs_cis = precompute_freqs_cis(HEAD_DIM, 2048)

    def forward(self, x):
        B, S = x.shape
        freqs_cis = self.freqs_cis[:S].to(x.device)
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        return self.lm_head(h)


def _set_weights(model: BingbangboomAdder) -> None:
    with torch.no_grad():
        a, b = _factorize_lowrank(LM_HEAD_WEIGHT, rank=2)
        model.lm_head.A.copy_(torch.from_numpy(a))
        model.lm_head.B.copy_(torch.from_numpy(b))

        l0 = model.layers[0]
        l0.self_attn.q_proj.u.copy_(torch.tensor([
            0.98502123, 0.17243294, 0.96630472, -0.25740093
        ]))
        l0.self_attn.k_proj.u.copy_(torch.tensor([-0.31672141, -0.94851863]))
        l0.gate_proj.W23.copy_(torch.tensor([
            [-3.3532020e-01, -1.3412670e03, 6.0353305e04],
            [-1.3743691e01, -1.3418693e03, 6.0353277e04],
        ]))
        l0.up_proj.v.copy_(torch.tensor([1.4898191e-02, 6.6922739e-04, 2.9977213e-05]))

        l1 = model.layers[1]
        l1.self_attn.q_proj.u.copy_(torch.tensor([
            -0.25507239, 0.96692199, 0.17478994, 0.98460573
        ]))
        l1.self_attn.k_proj.u.copy_(torch.tensor([0.32702553, -0.94501549]))
        l1.gate_proj.weight.copy_(torch.tensor([
            [-4.3951669e-01, 5.6323919e00, 4.9838150e-01, 1.3435575e03, 6.0357680e04],
            [-1.2112466e02, 3.2923722e-01, -5.0313854e00, 1.3449166e03, 6.0357438e04],
            [-1.3453412e02, -2.6000220e-01, -5.6458039e00, 1.3450677e03, 6.0357410e04],
        ]))
        l1.up_proj.v.copy_(torch.tensor([
            1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05
        ]))


def build_model() -> tuple[nn.Module, dict]:
    model = BingbangboomAdder()
    _set_weights(model)
    model.eval()
    metadata = {
        "name": "bingbangboom-87",
        "author": "bingbangboom-lab",
        "architecture": "2-layer Qwen3, d=5, 2h/1kv, hd=2, ff=3",
        "tricks": "hand-coded weights, cross-layer sharing, rank-1 projections, sparse gate, low-rank head, frozen scaling params",
    }
    return model, metadata


def encode(a: int, b: int) -> list[int]:
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    a_digits = [int(c) for c in a_str[::-1]]
    b_digits = [int(c) for c in b_str[::-1]]
    return [0] + a_digits + [0] + [0] + b_digits + [0]


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
