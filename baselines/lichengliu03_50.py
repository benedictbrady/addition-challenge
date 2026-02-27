"""Baseline: lichengliu03 50-param hand-coded custom GPT adder.

Adapted from: https://github.com/lichengliu03/TinyAdder-50p
Original: 50 params, 100% accuracy, hand-coded weights.
Architecture: 1-layer custom GPT, d=4, 2h, hd=2.
              Factorized embed, rotation Q (2 angles), tied embed+V dir,
              rank-1 MLP, parabolic head, sinusoidal PE (period 11).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE: int = 10
MAX_OUTPUT_LEN: int = 11

NUM_DIGITS = 10
OUTPUT_DIGITS = 11
D_MODEL = 4
N_HEAD = 2
HEAD_DIM = 2
THETA = 2 * math.pi / 11


class TinyAdderAttention(nn.Module):
    """Custom attention with rotation Q, shared K projection, scalar V."""

    def __init__(self):
        super().__init__()
        self.K_proj = nn.Parameter(torch.zeros(HEAD_DIM, D_MODEL))
        self.q_angles = nn.Parameter(torch.zeros(N_HEAD))
        self.o_dir = nn.Parameter(torch.zeros(N_HEAD, D_MODEL))

    def forward(self, x, embed_dir):
        B, T, _ = x.shape
        device = x.device

        pe_extracted = x @ self.K_proj.T
        k = pe_extracted

        v_scalar = (x * embed_dir).sum(-1)

        qs = []
        for h in range(N_HEAD):
            a = self.q_angles[h]
            c, s = torch.cos(a), torch.sin(a)
            q0 = -c * pe_extracted[:, :, 0] + s * pe_extracted[:, :, 1]
            q1 = s * pe_extracted[:, :, 0] + c * pe_extracted[:, :, 1]
            qs.append(torch.stack([q0, q1], dim=-1))

        q = torch.stack(qs, dim=1)
        k = k.unsqueeze(1).expand(-1, N_HEAD, -1, -1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)

        v_exp = v_scalar.unsqueeze(1).unsqueeze(-1).expand(-1, N_HEAD, -1, -1)
        weighted_v = (attn_weights @ v_exp).squeeze(-1)

        attn_out = torch.zeros(B, T, D_MODEL, device=device, dtype=x.dtype)
        for h in range(N_HEAD):
            attn_out = attn_out + weighted_v[:, h, :, None] * self.o_dir[h]
        return attn_out


class TinyAdder50(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit_values = nn.Parameter(torch.zeros(VOCAB_SIZE, 1))
        self.embed_dir = nn.Parameter(torch.zeros(D_MODEL))
        self.attn = TinyAdderAttention()
        self.carry_dir = nn.Parameter(torch.zeros(D_MODEL))
        self.wrap_dir = nn.Parameter(torch.zeros(D_MODEL))
        self.carry_bias = nn.Parameter(torch.zeros(2))
        self.wrap_bias = nn.Parameter(torch.zeros(2))
        self.accum_dir = nn.Parameter(torch.zeros(D_MODEL))
        self.head_lin = nn.Parameter(torch.zeros(1))
        self.head_quad = nn.Parameter(torch.zeros(1))

    def generate_pe(self, seq_len, device):
        pe = torch.zeros(seq_len, D_MODEL, device=device)
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        amp = torch.where(pos <= 21, 100.0, 1.0)
        pe[:, 1] = amp * torch.sin(pos * THETA)
        pe[:, 2] = amp * torch.cos(pos * THETA)
        return pe

    def forward(self, idx):
        B, T = idx.size()
        device = idx.device

        x = self.digit_values[idx] * self.embed_dir + self.generate_pe(T, device)
        x = x + self.attn(x, self.embed_dir)

        ci = (x * self.carry_dir).sum(-1)
        carry = F.relu(ci + self.carry_bias[0]) - F.relu(ci + self.carry_bias[1])

        wi = (x * self.wrap_dir).sum(-1)
        wrap = F.relu(wi + self.wrap_bias[1]) - F.relu(wi + self.wrap_bias[0])

        mlp_out = (carry + wrap).unsqueeze(-1) * self.accum_dir
        x = x + mlp_out

        d = torch.arange(VOCAB_SIZE, device=device, dtype=x.dtype)
        z = (x * self.accum_dir).sum(-1, keepdim=True)
        logits = self.head_lin * d * z + self.head_quad * d * d
        return logits


def _set_weights(model: TinyAdder50) -> None:
    with torch.no_grad():
        model.digit_values[:] = torch.arange(VOCAB_SIZE).float().unsqueeze(1)
        model.embed_dir[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])

        model.attn.K_proj[:] = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        model.attn.q_angles[:] = torch.tensor([8 * THETA, 9 * THETA])
        model.attn.o_dir[:] = torch.tensor([
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0, 0.0],
        ])

        model.carry_dir[:] = torch.tensor([-1.0, 1.0, 0.0, 0.0])
        model.carry_bias[:] = torch.tensor([-0.5, -1.5])

        model.wrap_dir[:] = torch.tensor([-10.0, 10.0, 0.0, 1000.0])
        model.wrap_bias[:] = torch.tensor([-9045.0, -9055.0])

        model.accum_dir[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        model.head_lin.fill_(2.0)
        model.head_quad.fill_(-1.0)


def build_model() -> tuple[nn.Module, dict]:
    model = TinyAdder50()
    _set_weights(model)
    model.eval()
    metadata = {
        "name": "lichengliu03-50",
        "author": "lichengliu03",
        "architecture": "1-layer custom GPT, d=4, 2h, hd=2",
        "tricks": "hand-coded weights, factorized embed, rotation Q, tied embed+V, rank-1 MLP, parabolic head, sinusoidal PE (period 11)",
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
