"""Shared low-rank transformer architecture used by rezabyt-311 and yinglunz-456.

Adapted from: https://github.com/rezabyt/digit-addition-311p
              https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition

Architecture: 1-layer low-rank GPT with RMSNorm, factorized embeddings,
low-rank QKV/FFN projections, and tied QKV variants.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layer: int = 1
    d_model: int = 7
    n_head: int = 1
    d_ff: int = 14
    dropout: float = 0.0
    max_seq_len: int = 33
    vocab_size: int = 14
    pos_rank: int = 0
    qkv_rank: int = 0
    attn_out_rank: int = 0
    ffn_rank: int = 0
    use_rmsnorm: bool = False
    tie_qkv: str = "none"


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))
        nn.init.normal_(self.A, std=math.sqrt(2.0 / (in_features + rank)))
        nn.init.normal_(self.B, std=math.sqrt(2.0 / (rank + out_features)))

    def forward(self, x):
        return x @ self.A @ self.B


class LowRankEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(num_embeddings, rank))
        self.B = nn.Parameter(torch.empty(rank, embedding_dim))
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.B, std=0.02)

    def forward(self, idx):
        return F.embedding(idx, self.A) @ self.B


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, qkv_rank=0,
                 attn_out_rank=0, tie_qkv="none"):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.tie_qkv = tie_qkv

        if tie_qkv == "shareA_tieKV":
            assert qkv_rank > 0
            self.qkv_A = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_Bq = nn.Parameter(torch.empty(qkv_rank, d_model))
            self.qkv_Bkv = nn.Parameter(torch.empty(qkv_rank, d_model))
            std_a = math.sqrt(2.0 / (d_model + qkv_rank))
            std_b = math.sqrt(2.0 / (qkv_rank + d_model))
            nn.init.normal_(self.qkv_A, std=std_a)
            nn.init.normal_(self.qkv_Bq, std=std_b)
            nn.init.normal_(self.qkv_Bkv, std=std_b)
        else:
            if qkv_rank > 0:
                self.qkv = LowRankLinear(d_model, 3 * d_model, qkv_rank)
            else:
                self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        if attn_out_rank > 0:
            self.proj = LowRankLinear(d_model, d_model, attn_out_rank)
        else:
            self.proj = nn.Linear(d_model, d_model, bias=False)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        bsz, seqlen, d = x.shape
        if self.tie_qkv == "shareA_tieKV":
            h = x @ self.qkv_A
            q = h @ self.qkv_Bq
            k = v = h @ self.qkv_Bkv
        else:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:seqlen, :seqlen], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(bsz, seqlen, d)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, ffn_rank=0):
        super().__init__()
        if ffn_rank > 0:
            self.fc1 = LowRankLinear(d_model, d_ff, ffn_rank)
            self.fc2 = LowRankLinear(d_ff, d_model, ffn_rank)
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=False)
            self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        norm_cls = RMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln1 = norm_cls(cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_head, cfg.max_seq_len,
            qkv_rank=cfg.qkv_rank, attn_out_rank=cfg.attn_out_rank,
            tie_qkv=cfg.tie_qkv,
        )
        self.ln2 = norm_cls(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, ffn_rank=cfg.ffn_rank)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_rank > 0:
            self.pos_emb = LowRankEmbedding(cfg.max_seq_len, cfg.d_model, cfg.pos_rank)
        else:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        norm_cls = RMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln_f = norm_cls(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len:]
            T = self.cfg.max_seq_len
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)
