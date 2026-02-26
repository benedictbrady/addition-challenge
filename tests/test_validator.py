"""Tests for the submission validator."""

import torch
import torch.nn as nn

from addition_challenge.loader import Submission
from addition_challenge.validator import (
    _check_causal_behavior,
    _check_decode_honesty,
    _check_encode_bounds,
    _check_encode_consistency,
    _check_interface,
    _check_structural_attention,
)


class SimpleAttentionModel(nn.Module):
    """Minimal model with attention for testing."""

    def __init__(self, vocab_size: int = 14, d_model: int = 16, max_seq_len: int = 30):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        return self.head(h)


class MLPModel(nn.Module):
    """Model without attention — should fail structural check."""

    def __init__(self, vocab_size: int = 14, d_model: int = 16, max_seq_len: int = 30):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.emb(x))


class NonCausalModel(nn.Module):
    """Model that looks at future tokens — should fail causal check."""

    def __init__(self, vocab_size: int = 14, d_model: int = 16, max_seq_len: int = 30):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        # No causal mask — bidirectional attention
        h, _ = self.attn(h, h, h)
        return self.head(h)


def _make_submission(model, encode_fn=None, decode_fn=None, vocab_size=14, max_output_len=12):
    if encode_fn is None:
        def encode_fn(a, b):
            a_digits = [int(d) for d in str(a)] if a > 0 else [0]
            b_digits = [int(d) for d in str(b)] if b > 0 else [0]
            return a_digits + [10] + b_digits + [11]

    if decode_fn is None:
        def decode_fn(tokens):
            digits = [t for t in tokens if 0 <= t <= 9]
            if not digits:
                return 0
            return int("".join(str(d) for d in reversed(digits)))

    return Submission(
        model=model,
        metadata={"name": "test"},
        encode=encode_fn,
        decode=decode_fn,
        vocab_size=vocab_size,
        max_output_len=max_output_len,
        module=None,
    )


def test_structural_attention_pass():
    model = SimpleAttentionModel()
    sub = _make_submission(model)
    result = _check_structural_attention(sub)
    assert result.passed


def test_structural_attention_fail_mlp():
    model = MLPModel()
    sub = _make_submission(model)
    result = _check_structural_attention(sub)
    assert not result.passed


def test_causal_behavior_pass():
    model = SimpleAttentionModel()
    sub = _make_submission(model)
    result = _check_causal_behavior(sub)
    assert result.passed


def test_causal_behavior_fail():
    model = NonCausalModel()
    sub = _make_submission(model)
    result = _check_causal_behavior(sub)
    assert not result.passed


def test_interface_bounds_pass():
    model = SimpleAttentionModel()
    sub = _make_submission(model, vocab_size=14, max_output_len=12)
    result = _check_interface(sub)
    assert result.passed


def test_interface_bounds_fail_vocab():
    model = SimpleAttentionModel()
    sub = _make_submission(model, vocab_size=300)
    result = _check_interface(sub)
    assert not result.passed


def test_interface_bounds_fail_output_len():
    model = SimpleAttentionModel()
    sub = _make_submission(model, max_output_len=50)
    result = _check_interface(sub)
    assert not result.passed


def test_encode_bounds_pass():
    model = SimpleAttentionModel()
    sub = _make_submission(model)
    results = _check_encode_bounds(sub)
    assert all(r.passed for r in results)


def test_encode_bounds_fail_range():
    def bad_encode(a, b):
        return [999, 0, 1]  # 999 is out of range for vocab_size=14

    model = SimpleAttentionModel()
    sub = _make_submission(model, encode_fn=bad_encode)
    results = _check_encode_bounds(sub)
    assert any(not r.passed for r in results)


def test_encode_consistency_pass():
    model = SimpleAttentionModel()
    sub = _make_submission(model)
    result = _check_encode_consistency(sub)
    assert result.passed


def test_decode_honesty_pass():
    def honest_decode(tokens):
        digits = [t for t in tokens if 0 <= t <= 9]
        if not digits:
            return 0
        return int("".join(str(d) for d in reversed(digits)))

    model = SimpleAttentionModel()
    sub = _make_submission(model, decode_fn=honest_decode)
    result = _check_decode_honesty(sub)
    assert result.passed
