"""Tests for the parameter counter."""

import torch
import torch.nn as nn

from addition_challenge.param_counter import count_forward_constants, count_unique_parameters


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5, bias=True)  # 10*5 + 5 = 55 params

    def forward(self, x):
        return self.linear(x)


class TiedWeightModel(nn.Module):
    """Model with weight tying — shared params should be counted once."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 8)  # 10*8 = 80 params
        self.head = nn.Linear(8, 10, bias=False)  # 10*8 = 80 params
        self.head.weight = self.emb.weight  # tied!

    def forward(self, x):
        return self.head(self.emb(x))


class FrozenParamModel(nn.Module):
    """Model with some frozen parameters — both count now."""

    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(10, 5, bias=False)  # 50 params
        self.frozen = nn.Linear(10, 5, bias=False)  # 50 params, frozen
        self.frozen.weight.requires_grad = False

    def forward(self, x):
        return self.trainable(x) + self.frozen(x)


class BoolBufferModel(nn.Module):
    """Model with a boolean buffer (e.g. attention mask) — NOT counted."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)  # 12 params
        self.register_buffer("mask", torch.ones(5, 5, dtype=torch.bool))

    def forward(self, x):
        return self.linear(x)


class FloatBufferModel(nn.Module):
    """Model with float buffers (weight-like) — counted."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)  # 12 params
        self.register_buffer("scale", torch.ones(3))  # 3 float scalars

    def forward(self, x):
        return self.linear(x) * self.scale


class BufferOnlyModel(nn.Module):
    """Model with only float buffers and no parameters — all counted."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weights", torch.randn(5, 3))  # 15 scalars
        self.register_buffer("bias", torch.randn(3))  # 3 scalars

    def forward(self, x):
        return x @ self.weights + self.bias


class MixedBufferModel(nn.Module):
    """Model with both bool and float buffers — only float counted."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.ones(4, 4, dtype=torch.bool))  # skipped
        self.register_buffer("weights", torch.randn(4, 3))  # 12 counted
        self.register_buffer("indices", torch.arange(4, dtype=torch.long))  # 4 counted (int, not bool)

    def forward(self, x):
        return x @ self.weights


def test_simple_model():
    model = SimpleModel()
    assert count_unique_parameters(model) == 55


def test_tied_weights():
    model = TiedWeightModel()
    assert count_unique_parameters(model) == 80  # not 160


def test_frozen_params_counted():
    model = FrozenParamModel()
    assert count_unique_parameters(model) == 100  # both trainable and frozen


def test_bool_buffer_not_counted():
    model = BoolBufferModel()
    assert count_unique_parameters(model) == 12  # bool mask excluded


def test_float_buffer_counted():
    model = FloatBufferModel()
    assert count_unique_parameters(model) == 15  # 12 params + 3 float buffer


def test_buffer_only_model():
    model = BufferOnlyModel()
    assert count_unique_parameters(model) == 18  # 15 + 3


def test_mixed_buffers():
    model = MixedBufferModel()
    assert count_unique_parameters(model) == 16  # 12 float + 4 int, bool skipped


def test_empty_model():
    model = nn.Module()
    assert count_unique_parameters(model) == 0


# -- Forward constant counting tests ------------------------------------------


class HonestForwardModel(nn.Module):
    """Model that uses only registered parameters in forward."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(14, 8)
        self.head = nn.Linear(8, 14)

    def forward(self, x):
        return self.head(self.emb(x))


class ConstantForwardModel(nn.Module):
    """Model that creates constant tensors inside forward (the cheat pattern)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, T = x.shape
        # Hardcoded embedding table via torch.tensor — this is the cheat
        weights = torch.tensor([
            [float(i) * 10.0 for _ in range(8)]
            for i in range(14)
        ])
        # Hardcoded output projection via torch.arange
        scale = torch.arange(1, 15, dtype=torch.float32)
        h = weights[x]  # (B, T, 8)
        logits = h.sum(dim=-1, keepdim=True) * scale.unsqueeze(0).unsqueeze(0)
        return logits


def test_forward_constants_honest():
    """Honest model should create few forward constants."""
    model = HonestForwardModel()
    sample = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    stats = count_forward_constants(model, sample)
    # Honest model doesn't use torch.tensor/arange in forward
    assert stats.total_elements == 0


def test_forward_constants_cheat():
    """Model with hardcoded constants in forward should have high count."""
    model = ConstantForwardModel()
    sample = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    stats = count_forward_constants(model, sample)
    # torch.tensor creates 14*8=112 elements, torch.arange creates 14
    assert stats.total_elements >= 100
    assert "tensor" in stats.call_counts
    assert "arange" in stats.call_counts
