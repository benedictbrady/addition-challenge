"""Tests for the parameter counter."""

import torch
import torch.nn as nn

from addition_challenge.param_counter import count_unique_parameters


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
    """Model with some frozen parameters."""

    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(10, 5, bias=False)  # 50 params
        self.frozen = nn.Linear(10, 5, bias=False)  # 50 params, frozen
        self.frozen.weight.requires_grad = False

    def forward(self, x):
        return self.trainable(x) + self.frozen(x)


class BufferModel(nn.Module):
    """Model with buffers — should not be counted."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)  # 12 params
        self.register_buffer("mask", torch.ones(3))  # buffer, not a parameter

    def forward(self, x):
        return self.linear(x) * self.mask


def test_simple_model():
    model = SimpleModel()
    assert count_unique_parameters(model) == 55


def test_tied_weights():
    model = TiedWeightModel()
    # Weight tying means emb.weight and head.weight share the same tensor
    assert count_unique_parameters(model) == 80  # not 160


def test_frozen_params():
    model = FrozenParamModel()
    assert count_unique_parameters(model) == 50  # only trainable


def test_buffers_not_counted():
    model = BufferModel()
    assert count_unique_parameters(model) == 12  # buffer excluded


def test_empty_model():
    model = nn.Module()
    assert count_unique_parameters(model) == 0
