"""Tests for the autoregressive generation harness."""

import torch
import torch.nn as nn

from addition_challenge.harness import generate


class EchoModel(nn.Module):
    """Trivial model that always predicts the next token as (last_token + 1) % vocab_size."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        # Need at least one parameter to be a valid nn.Module
        self.dummy = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        # Predict (last_token + 1) % vocab_size for each position
        logits = torch.zeros(B, T, self.vocab_size)
        for b in range(B):
            for t in range(T):
                next_tok = (x[b, t].item() + 1) % self.vocab_size
                logits[b, t, next_tok] = 100.0  # high logit for the predicted token
        return logits


def test_generate_length():
    """Generated output should have exactly max_output_len tokens."""
    model = EchoModel(vocab_size=10)
    input_tokens = [1, 2, 3]
    result = generate(model, input_tokens, vocab_size=10, max_output_len=5)
    assert len(result) == 5


def test_generate_deterministic():
    """Same input should produce same output."""
    model = EchoModel(vocab_size=10)
    input_tokens = [1, 2, 3]
    result1 = generate(model, input_tokens, vocab_size=10, max_output_len=5)
    result2 = generate(model, input_tokens, vocab_size=10, max_output_len=5)
    assert result1 == result2


def test_generate_autoregressive():
    """Verify the echo model produces expected sequence: each token = (prev + 1) % vocab."""
    model = EchoModel(vocab_size=10)
    input_tokens = [3]
    result = generate(model, input_tokens, vocab_size=10, max_output_len=4)
    # After input [3], model predicts 4, then with [3,4] last is 4 → predicts 5, etc.
    assert result == [4, 5, 6, 7]


def test_generate_empty_input():
    """Generation should work with a single input token."""
    model = EchoModel(vocab_size=10)
    input_tokens = [0]
    result = generate(model, input_tokens, vocab_size=10, max_output_len=3)
    assert len(result) == 3
    assert result == [1, 2, 3]
