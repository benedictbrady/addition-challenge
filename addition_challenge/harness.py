"""Autoregressive generation loop — the core of the evaluation harness.

The harness controls generation, not the user's model. The model only provides
forward(token_ids) -> logits. This makes autoregressive behavior unforgeable.
"""

import torch
import torch.nn as nn


def generate(
    model: nn.Module,
    input_tokens: list[int],
    vocab_size: int,
    max_output_len: int,
    device: str = "cpu",
) -> list[int]:
    """Generate output tokens autoregressively.

    The model receives the full context (input + generated so far) at each step.
    We take argmax of the last position's logits and append to context.
    The model never sees future output tokens.
    """
    model.eval()
    context = torch.tensor([input_tokens], dtype=torch.long, device=device)
    generated: list[int] = []

    with torch.no_grad():
        for _ in range(max_output_len):
            logits = model(context)  # (1, seq_len, vocab_size)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            next_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device=device
            )
            context = torch.cat([context, next_tensor], dim=1)

    return generated
