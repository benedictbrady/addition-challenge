"""Unique trainable parameter counting."""

import torch.nn as nn


def count_unique_parameters(model: nn.Module) -> int:
    """Count unique trainable parameters, handling weight tying and frozen params.

    Deduplicates by data_ptr() so tied weights are only counted once.
    Skips parameters with requires_grad=False (frozen/non-trainable).
    """
    seen_ptrs: set[int] = set()
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total += param.numel()
    return total
