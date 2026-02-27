"""Unique parameter counting (parameters + numeric buffers).

Counts all nn.Parameter tensors (including frozen ones) and all
non-boolean registered buffers. This prevents the exploit of hiding
weights in buffers or freezing all params to game the count.

Boolean buffers (e.g. causal attention masks) are excluded because
they encode structure, not learned/designed numerical values.
"""

import torch
import torch.nn as nn


def count_unique_parameters(model: nn.Module) -> int:
    """Count unique stored scalars across parameters and numeric buffers.

    Deduplicates by data_ptr() so tied weights are only counted once.
    Counts all nn.Parameter tensors regardless of requires_grad, plus
    any non-boolean registered buffers (float, int, complex, etc.).
    """
    seen_ptrs: set[int] = set()
    total = 0

    # All parameters (trainable or frozen)
    for _, param in model.named_parameters():
        ptr = param.data.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total += param.numel()

    # Non-boolean buffers (skips structural masks)
    for _, buf in model.named_buffers():
        if buf.dtype == torch.bool:
            continue
        ptr = buf.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total += buf.numel()

    return total
