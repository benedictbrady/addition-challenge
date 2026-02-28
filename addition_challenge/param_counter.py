"""Unique parameter counting (parameters + numeric buffers + forward constants).

Counts all nn.Parameter tensors (including frozen ones) and all
non-boolean registered buffers. This prevents the exploit of hiding
weights in buffers or freezing all params to game the count.

Boolean buffers (e.g. causal attention masks) are excluded because
they encode structure, not learned/designed numerical values.

Additionally provides ``count_forward_constants`` to detect numeric
data created at forward-time via tensor factory functions (e.g.
torch.tensor, torch.arange).  Models that hardcode weights as local
variables inside forward() instead of registering them as parameters
will show a high forward-constant count.
"""

import contextlib
import functools
import unittest.mock
from dataclasses import dataclass

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


# -- Forward-constant counting ------------------------------------------------
# These torch functions create tensors with specific numeric content (not just
# scaffolding like zeros/ones/empty).  If a model creates these inside forward()
# the resulting elements represent hardcoded constant data.

_TENSOR_DATA_FACTORIES = [
    "tensor",
    "arange",
    "linspace",
    "logspace",
    "full",
]


@dataclass
class ForwardConstantStats:
    """Stats from instrumenting tensor creation during a forward pass."""

    total_elements: int
    call_counts: dict[str, int]


@contextlib.contextmanager
def _patch_tensor_factories(stats: ForwardConstantStats):
    """Context manager that monkey-patches torch tensor factories to count elements.

    Only counts non-boolean tensors created by functions that embed specific
    numeric data (torch.tensor, torch.arange, torch.full, etc.).
    """
    originals: dict[str, object] = {}
    patches = []

    for name in _TENSOR_DATA_FACTORIES:
        orig = getattr(torch, name)
        originals[name] = orig

        @functools.wraps(orig)
        def wrapper(*args, _orig=orig, _name=name, **kwargs):
            result = _orig(*args, **kwargs)
            if isinstance(result, torch.Tensor) and result.dtype != torch.bool:
                stats.total_elements += result.numel()
                stats.call_counts[_name] = stats.call_counts.get(_name, 0) + 1
            return result

        patches.append(unittest.mock.patch(f"torch.{name}", wrapper))

    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


def count_forward_constants(
    model: nn.Module,
    sample_input: torch.Tensor,
) -> ForwardConstantStats:
    """Count non-boolean tensor elements created by factory functions during one forward pass.

    This detects models that hide weights as local tensors inside forward()
    instead of registering them as nn.Parameter or buffers.  A high count
    relative to the registered parameter count indicates hardcoded constants.

    Args:
        model: The model to instrument.
        sample_input: A (1, seq_len) long tensor to feed through the model.

    Returns:
        ForwardConstantStats with total element count and per-function breakdown.
    """
    stats = ForwardConstantStats(total_elements=0, call_counts={})
    model.eval()

    with torch.no_grad(), _patch_tensor_factories(stats):
        model(sample_input)

    return stats
