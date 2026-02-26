"""Dynamic submission module loader."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch.nn as nn


@dataclass
class Submission:
    model: nn.Module
    metadata: dict
    encode: Callable[[int, int], list[int]]
    decode: Callable[[list[int]], int]
    vocab_size: int
    max_output_len: int
    module: Any  # the raw module, for AST inspection


def load_submission(path: str) -> Submission:
    """Load a submission .py file and extract the 5 required exports."""
    filepath = Path(path).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Submission file not found: {filepath}")
    if not filepath.suffix == ".py":
        raise ValueError(f"Submission must be a .py file, got: {filepath}")

    module_name = f"submission_{filepath.stem}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Validate required exports
    required = ["build_model", "encode", "decode", "VOCAB_SIZE", "MAX_OUTPUT_LEN"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Submission missing required exports: {missing}")

    # Validate types
    vocab_size = module.VOCAB_SIZE
    max_output_len = module.MAX_OUTPUT_LEN
    if not isinstance(vocab_size, int):
        raise TypeError(f"VOCAB_SIZE must be int, got {type(vocab_size)}")
    if not isinstance(max_output_len, int):
        raise TypeError(f"MAX_OUTPUT_LEN must be int, got {type(max_output_len)}")

    # Build model
    result = module.build_model()
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("build_model() must return (model, metadata) tuple")
    model, metadata = result
    if not isinstance(model, nn.Module):
        raise TypeError(f"build_model() model must be nn.Module, got {type(model)}")
    if not isinstance(metadata, dict):
        raise TypeError(f"build_model() metadata must be dict, got {type(metadata)}")

    return Submission(
        model=model,
        metadata=metadata,
        encode=module.encode,
        decode=module.decode,
        vocab_size=vocab_size,
        max_output_len=max_output_len,
        module=module,
    )
