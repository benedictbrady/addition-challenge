"""Accuracy evaluation engine."""

import time
from dataclasses import dataclass

import torch.nn as nn

from .harness import generate
from .loader import Submission
from .param_counter import count_unique_parameters
from .test_cases import EDGE_CASES, get_test_cases


@dataclass
class EvalResult:
    edge_correct: int
    edge_total: int
    random_correct: int
    random_total: int
    total_correct: int
    total_total: int
    accuracy: float
    qualified: bool
    param_count: int
    elapsed_seconds: float
    failures: list[tuple[int, int, int, int]]  # (a, b, expected, got)


def evaluate(
    submission: Submission,
    num_random: int = 10000,
    seed: int = 2025,
    device: str = "cpu",
    max_failures: int = 20,
) -> EvalResult:
    """Run full evaluation: generate answers for all test cases, compare with ground truth."""
    model = submission.model
    model.to(device)
    param_count = count_unique_parameters(model)

    test_cases = get_test_cases(num_random=num_random, seed=seed)
    num_edge = len(EDGE_CASES)

    edge_correct = 0
    random_correct = 0
    failures: list[tuple[int, int, int, int]] = []

    start = time.time()

    for i, (a, b) in enumerate(test_cases):
        expected = a + b
        input_tokens = submission.encode(a, b)
        output_tokens = generate(
            model, input_tokens, submission.vocab_size, submission.max_output_len, device=device
        )
        try:
            got = submission.decode(output_tokens)
        except Exception:
            got = -1

        if got == expected:
            if i < num_edge:
                edge_correct += 1
            else:
                random_correct += 1
        else:
            if len(failures) < max_failures:
                failures.append((a, b, expected, got))

    elapsed = time.time() - start

    total_correct = edge_correct + random_correct
    total_total = len(test_cases)
    accuracy = total_correct / total_total if total_total > 0 else 0.0

    return EvalResult(
        edge_correct=edge_correct,
        edge_total=num_edge,
        random_correct=random_correct,
        random_total=num_random,
        total_correct=total_correct,
        total_total=total_total,
        accuracy=accuracy,
        qualified=accuracy >= 0.99,
        param_count=param_count,
        elapsed_seconds=elapsed,
        failures=failures,
    )
