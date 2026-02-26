"""Test case generation for addition evaluation."""

import random

# Edge cases matching AdderBoard patterns
EDGE_CASES: list[tuple[int, int]] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (9999999999, 0),
    (0, 9999999999),
    (9999999999, 9999999999),
    (9999999999, 1),
    (1, 9999999999),
    (5555555555, 4444444445),  # full carry chain
    (1111111111, 8888888889),  # full carry chain
]


def generate_random_pairs(n: int = 10000, seed: int = 2025) -> list[tuple[int, int]]:
    """Generate random 10-digit addition pairs with fixed seed."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        a = rng.randint(0, 9999999999)
        b = rng.randint(0, 9999999999)
        pairs.append((a, b))
    return pairs


def get_test_cases(num_random: int = 10000, seed: int = 2025) -> list[tuple[int, int]]:
    """Get full test suite: edge cases + random pairs."""
    return EDGE_CASES + generate_random_pairs(n=num_random, seed=seed)
