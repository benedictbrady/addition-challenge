"""Anti-cheat validation for submissions.

Layered defense:
1. Interface check — all exports present, types correct, bounds valid
2. Structural attention check — model contains self-attention layers
   AND they are actually invoked during forward (not decorative)
3. Behavioral causal check — prefix consistency proves causal masking
4. Encode bounds check — output length and token range
5. Encode consistency check — catches carry-steganography
6. Decode honesty test — catches side-channel cheating
7. AST analysis — catches global state mutation in encode/decode
8. Parameter influence check — registered params must affect output
   (catches models that compute entirely from hardcoded constants)
"""

import ast
import inspect
import random
import textwrap
from dataclasses import dataclass

import torch
import torch.nn as nn

from .loader import Submission

MAX_ENCODE_TOKENS = 35
CAUSAL_CHECK_TOLERANCE = 1e-3


@dataclass
class ValidationResult:
    passed: bool
    check_name: str
    message: str


def validate_submission(submission: Submission, verbose: bool = True) -> list[ValidationResult]:
    """Run all validation checks on a submission. Returns list of results."""
    results: list[ValidationResult] = []

    results.append(_check_interface(submission))
    results.append(_check_structural_attention(submission))
    results.append(_check_causal_behavior(submission))
    results.extend(_check_encode_bounds(submission))
    results.append(_check_encode_consistency(submission))
    results.append(_check_decode_honesty(submission))
    results.extend(_check_ast_safety(submission))
    results.append(_check_parameter_influence(submission))

    return results


def _check_interface(sub: Submission) -> ValidationResult:
    """Check VOCAB_SIZE and MAX_OUTPUT_LEN bounds."""
    if sub.vocab_size > 256:
        return ValidationResult(False, "Interface bounds", f"VOCAB_SIZE={sub.vocab_size} exceeds max 256")
    if sub.vocab_size < 2:
        return ValidationResult(False, "Interface bounds", f"VOCAB_SIZE={sub.vocab_size} must be >= 2")
    if sub.max_output_len > 30:
        return ValidationResult(False, "Interface bounds", f"MAX_OUTPUT_LEN={sub.max_output_len} exceeds max 30")
    if sub.max_output_len < 1:
        return ValidationResult(False, "Interface bounds", f"MAX_OUTPUT_LEN={sub.max_output_len} must be >= 1")
    return ValidationResult(True, "Interface bounds", f"VOCAB_SIZE={sub.vocab_size}, MAX_OUTPUT_LEN={sub.max_output_len}")


def _find_attention_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find all modules that look like self-attention layers."""
    found = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            found.append((name, module))
            continue

        child_names = {n for n, _ in module.named_children()}
        param_names = {n for n, _ in module.named_parameters(recurse=False)}
        all_names = child_names | param_names

        has_qkv = (
            ("q_proj" in all_names and "k_proj" in all_names and "v_proj" in all_names)
            or ("query" in all_names and "key" in all_names and "value" in all_names)
            or ("wq" in all_names and "wk" in all_names and "wv" in all_names)
            or ("W_Q" in all_names and "W_K" in all_names and "W_V" in all_names)
            or ("to_q" in all_names and "to_k" in all_names and "to_v" in all_names)
            or ("in_proj_weight" in all_names)  # packed QKV
        )
        if has_qkv:
            found.append((name, module))
            continue

        class_name = type(module).__name__.lower()
        if "attention" in class_name and name != "":
            found.append((name, module))

    return found


def _check_structural_attention(sub: Submission) -> ValidationResult:
    """Verify model contains self-attention that is actually invoked during forward.

    Two-part check:
    1. Model has modules that look like self-attention (QKV projections, etc.)
    2. At least one such module is actually called during a forward pass
       (catches decorative/dummy attention modules with nn.Identity projections)
    """
    model = sub.model
    model.eval()

    attention_modules = _find_attention_modules(model)
    if not attention_modules:
        return ValidationResult(False, "Self-attention layer found", "No self-attention layer detected in model")

    # Verify at least one attention module is actually called during forward
    called: set[str] = set()
    hooks = []
    for name, module in attention_modules:
        def _make_hook(n: str):
            def hook(mod, inp, out):
                called.add(n)
            return hook
        hooks.append(module.register_forward_hook(_make_hook(name)))

    try:
        test_input = sub.encode(12345, 67890)
        with torch.no_grad():
            model(torch.tensor([test_input], dtype=torch.long))
    finally:
        for h in hooks:
            h.remove()

    if not called:
        names = [n for n, _ in attention_modules]
        return ValidationResult(
            False,
            "Self-attention layer found",
            f"Attention modules found ({names}) but none invoked during forward — modules may be decorative",
        )

    called_name = next(iter(called))
    return ValidationResult(True, "Self-attention layer found", f"Attention module at '{called_name}' invoked during forward")


def _check_causal_behavior(sub: Submission) -> ValidationResult:
    """Behavioral causal mask test.

    Give model [A,B,C,D] vs [A,B,C,X] and verify logits at positions 0-2
    are identical. This proves the model uses causal masking.
    """
    model = sub.model
    model.eval()
    vocab_size = sub.vocab_size

    rng = random.Random(42)
    seq_len = min(8, 25)  # short sequence for test
    base_tokens = [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
    alt_tokens = base_tokens.copy()
    # Change the last token
    alt_tokens[-1] = (alt_tokens[-1] + 1) % vocab_size

    with torch.no_grad():
        input_a = torch.tensor([base_tokens], dtype=torch.long)
        input_b = torch.tensor([alt_tokens], dtype=torch.long)
        logits_a = model(input_a)  # (1, seq_len, vocab_size)
        logits_b = model(input_b)

    # Check that logits at positions 0 through seq_len-2 are identical
    # (only the last position should differ since only last token changed)
    shared_logits_a = logits_a[0, :-1]  # (seq_len-1, vocab_size)
    shared_logits_b = logits_b[0, :-1]

    max_diff = (shared_logits_a - shared_logits_b).abs().max().item()
    if max_diff > CAUSAL_CHECK_TOLERANCE:
        return ValidationResult(
            False,
            "Causal behavior verified",
            f"Logits at shared positions differ by {max_diff:.6f} — model may not use causal masking",
        )

    return ValidationResult(True, "Causal behavior verified", f"Max logit diff at shared positions: {max_diff:.2e}")


def _check_encode_bounds(sub: Submission) -> list[ValidationResult]:
    """Check encode output length and token range on sample inputs."""
    results = []
    test_pairs = [(0, 0), (9999999999, 9999999999), (12345, 67890), (1, 9999999999)]

    for a, b in test_pairs:
        tokens = sub.encode(a, b)
        if not isinstance(tokens, list):
            results.append(ValidationResult(False, "Encode bounds valid", f"encode({a}, {b}) returned {type(tokens)}, expected list"))
            return results

        if len(tokens) > MAX_ENCODE_TOKENS:
            results.append(ValidationResult(False, "Encode bounds valid", f"encode({a}, {b}) returned {len(tokens)} tokens, max is {MAX_ENCODE_TOKENS}"))
            return results

        out_of_range = [t for t in tokens if t < 0 or t >= sub.vocab_size]
        if out_of_range:
            results.append(ValidationResult(False, "Encode bounds valid", f"encode({a}, {b}) has tokens outside [0, {sub.vocab_size}): {out_of_range}"))
            return results

    results.append(ValidationResult(True, "Encode bounds valid", f"All sample encodes within bounds (≤{MAX_ENCODE_TOKENS} tokens, range [0, {sub.vocab_size})"))
    return results


def _check_encode_consistency(sub: Submission) -> ValidationResult:
    """Check that encode doesn't smuggle carry/answer information.

    Tests that encode is deterministic and that changing inputs only affects
    expected positions. Paired-token schemes (where a and b share positions)
    are allowed — the key check is that encode(a, b) doesn't vary with the
    *sum* a+b when a and b are individually held constant.
    """
    rng = random.Random(123)

    # Test 1: determinism — same inputs produce same output
    a = rng.randint(0, 9999999999)
    b = rng.randint(0, 9999999999)
    tokens_a = sub.encode(a, b)
    tokens_b = sub.encode(a, b)
    if tokens_a != tokens_b:
        return ValidationResult(False, "Encode consistency", "encode is non-deterministic")

    # Test 2: changing b should change *something*
    b1 = rng.randint(0, 9999999999)
    b2 = rng.randint(0, 9999999999)
    while b1 == b2:
        b2 = rng.randint(0, 9999999999)
    tokens1 = sub.encode(a, b1)
    tokens2 = sub.encode(a, b2)

    if tokens1 == tokens2:
        return ValidationResult(False, "Encode consistency", "encode returns identical tokens for different b values — suspicious")

    # Test 3: anti-steganography — encode should not embed the sum.
    # Use pairs where a+b1 == a+b2 but b1 != b2 (impossible),
    # so instead check that encode(a1, b) and encode(a2, b) differ only
    # in ways consistent with a changing (not smuggling a+b).
    # The real check: encode(a, b) should depend only on a and b individually,
    # not on their sum. Test with two pairs that have the same sum:
    # (a1, b1) and (a2, b2) where a1+b1 == a2+b2 but a1 != a2.
    a1 = rng.randint(1, 9999999998)
    b1 = rng.randint(1, 9999999998)
    total = a1 + b1
    a2 = rng.randint(1, min(total - 1, 9999999999))
    while a2 == a1:
        a2 = rng.randint(1, min(total - 1, 9999999999))
    b2 = total - a2
    if 0 <= b2 <= 9999999999 and (a1, b1) != (a2, b2):
        t1 = sub.encode(a1, b1)
        t2 = sub.encode(a2, b2)
        if t1 == t2:
            return ValidationResult(
                False,
                "Encode consistency",
                "encode produces identical tokens for different (a,b) pairs with the same sum — smuggling answer",
            )

    return ValidationResult(True, "Encode consistency", "Encode changes are consistent")


def _check_decode_honesty(sub: Submission) -> ValidationResult:
    """Feed random garbage tokens to decode — if it returns valid answers, it's cheating.

    If decode ignores its input and uses a side channel (global state from encode),
    it will still produce correct answers for garbage input.
    """
    rng = random.Random(999)

    # First, run a few encodes to potentially populate any side channel
    test_pairs = [(123, 456), (9999999999, 1), (5555, 4444)]
    expected_sums = [a + b for a, b in test_pairs]

    for a, b in test_pairs:
        sub.encode(a, b)

    # Now feed random garbage to decode
    garbage_results = []
    for _ in range(10):
        garbage = [rng.randint(0, sub.vocab_size - 1) for _ in range(rng.randint(1, sub.max_output_len))]
        try:
            result = sub.decode(garbage)
            if result is not None and result in expected_sums:
                garbage_results.append(result)
        except Exception:
            pass  # decode failing on garbage is fine

    if len(garbage_results) >= 2:
        return ValidationResult(
            False,
            "Decode honesty",
            f"decode returned valid answers {garbage_results} for random garbage — likely using side channel",
        )

    return ValidationResult(True, "Decode honesty", "decode does not produce suspicious results for garbage input")


def _check_ast_safety(sub: Submission) -> list[ValidationResult]:
    """Scan source of encode and decode for dangerous patterns."""
    results = []
    for func_name in ["encode", "decode"]:
        func = getattr(sub.module, func_name)
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError) as e:
            results.append(ValidationResult(False, f"AST safety ({func_name})", f"Cannot parse source: {e}"))
            continue

        violations = []

        for node in ast.walk(tree):
            # Check for global/nonlocal statements
            if isinstance(node, ast.Global):
                violations.append(f"'global' statement for {node.names}")
            if isinstance(node, ast.Nonlocal):
                violations.append(f"'nonlocal' statement for {node.names}")

            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("os", "sys", "subprocess", "socket", "ctypes", "pickle"):
                        violations.append(f"import of '{alias.name}'")
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in ("os", "sys", "subprocess", "socket", "ctypes", "pickle"):
                    violations.append(f"import from '{node.module}'")

            # Check for eval/exec/__import__
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "__import__", "compile"):
                        violations.append(f"call to '{node.func.id}'")

            # Check for attribute mutation on external objects (setattr)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "setattr":
                    violations.append("call to 'setattr'")

        if violations:
            results.append(ValidationResult(False, f"AST safety ({func_name})", f"Violations: {', '.join(violations)}"))
        else:
            results.append(ValidationResult(True, f"AST safety ({func_name})", "No dangerous patterns found"))

    return results


def _check_parameter_influence(sub: Submission) -> ValidationResult:
    """Verify that registered parameters actually affect the model's output.

    Catches models that store all computation logic in hardcoded constants
    (created inside forward()) and register only decorative/unused parameters
    to satisfy the param counter.

    Approach: perturb all parameters simultaneously with noise and check
    whether the output changes. If output is identical, the parameters
    are decorative and the model computes entirely from hidden constants.
    """
    model = sub.model
    model.eval()

    params = list(model.named_parameters())
    if not params:
        return ValidationResult(True, "Parameter influence", "Model has no parameters (nothing to check)")

    test_input = sub.encode(12345, 67890)
    input_tensor = torch.tensor([test_input], dtype=torch.long)

    # Get baseline output
    with torch.no_grad():
        baseline_logits = model(input_tensor).clone()

    # Save original parameter values
    original_state = {name: p.data.clone() for name, p in params}

    # Perturb ALL parameters with significant noise
    with torch.no_grad():
        for _, p in params:
            noise = torch.randn_like(p.float()).to(p.dtype) * 0.5
            p.data.add_(noise)

    # Get perturbed output
    with torch.no_grad():
        perturbed_logits = model(input_tensor)

    # Restore original parameters
    with torch.no_grad():
        for name, p in params:
            p.data.copy_(original_state[name])

    # Check if output changed
    max_diff = (baseline_logits.float() - perturbed_logits.float()).abs().max().item()
    if max_diff < 1e-6:
        return ValidationResult(
            False,
            "Parameter influence",
            "Model output is independent of all registered parameters — "
            "parameters are decorative and real weights may be hardcoded constants",
        )

    return ValidationResult(
        True,
        "Parameter influence",
        f"Parameters influence output (max logit change: {max_diff:.2e})",
    )
