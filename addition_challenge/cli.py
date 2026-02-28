"""CLI entry point: addition validate|eval|count-params."""

import argparse
import sys

from .evaluator import evaluate
from .loader import load_submission
from .param_counter import count_forward_constants, count_unique_parameters
from .validator import validate_submission


def cmd_validate(args: argparse.Namespace) -> None:
    print(f"Loading submission from {args.submission}...")
    try:
        submission = load_submission(args.submission)
    except Exception as e:
        print(f"  [FAIL] Load error: {e}")
        sys.exit(1)
    print("  Loaded successfully")

    print("Validating model...")
    results = validate_submission(submission, verbose=True)

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.check_name}: {r.message}")
        if not r.passed:
            all_passed = False

    if all_passed:
        param_count = count_unique_parameters(submission.model)
        print(f"Parameters: {param_count:,} (unique registered)")

        sample_input = submission.encode(12345, 67890)
        import torch
        fwd_stats = count_forward_constants(
            submission.model,
            torch.tensor([sample_input], dtype=torch.long),
        )
        if fwd_stats.total_elements > 0:
            print(f"Forward constants: {fwd_stats.total_elements:,} elements ({fwd_stats.call_counts})")

        print("Validation: PASSED")
    else:
        print("Validation: FAILED")
        sys.exit(1)


def cmd_eval(args: argparse.Namespace) -> None:
    print(f"Loading submission from {args.submission}...")
    try:
        submission = load_submission(args.submission)
    except Exception as e:
        print(f"  [FAIL] Load error: {e}")
        sys.exit(1)
    print("  Loaded successfully")

    print("Validating model...")
    results = validate_submission(submission, verbose=True)
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.check_name}")
        if not r.passed:
            print(f"         {r.message}")
            all_passed = False

    if not all_passed:
        print("Validation FAILED — skipping evaluation")
        sys.exit(1)

    param_count = count_unique_parameters(submission.model)
    print(f"Parameters: {param_count:,} (unique trainable)")

    num_random = args.num_random
    total = len(__import__("addition_challenge.test_cases", fromlist=["EDGE_CASES"]).EDGE_CASES) + num_random
    print(f"Evaluating ({total:,} test cases)...")

    result = evaluate(
        submission,
        num_random=num_random,
        seed=args.seed,
        device=args.device,
    )

    edge_pct = 100.0 * result.edge_correct / result.edge_total if result.edge_total > 0 else 0.0
    rand_pct = 100.0 * result.random_correct / result.random_total if result.random_total > 0 else 0.0
    total_pct = 100.0 * result.accuracy
    status = "QUALIFIED" if result.qualified else "NOT QUALIFIED"

    print(f"  Edge cases:   {result.edge_correct}/{result.edge_total}   ({edge_pct:.1f}%)")
    print(f"  Random:       {result.random_correct:,}/{result.random_total:,} ({rand_pct:.2f}%)")
    print(f"  Overall:      {result.total_correct:,}/{result.total_total:,} ({total_pct:.2f}%) [{status}]")
    print(f"  Time:         {result.elapsed_seconds:.1f}s")

    if result.failures:
        print(f"  Failures (showing up to {len(result.failures)}):")
        for a, b, expected, got in result.failures:
            print(f"    {a} + {b} = {expected}, got {got}")

    print(f"Result: {result.param_count:,} params @ {total_pct:.2f}% accuracy")

    if not result.qualified:
        sys.exit(1)


def cmd_count_params(args: argparse.Namespace) -> None:
    print(f"Loading submission from {args.submission}...")
    try:
        submission = load_submission(args.submission)
    except Exception as e:
        print(f"  [FAIL] Load error: {e}")
        sys.exit(1)

    import torch

    param_count = count_unique_parameters(submission.model)
    print(f"Parameters: {param_count:,} (unique registered)")

    sample_input = submission.encode(12345, 67890)
    fwd_stats = count_forward_constants(
        submission.model,
        torch.tensor([sample_input], dtype=torch.long),
    )
    if fwd_stats.total_elements > 0:
        print(f"Forward constants: {fwd_stats.total_elements:,} elements ({fwd_stats.call_counts})")
    print(f"Total cost: {param_count + fwd_stats.total_elements:,} (registered + forward constants)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="addition",
        description="Addition Challenge — evaluate autoregressive transformer submissions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate
    p_validate = subparsers.add_parser("validate", help="Run structural and behavioral validation")
    p_validate.add_argument("submission", help="Path to submission .py file")

    # eval
    p_eval = subparsers.add_parser("eval", help="Full accuracy evaluation")
    p_eval.add_argument("submission", help="Path to submission .py file")
    p_eval.add_argument("--num-random", type=int, default=10000, help="Number of random test pairs (default: 10000)")
    p_eval.add_argument("--seed", type=int, default=2025, help="Random seed for test generation (default: 2025)")
    p_eval.add_argument("--device", default="cpu", help="Device to run on (default: cpu)")

    # count-params
    p_count = subparsers.add_parser("count-params", help="Count unique trainable parameters")
    p_count.add_argument("submission", help="Path to submission .py file")

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "count-params":
        cmd_count_params(args)


if __name__ == "__main__":
    main()
