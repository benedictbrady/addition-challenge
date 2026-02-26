"""Self-contained training script for the starter model.

Generates random addition pairs on the fly — no dataset files needed.
Trains with teacher forcing, cross-entropy loss on output positions only.
Uses zero-padded fixed-length format for positional alignment.

Usage:
    uv run python starter/train.py
    uv run python starter/train.py --steps 20000 --d-model 64 --batch-size 256
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from starter.model import AdditionTransformer

# Token constants (must match submission.py)
TOKEN_PLUS = 10
TOKEN_EQ = 11
TOKEN_EOS = 12
VOCAB_SIZE = 13

NUM_DIGITS = 10      # zero-pad inputs to this width
RESULT_DIGITS = 11   # max digits in sum of two 10-digit numbers

# Fixed sequence length: 10 + 1 + 10 + 1 + 11 + 1 = 34
INPUT_LEN = NUM_DIGITS + 1 + NUM_DIGITS + 1  # "0000000001+0000000002=" = 22
OUTPUT_LEN = RESULT_DIGITS + 1                # reversed digits + EOS = 12
SEQ_LEN = INPUT_LEN + OUTPUT_LEN             # 34


def encode_pair(a: int, b: int) -> tuple[list[int], list[int]]:
    """Encode an addition pair into fixed-length input and output tokens.

    Input:  zero-padded digits + PLUS + zero-padded digits + EQ (22 tokens)
    Output: reversed digits of sum, zero-padded to 11 digits + EOS (12 tokens)
    """
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    input_tokens = [int(d) for d in a_str] + [TOKEN_PLUS] + [int(d) for d in b_str] + [TOKEN_EQ]

    result = a + b
    result_str = str(result).zfill(RESULT_DIGITS)
    output_tokens = [int(d) for d in reversed(result_str)] + [TOKEN_EOS]

    return input_tokens, output_tokens


def make_batch(batch_size: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of random addition problems with fixed-length format."""
    max_val = 10**NUM_DIGITS - 1

    all_input_ids = []
    all_targets = []

    for _ in range(batch_size):
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)
        inp, out = encode_pair(a, b)

        full_seq = inp + out
        target = [-100] * INPUT_LEN + out

        all_input_ids.append(full_seq)
        all_targets.append(target)

    return (
        torch.tensor(all_input_ids, dtype=torch.long),
        torch.tensor(all_targets, dtype=torch.long),
    )


def evaluate_accuracy(model, num_samples: int = 500, device: str = "cpu") -> float:
    """Evaluate model accuracy using autoregressive generation."""
    model.eval()
    rng = random.Random(12345)
    max_val = 10**NUM_DIGITS - 1
    correct = 0

    with torch.no_grad():
        for _ in range(num_samples):
            a = rng.randint(0, max_val)
            b = rng.randint(0, max_val)
            expected = a + b

            a_str = str(a).zfill(NUM_DIGITS)
            b_str = str(b).zfill(NUM_DIGITS)
            input_tokens = [int(d) for d in a_str] + [TOKEN_PLUS] + [int(d) for d in b_str] + [TOKEN_EQ]

            context = torch.tensor([input_tokens], dtype=torch.long, device=device)
            generated = []
            for _ in range(OUTPUT_LEN):
                logits = model(context)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)
                if next_token == TOKEN_EOS:
                    break
                context = torch.cat(
                    [context, torch.tensor([[next_token]], dtype=torch.long, device=device)],
                    dim=1,
                )

            # Decode reversed digits
            digits = []
            for t in generated:
                if t == TOKEN_EOS or t < 0 or t > 9:
                    break
                digits.append(t)
            if digits:
                result = int("".join(str(d) for d in reversed(digits)))
            else:
                result = -1

            if result == expected:
                correct += 1

    model.train()
    return correct / num_samples


def main():
    parser = argparse.ArgumentParser(description="Train the starter addition model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Peak learning rate")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=128, help="Feedforward dimension")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--checkpoint", type=str, default="starter/checkpoint.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = args.device
    print(f"Training on {device}")

    model = AdditionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=SEQ_LEN,
        weight_tying=True,
    ).to(device)

    seen = set()
    unique_params = 0
    for p in model.parameters():
        if p.requires_grad and p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique_params += p.numel()
    print(f"Model parameters: {unique_params:,} (unique trainable)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    warmup_steps = 400
    total_steps = args.steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    rng = random.Random(42)
    best_accuracy = 0.0

    print(f"Training for {total_steps} steps")
    print(f"Fixed format: {INPUT_LEN} input tokens + {OUTPUT_LEN} output tokens = {SEQ_LEN} total")

    epoch_loss = 0.0
    log_interval = 500
    eval_interval = 2000
    start_time = time.time()

    for step in range(1, total_steps + 1):
        input_ids, targets = make_batch(args.batch_size, rng=rng)
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits = model(input_ids[:, :-1])
        target_shifted = targets[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            target_shifted.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        if step % log_interval == 0:
            avg_loss = epoch_loss / log_interval
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | Time: {elapsed:.0f}s"
            )
            epoch_loss = 0.0

        if step % eval_interval == 0 or step == total_steps:
            accuracy = evaluate_accuracy(model, num_samples=500, device=device)
            print(f"  --> Eval accuracy: {accuracy:.1%}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                ckpt_path = Path(args.checkpoint)
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                print(f"  --> Saved checkpoint ({accuracy:.1%})")

    print(f"\nTraining complete. Best accuracy: {best_accuracy:.1%}")
    print(f"Checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
