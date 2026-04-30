# Bug 001: Softmax on Wrong Axis

## The Bug

When computing softmax for a batch of sequences, applying softmax along the wrong axis
produces incorrect probability distributions.

### What happens

Instead of normalizing each row (each token's logits) independently, the bug normalizes
each column, producing probabilities that don't sum to 1 along the intended dimension.

### How to detect

- Each row of the softmax output should sum to 1.0.
- If rows don't sum to 1, the axis is likely wrong.

### The Fix

Always specify `axis=-1` (or the correct last dimension) when computing softmax over
vocabulary logits or attention scores.

## Files

- `wrong.py` — Buggy implementation (softmax along axis=0)
- `correct.py` — Fixed implementation (softmax along axis=-1)
- `test_001.py` — Test that catches the bug
