"""
Bug 001: Softmax on wrong axis.

BUG: axis=0 normalizes across the batch dimension, not across vocabulary logits.
Each row should sum to 1, but with axis=0, each COLUMN sums to 1 instead.
"""

import numpy as np


def softmax_wrong(x: np.ndarray) -> np.ndarray:
    """BUG: axis=0 is wrong for batched logits."""
    x_shifted = x - np.max(x, axis=0, keepdims=True)  # BUG: axis=0
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)  # BUG: axis=0


if __name__ == "__main__":
    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.0, 1.0],
    ])
    result = softmax_wrong(logits)
    print("Softmax (WRONG, axis=0):")
    print(result)
    print(f"Row sums: {result.sum(axis=1)}")  # Won't be [1, 1]!
    print(f"Col sums: {result.sum(axis=0)}")  # These will be [1, 1] instead
