"""
Bug 001 Fix: Softmax on correct axis (-1).

CORRECT: axis=-1 normalizes across the last dimension (vocabulary logits).
Each row sums to 1.0 as expected.
"""

import numpy as np


def softmax_correct(x: np.ndarray) -> np.ndarray:
    """CORRECT: axis=-1 normalizes each row independently."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.0, 1.0],
    ])
    result = softmax_correct(logits)
    print("Softmax (CORRECT, axis=-1):")
    print(result)
    print(f"Row sums: {result.sum(axis=1)}")  # [1.0, 1.0] ✓
