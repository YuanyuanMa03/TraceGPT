"""Bug 003: Missing sqrt(d_k) scaling in attention scores."""

import numpy as np


def attention_wrong(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """BUG: No scaling by 1/sqrt(d_k)."""
    scores = Q @ K.T  # BUG: missing / sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V


if __name__ == "__main__":
    Q = np.array([[1.0, 2.0, 3.0]])
    K = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = attention_wrong(Q, K, V)
    print("Attention without scaling (WRONG):")
    print(f"  scores = {(Q @ K.T)[0]}")  # Very large!
    print(f"  output = {result}")
