"""Bug 003 Fix: Correct scaling by 1/sqrt(d_k)."""

import numpy as np


def attention_correct(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """CORRECT: Scale by 1/sqrt(d_k)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # CORRECT: scaled
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V


if __name__ == "__main__":
    Q = np.array([[1.0, 2.0, 3.0]])
    K = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = attention_correct(Q, K, V)
    print("Attention with 1/sqrt(d_k) scaling (CORRECT):")
    d_k = Q.shape[-1]
    print(f"  scores = {(Q @ K.T / np.sqrt(d_k))[0]}")
    print(f"  output = {result}")
