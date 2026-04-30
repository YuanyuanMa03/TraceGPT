"""Bug 004 Fix: Q @ K^T (correct order)."""

import numpy as np


def attention_scores_correct(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """CORRECT: Q @ K^T gives proper attention scores."""
    return Q @ K.T


if __name__ == "__main__":
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [1.0, 0.0]])

    result = attention_scores_correct(Q, K)
    print(f"Correct (Q @ K^T):\n{result}")
