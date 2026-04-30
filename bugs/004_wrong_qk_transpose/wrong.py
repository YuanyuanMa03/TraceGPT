"""Bug 004: K @ Q^T instead of Q @ K^T."""

import numpy as np


def attention_scores_wrong(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """BUG: K @ Q^T gives transposed attention pattern."""
    return K @ Q.T  # BUG: should be Q @ K.T


if __name__ == "__main__":
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [1.0, 0.0]])

    wrong = attention_scores_wrong(Q, K)
    correct = Q @ K.T

    print(f"Wrong (K @ Q^T):\n{wrong}")
    print(f"Correct (Q @ K^T):\n{correct}")
    print(f"Are they transposes? {np.allclose(wrong, correct.T)}")
