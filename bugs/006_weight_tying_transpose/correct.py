"""Bug 006 Fix: Correct weight tying — logits = X @ E.T"""

import numpy as np


def output_projection_correct(X: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    """CORRECT: X @ E.T projects from d_model to vocab_size."""
    return X @ embedding.T


if __name__ == "__main__":
    X = np.random.randn(3, 4)
    E = np.random.randn(8, 4)

    result = output_projection_correct(X, E)
    print(f"Correct (X @ E.T) shape: {result.shape}")  # (3, 8)
