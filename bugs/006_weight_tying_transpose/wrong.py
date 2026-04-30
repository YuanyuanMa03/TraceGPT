"""Bug 006: Wrong transpose in weight tying — logits = E @ X instead of X @ E.T"""

import numpy as np


def output_projection_wrong(X: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    """BUG: E @ X transposes the wrong way."""
    return embedding @ X.T  # BUG: shape will be (vocab, seq) instead of (seq, vocab)


if __name__ == "__main__":
    X = np.random.randn(3, 4)      # (seq_len=3, d_model=4)
    E = np.random.randn(8, 4)      # (vocab_size=8, d_model=4)

    print(f"X shape: {X.shape}")
    print(f"E shape: {E.shape}")

    wrong = output_projection_wrong(X, E)
    print(f"Wrong (E @ X.T) shape: {wrong.shape}")   # (8, 3) — wrong!

    correct = X @ E.T
    print(f"Correct (X @ E.T) shape: {correct.shape}")  # (3, 8) — correct!
