"""Bug 002 Fix: Correct causal mask (lower-triangular)."""

import numpy as np


def causal_mask_correct(seq_len: int) -> np.ndarray:
    """CORRECT: np.tril allows attending to CURRENT and PAST tokens only."""
    return np.tril(np.ones((seq_len, seq_len)))


if __name__ == "__main__":
    print("Causal mask (CORRECT):")
    print(causal_mask_correct(3))
    # [[1, 0, 0],    ← token 0 sees only itself
    #  [1, 1, 0],    ← token 1 sees tokens 0,1
    #  [1, 1, 1]]
