"""Bug 002: Causal mask reversed (upper-triangular instead of lower-triangular)."""

import numpy as np


def causal_mask_wrong(seq_len: int) -> np.ndarray:
    """BUG: np.triu allows attending to FUTURE tokens only."""
    return np.triu(np.ones((seq_len, seq_len)))


if __name__ == "__main__":
    print("Causal mask (WRONG):")
    print(causal_mask_wrong(3))
    # [[1, 1, 1],    ← token 0 can see tokens 0,1,2 (future!)
    #  [0, 1, 1],    ← token 1 can see tokens 1,2 (future!)
    #  [0, 0, 1]]
