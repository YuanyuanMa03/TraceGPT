"""Bug 005: Labels not shifted — model predicts current token, not next token."""

import numpy as np


def prepare_data_wrong(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BUG: Input and labels are the same (no shift)."""
    return tokens, tokens  # BUG: should be tokens[:-1], tokens[1:]


if __name__ == "__main__":
    tokens = np.array([10, 20, 30, 40])  # "I", "love", "AI", "<EOS>"

    x, y = prepare_data_wrong(tokens)
    print(f"Input:  {x}")
    print(f"Labels: {y}")
    print("BUG: input[i] == labels[i], so model learns to copy, not predict!")
