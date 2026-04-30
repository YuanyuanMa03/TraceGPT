"""Bug 005 Fix: Correct label shifting for next-token prediction."""

import numpy as np


def prepare_data_correct(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CORRECT: labels are shifted by one position."""
    return tokens[:-1], tokens[1:]


if __name__ == "__main__":
    tokens = np.array([10, 20, 30, 40])  # "I", "love", "AI", "<EOS>"

    x, y = prepare_data_correct(tokens)
    print(f"Input:  {x}")   # [10, 20, 30]
    print(f"Labels: {y}")   # [20, 30, 40]
    print("CORRECT: labels[i] is the next token after input[i]")
