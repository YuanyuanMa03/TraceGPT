"""Bug 007 Fix: Generation with proper truncation."""

import numpy as np


def generate_correct(token_ids: list, model, max_seq_len: int, max_new_tokens: int = 10) -> list:
    """CORRECT: Truncate input to max_seq_len at each step."""
    generated = list(token_ids)
    for _ in range(max_new_tokens):
        input_ids = np.array(generated[-max_seq_len:])  # CORRECT: truncate!
        logits = model.forward(input_ids)
        next_token = int(np.argmax(logits[-1]))
        generated.append(next_token)
    return generated


if __name__ == "__main__":
    class FakeModel:
        def forward(self, ids):
            if len(ids) > 4:
                raise IndexError(f"Sequence too long: {len(ids)}")
            return np.zeros((len(ids), 10))

    model = FakeModel()
    result = generate_correct([0, 1, 2], model, max_seq_len=4, max_new_tokens=5)
    print(f"Generated (truncated): {result}")
    print(f"Length: {len(result)} — no crash because input is always ≤ 4")
