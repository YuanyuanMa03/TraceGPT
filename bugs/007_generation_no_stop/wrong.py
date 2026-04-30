"""Bug 007: Generation without sequence length truncation."""

import numpy as np


def generate_wrong(token_ids: list, model, max_new_tokens: int = 10) -> list:
    """BUG: Never truncates input, will exceed max_seq_len."""
    generated = list(token_ids)
    for _ in range(max_new_tokens):
        input_ids = np.array(generated)  # BUG: no truncation!
        # This will crash or produce garbage when len(generated) > max_seq_len
        logits = model.forward(input_ids)
        next_token = int(np.argmax(logits[-1]))
        generated.append(next_token)
    return generated


if __name__ == "__main__":
    # Simulated: PE only has 4 positions
    class FakeModel:
        class Config:
            max_seq_len = 4
        def forward(self, ids):
            if len(ids) > 4:
                raise IndexError(f"Sequence length {len(ids)} > max_seq_len 4!")
            return np.zeros((len(ids), 10))

    model = FakeModel()
    try:
        result = generate_wrong([0, 1, 2], model, max_new_tokens=5)
        print(f"Generated: {result}")
    except IndexError as e:
        print(f"BUG CAUGHT: {e}")
