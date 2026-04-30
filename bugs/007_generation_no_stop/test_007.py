"""Test for Bug 007: Generation without sequence length truncation."""

import numpy as np
import pytest

from tracegpt.model import TinyGPT, GPTConfig


def test_generate_exceeds_max_seq_len():
    """Model.generate() should handle sequences longer than max_seq_len."""
    config = GPTConfig(vocab_size=8, max_seq_len=4, d_model=8, n_heads=2, n_layers=1, d_ff=16)
    model = TinyGPT(config)

    # Prompt of 3 tokens, generate 5 more → total 8, but max_seq_len=4
    prompt = np.array([0, 1, 2])
    result = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    assert len(result) == 8  # 3 + 5
    assert not np.any(np.isnan(result))


def test_generate_output_longer_than_input():
    """Generated sequence should be longer than the prompt."""
    config = GPTConfig(vocab_size=8, max_seq_len=6, d_model=8, n_heads=2, n_layers=1, d_ff=16)
    model = TinyGPT(config)

    prompt = np.array([1, 2])
    result = model.generate(prompt, max_new_tokens=3, temperature=0.0)
    assert len(result) == 5  # 2 + 3
    # First tokens should match prompt
    np.testing.assert_array_equal(result[:2], prompt)


def test_generate_greedy_reproducible():
    """Greedy generation (temperature=0) should be deterministic."""
    config = GPTConfig(vocab_size=8, max_seq_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=16)
    model = TinyGPT(config)

    prompt = np.array([1, 2, 3])
    gen1 = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    gen2 = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    np.testing.assert_array_equal(gen1, gen2)


def test_wrong_no_truncation_crashes():
    """Without truncation, exceeding max_seq_len should fail."""
    class FakeModel:
        def forward(self, ids):
            if len(ids) > 4:
                raise IndexError("seq_len > max_seq_len")
            return np.zeros((len(ids), 8))

    model = FakeModel()
    generated = [0, 1, 2]

    # Without truncation, step 3+ will exceed max_seq_len=4
    for _ in range(2):
        input_ids = np.array(generated)  # no truncation
        _ = model.forward(input_ids)
        generated.append(0)

    # Now generated has 5 elements
    with pytest.raises(IndexError):
        input_ids = np.array(generated)  # no truncation, len=5 > 4
        model.forward(input_ids)
