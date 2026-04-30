"""Test for Bug 006: Weight tying transpose error."""

import numpy as np
import pytest


def test_wrong_shape():
    """E @ X.T produces (vocab, seq) instead of (seq, vocab)."""
    X = np.random.randn(3, 4)
    E = np.random.randn(8, 4)

    wrong = E @ X.T
    assert wrong.shape == (8, 3), "Should be (vocab, seq) — the wrong orientation"

    correct = X @ E.T
    assert correct.shape == (3, 8), "Should be (seq, vocab) — the correct orientation"

    assert wrong.shape != correct.shape


def test_correct_produces_right_shape():
    """X @ E.T should produce (seq_len, vocab_size)."""
    seq_len, d_model, vocab_size = 5, 8, 16
    X = np.random.randn(seq_len, d_model)
    E = np.random.randn(vocab_size, d_model)

    logits = X @ E.T
    assert logits.shape == (seq_len, vocab_size)


def test_wrong_transpose_is_not_correct():
    """E @ X.T transposed equals X @ E.T, but the original is wrong."""
    X = np.random.randn(3, 4)
    E = np.random.randn(8, 4)

    wrong = E @ X.T
    correct = X @ E.T

    # The wrong result transposed should equal correct
    np.testing.assert_allclose(wrong.T, correct)
    # But wrong itself is NOT correct (different shape)
    assert wrong.shape != correct.shape
