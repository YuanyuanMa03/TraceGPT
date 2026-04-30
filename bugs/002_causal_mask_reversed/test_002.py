"""Test for Bug 002: Causal mask reversed."""

import numpy as np
import pytest

from tracegpt.ops import causal_mask


def test_causal_mask_is_lower_triangular():
    """Causal mask must be lower-triangular."""
    mask = causal_mask(4)
    expected = np.tril(np.ones((4, 4)))
    np.testing.assert_array_equal(mask, expected)


def test_causal_mask_first_row():
    """First position can only attend to itself."""
    mask = causal_mask(3)
    assert mask[0, 0] == 1.0
    assert mask[0, 1] == 0.0
    assert mask[0, 2] == 0.0


def test_causal_mask_diagonal_is_one():
    """Diagonal should be all 1s (self-attention is always allowed)."""
    mask = causal_mask(5)
    np.testing.assert_array_equal(np.diag(mask), np.ones(5))


def test_reversed_mask_detected():
    """Upper-triangular mask should fail the lower-triangular check."""
    wrong_mask = np.triu(np.ones((3, 3)))
    assert not np.array_equal(wrong_mask, np.tril(np.ones((3, 3))))
