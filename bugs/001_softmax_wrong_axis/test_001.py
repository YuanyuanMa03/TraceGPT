"""Test for Bug 001: Softmax on wrong axis."""

import numpy as np
import pytest

from tracegpt.ops import softmax


def test_softmax_rows_sum_to_one():
    """Each row of softmax output should sum to 1.0."""
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.0, 1.0]])
    result = softmax(logits, axis=-1)
    row_sums = result.sum(axis=-1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-6)


def test_softmax_wrong_axis_fails():
    """Softmax with axis=0 should NOT make rows sum to 1."""
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.0, 1.0]])
    x_shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp_x = np.exp(x_shifted)
    wrong_result = exp_x / np.sum(exp_x, axis=0, keepdims=True)
    row_sums = wrong_result.sum(axis=1)
    # These should NOT both be 1.0
    assert not np.allclose(row_sums, [1.0, 1.0], atol=0.01), \
        "Bug 001 not reproduced: axis=0 accidentally gives correct row sums"


def test_softmax_correct_values():
    """Verify softmax produces correct probabilities for known input."""
    logits = np.array([1.0, 2.0, 3.0])
    result = softmax(logits)
    # The largest logit should have the highest probability
    assert result[2] > result[1] > result[0]
    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)
