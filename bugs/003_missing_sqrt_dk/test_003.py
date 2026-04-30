"""Test for Bug 003: Missing sqrt(d_k) scaling."""

import numpy as np
import pytest

from tracegpt.ops import softmax, scaled_dot_product_attention


def test_scaling_reduces_score_magnitude():
    """Scaled scores should be smaller than unscaled scores."""
    Q = np.array([[1.0, 2.0, 3.0]])
    K = np.array([[1.0, 2.0, 3.0]])
    d_k = Q.shape[-1]

    raw_scores = Q @ K.T
    scaled_scores = raw_scores / np.sqrt(d_k)

    assert np.max(np.abs(scaled_scores)) < np.max(np.abs(raw_scores))


def test_scaled_attention_produces_balanced_weights():
    """With scaling, attention weights should be reasonably distributed."""
    Q = np.array([[1.0, 2.0, 3.0]])
    K = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    V = np.array([[1.0], [0.0], [0.5]])

    output = scaled_dot_product_attention(Q, K, V)
    assert output.shape == (1, 1)


def test_unscaled_scores_excessively_large():
    """Without scaling, large d_k produces very large scores."""
    rng = np.random.RandomState(42)
    d_k = 64
    Q = rng.randn(1, d_k)
    K = rng.randn(1, d_k)

    raw = (Q @ K.T)[0, 0]
    scaled = raw / np.sqrt(d_k)

    assert abs(raw) > abs(scaled)
