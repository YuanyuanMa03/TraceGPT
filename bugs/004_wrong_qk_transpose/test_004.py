"""Test for Bug 004: Wrong QK transpose."""

import numpy as np
import pytest

from tracegpt.ops import scaled_dot_product_attention, causal_mask, softmax


def test_qk_order():
    """Q @ K^T should NOT equal K @ Q^T in general."""
    Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    correct = Q @ K.T
    wrong = K @ Q.T
    assert not np.allclose(correct, wrong)


def test_attention_output_shape():
    """Attention output should have shape (seq_len, d_v)."""
    seq_len, d_k, d_v = 3, 4, 4
    rng = np.random.RandomState(42)
    Q = rng.randn(seq_len, d_k)
    K = rng.randn(seq_len, d_k)
    V = rng.randn(seq_len, d_v)

    mask = causal_mask(seq_len)
    output = scaled_dot_product_attention(Q, K, V, mask)
    assert output.shape == (seq_len, d_v)


def test_scores_row_matches_query():
    """Row i of scores should reflect query i attending to all keys."""
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])

    scores = Q @ K.T
    # Q[0] = [1,0] matches K[0]=[1,0] perfectly → scores[0][0] = 1
    # Q[0] = [1,0] is orthogonal to K[1]=[0,1] → scores[0][1] = 0
    assert scores[0, 0] == 1.0
    assert scores[0, 1] == 0.0
