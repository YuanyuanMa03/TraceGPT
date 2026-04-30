"""
Tests for v0.2 features: positional encoding, multi-head attention.
"""

import numpy as np
import pytest

from tracegpt.ops import (
    softmax,
    causal_mask,
    sinusoidal_position_encoding,
    multi_head_attention,
    scaled_dot_product_attention,
)


class TestSinusoidalPositionEncoding:
    def test_shape(self):
        pe = sinusoidal_position_encoding(10, 8)
        assert pe.shape == (10, 8)

    def test_even_d_model_required(self):
        with pytest.raises(AssertionError):
            sinusoidal_position_encoding(5, 3)  # odd d_model

    def test_pos_zero_even_dims_are_zero_sin(self):
        """At position 0, sin(0) = 0 for all even dimensions."""
        pe = sinusoidal_position_encoding(5, 4)
        # PE[0, 0] = sin(0) = 0, PE[0, 2] = sin(0) = 0
        assert abs(pe[0, 0]) < 1e-6
        assert abs(pe[0, 2]) < 1e-6

    def test_pos_zero_odd_dims_are_one(self):
        """At position 0, cos(0) = 1 for all odd dimensions."""
        pe = sinusoidal_position_encoding(5, 4)
        # PE[0, 1] = cos(0) = 1, PE[0, 3] = cos(0) = 1
        np.testing.assert_allclose(pe[0, 1], 1.0, atol=1e-6)
        np.testing.assert_allclose(pe[0, 3], 1.0, atol=1e-6)

    def test_positions_are_unique(self):
        pe = sinusoidal_position_encoding(10, 8)
        for i in range(10):
            for j in range(i + 1, 10):
                assert not np.allclose(pe[i], pe[j]), f"pos {i} ≈ pos {j}"

    def test_reproducible(self):
        pe1 = sinusoidal_position_encoding(5, 4)
        pe2 = sinusoidal_position_encoding(5, 4)
        np.testing.assert_array_equal(pe1, pe2)

    def test_single_position(self):
        pe = sinusoidal_position_encoding(1, 6)
        assert pe.shape == (1, 6)

    def test_values_bounded(self):
        """All PE values should be in [-1, 1]."""
        pe = sinusoidal_position_encoding(100, 64)
        assert np.all(pe >= -1.0 - 1e-6)
        assert np.all(pe <= 1.0 + 1e-6)


class TestMultiHeadAttention:
    def test_output_shape(self):
        seq_len, d_model, n_heads = 4, 8, 2
        rng = np.random.RandomState(42)
        X = rng.randn(seq_len, d_model)
        W_Q = rng.randn(d_model, d_model) * 0.1
        W_K = rng.randn(d_model, d_model) * 0.1
        W_V = rng.randn(d_model, d_model) * 0.1
        W_O = rng.randn(d_model, d_model) * 0.1

        output = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads)
        assert output.shape == (seq_len, d_model)

    def test_with_causal_mask(self):
        seq_len, d_model, n_heads = 3, 4, 2
        rng = np.random.RandomState(42)
        X = rng.randn(seq_len, d_model)
        W_Q = np.eye(d_model)
        W_K = np.eye(d_model)
        W_V = np.eye(d_model)
        W_O = np.eye(d_model)

        mask = causal_mask(seq_len)
        output = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask)
        assert output.shape == (seq_len, d_model)
        assert not np.any(np.isnan(output))

    def test_single_head_matches_sdpa(self):
        """With 1 head, multi_head_attention should match scaled_dot_product_attention."""
        seq_len, d_model, n_heads = 3, 4, 1
        rng = np.random.RandomState(99)
        X = rng.randn(seq_len, d_model)
        W_Q = np.eye(d_model) * 0.5
        W_K = np.eye(d_model) * 0.5
        W_V = np.eye(d_model) * 0.5
        W_O = np.eye(d_model)

        mask = causal_mask(seq_len)

        # Multi-head (1 head)
        mha_output = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask)

        # Single-head (sdpa)
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        sdpa_output = scaled_dot_product_attention(Q, K, V, mask) @ W_O

        np.testing.assert_allclose(mha_output, sdpa_output, atol=1e-5)

    def test_heads_not_divisible_raises(self):
        seq_len, d_model, n_heads = 3, 5, 2  # 5 not divisible by 2
        X = np.ones((seq_len, d_model))
        with pytest.raises(AssertionError):
            multi_head_attention(X, np.eye(d_model), np.eye(d_model),
                                 np.eye(d_model), np.eye(d_model), n_heads)

    def test_no_nan_output(self):
        seq_len, d_model, n_heads = 4, 8, 4
        rng = np.random.RandomState(7)
        X = rng.randn(seq_len, d_model)
        W_Q = rng.randn(d_model, d_model) * 0.1
        W_K = rng.randn(d_model, d_model) * 0.1
        W_V = rng.randn(d_model, d_model) * 0.1
        W_O = rng.randn(d_model, d_model) * 0.1
        mask = causal_mask(seq_len)

        output = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask)
        assert not np.any(np.isnan(output))
