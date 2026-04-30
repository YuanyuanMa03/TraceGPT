"""
Tests for TraceGPT core operations and tracer.

Run: pytest tests/ -v
"""

import numpy as np
import pytest

from tracegpt.ops import softmax, causal_mask, layer_norm, linear, relu, gelu, scaled_dot_product_attention
from tracegpt.tracer import Tracer
from tracegpt.utils import assert_close, tiny_matrix, tiny_vector, one_hot, format_array


# ============================================================
# ops.py tests
# ============================================================


class TestSoftmax:
    def test_sums_to_one_1d(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert_close(result.sum(), np.array(1.0), label="softmax sum 1d")

    def test_sums_to_one_2d_rows(self):
        x = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
        result = softmax(x)
        row_sums = result.sum(axis=-1)
        assert_close(row_sums, np.array([1.0, 1.0]), label="softmax row sums")

    def test_all_positive(self):
        x = np.array([-5.0, 0.0, 5.0])
        result = softmax(x)
        assert np.all(result >= 0)

    def test_monotonicity(self):
        """Larger inputs should produce larger probabilities."""
        x = np.array([0.0, 1.0, 2.0])
        result = softmax(x)
        assert result[2] > result[1] > result[0]

    def test_numerical_stability(self):
        """Large values should not cause overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert not np.any(np.isnan(result))
        assert_close(result.sum(), np.array(1.0), label="softmax stability")


class TestCausalMask:
    def test_shape(self):
        mask = causal_mask(4)
        assert mask.shape == (4, 4)

    def test_lower_triangular(self):
        mask = causal_mask(3)
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float)
        np.testing.assert_array_equal(mask, expected)

    def test_diagonal_ones(self):
        mask = causal_mask(5)
        np.testing.assert_array_equal(np.diag(mask), np.ones(5))

    def test_upper_triangle_zero(self):
        mask = causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j] == 0.0


class TestLayerNorm:
    def test_output_mean_near_zero(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.ones(4)
        beta = np.zeros(4)
        result = layer_norm(x, gamma, beta)
        # After layer norm, mean should be ~0
        assert abs(np.mean(result)) < 1e-4

    def test_output_std_near_one(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.ones(4)
        beta = np.zeros(4)
        result = layer_norm(x, gamma, beta)
        assert abs(np.std(result) - 1.0) < 0.1

    def test_gamma_beta_scaling(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.array([2.0, 2.0, 2.0, 2.0])
        beta = np.array([1.0, 1.0, 1.0, 1.0])
        result = layer_norm(x, gamma, beta)
        # With gamma=2, beta=1, output should be scaled and shifted
        assert result.shape == x.shape

    def test_same_input_same_output(self):
        x = np.array([[3.0, 3.0, 3.0]])
        gamma = np.ones(3)
        beta = np.zeros(3)
        result = layer_norm(x, gamma, beta)
        # All-same input → normalized to 0 (before gamma/beta)
        np.testing.assert_allclose(result, np.zeros((1, 3)), atol=1e-4)


class TestLinear:
    def test_shape(self):
        x = np.array([[1.0, 2.0]])  # (1, 2)
        W = np.ones((2, 3))
        b = np.zeros(3)
        result = linear(x, W, b)
        assert result.shape == (1, 3)

    def test_identity(self):
        x = np.array([[1.0, 2.0, 3.0]])
        W = np.eye(3)
        b = np.zeros(3)
        result = linear(x, W, b)
        np.testing.assert_allclose(result, x)

    def test_bias(self):
        x = np.array([[0.0, 0.0]])
        W = np.eye(2)
        b = np.array([1.0, 2.0])
        result = linear(x, W, b)
        np.testing.assert_allclose(result, np.array([[1.0, 2.0]]))

    def test_known_computation(self):
        x = np.array([[1.0, 2.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.5, -0.5])
        result = linear(x, W, b)
        expected = np.array([[1.5, 1.5]])
        np.testing.assert_allclose(result, expected)


class TestRelu:
    def test_positive_unchanged(self):
        x = np.array([1.0, 2.0, 3.0])
        result = relu(x)
        np.testing.assert_allclose(result, x)

    def test_negative_zeroed(self):
        x = np.array([-1.0, -2.0, 0.0, 1.0])
        result = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_zero_unchanged(self):
        x = np.array([0.0])
        result = relu(x)
        np.testing.assert_allclose(result, x)


class TestScaledDotProductAttention:
    def test_output_shape(self):
        seq_len, d_k, d_v = 3, 4, 4
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)
        mask = causal_mask(seq_len)
        output = scaled_dot_product_attention(Q, K, V, mask)
        assert output.shape == (seq_len, d_v)

    def test_identical_qk_uniform_weights(self):
        """If Q == K, all valid positions should have similar attention."""
        Q = K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        V = np.array([[1.0], [2.0], [3.0]])
        output = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (3, 1)

    def test_with_causal_mask(self):
        """First token should only attend to itself."""
        Q = K = V = np.eye(3)
        mask = causal_mask(3)
        output = scaled_dot_product_attention(Q, K, V, mask)
        # First row of attention output should equal V[0] = [1, 0, 0]
        np.testing.assert_allclose(output[0], V[0], atol=1e-5)


# ============================================================
# tracer.py tests
# ============================================================


class TestTracer:
    def test_record_single_step(self):
        tracer = Tracer()
        x = np.array([1.0, 2.0, 3.0])
        y = softmax(x)
        tracer.trace("softmax", "softmax(x)", {"x": x}, y, "test softmax")
        assert len(tracer) == 1

    def test_export_dict(self):
        tracer = Tracer()
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        tracer.trace("double", "y = 2x", {"x": x}, y, "double the input")
        result = tracer.export_dict()
        assert len(result) == 1
        assert result[0]["name"] == "double"

    def test_shapes_recorded(self):
        tracer = Tracer()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[2.0, 4.0], [6.0, 8.0]])
        unit = tracer.trace("double", "y = 2x", {"x": x}, y, "double")
        assert unit.shapes["x"] == (2, 2)
        assert unit.shapes["output"] == (2, 2)

    def test_export_markdown(self, tmp_path):
        tracer = Tracer()
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        tracer.trace("double", "y = 2x", {"x": x}, y, "double the input")
        path = str(tmp_path / "test_report.md")
        content = tracer.export_markdown(path, title="Test Report")
        assert "## Step 1: double" in content
        assert "softmax" not in content  # shouldn't have softmax
        assert "y = 2x" in content


# ============================================================
# utils.py tests
# ============================================================


class TestUtils:
    def test_tiny_matrix_shape(self):
        m = tiny_matrix(3, 4)
        assert m.shape == (3, 4)

    def test_tiny_matrix_reproducible(self):
        m1 = tiny_matrix(3, 4, seed=42)
        m2 = tiny_matrix(3, 4, seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_tiny_vector(self):
        v = tiny_vector(5)
        assert v.shape == (5,)

    def test_one_hot(self):
        v = one_hot(2, 4)
        expected = np.array([0.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(v, expected)

    def test_assert_close_passes(self):
        assert_close(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_assert_close_raises_on_shape(self):
        with pytest.raises(AssertionError):
            assert_close(np.array([1.0]), np.array([1.0, 2.0]))

    def test_format_array(self):
        result = format_array(np.array([1.0, 2.0]))
        assert "1.0000" in result
