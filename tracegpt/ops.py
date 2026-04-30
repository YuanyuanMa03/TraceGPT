"""
ops.py — Core tensor operations for TraceGPT.

Every function follows the same contract:
  - Accepts numpy arrays (tiny, hand-verifiable matrices).
  - Returns the result as a numpy array.
  - Pure NumPy, no PyTorch.

Operations:
  - softmax(x, axis=-1)
  - causal_mask(seq_len)
  - layer_norm(x, gamma, beta, eps=1e-5)
  - linear(x, W, b)
  - relu(x)
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically-stable softmax.

    formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Parameters
    ----------
    x : np.ndarray
        Input logits.
    axis : int
        Axis along which to compute softmax (default: last axis).

    Returns
    -------
    np.ndarray
        Probability distribution (sums to 1 along the given axis).
    """
    # Numerical stability: subtract max
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (lower-triangular) attention mask.

    formula: M[i][j] = 1 if j <= i, else 0

    Token at position i can only attend to positions j where j <= i.
    This prevents "looking into the future" during autoregressive generation.

    Parameters
    ----------
    seq_len : int
        Length of the sequence.

    Returns
    -------
    np.ndarray
        A (seq_len, seq_len) lower-triangular mask of 0s and 1s.

    Example
    -------
    >>> causal_mask(3)
    array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]])
    """
    return np.tril(np.ones((seq_len, seq_len)))


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Layer normalization.

    formula:
        mu = mean(x)
        sigma = sqrt(mean((x - mu)^2) + eps)
        output = gamma * (x - mu) / sigma + beta

    Parameters
    ----------
    x : np.ndarray
        Input tensor, shape (..., d_model).
    gamma : np.ndarray
        Scale parameter, shape (d_model,).
    beta : np.ndarray
        Shift parameter, shape (d_model,).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray
        Normalized tensor, same shape as x.
    """
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
    sigma = np.sqrt(var + eps)
    x_norm = (x - mu) / sigma
    return gamma * x_norm + beta


def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Linear (fully-connected) layer.

    formula: output = x @ W + b

    Parameters
    ----------
    x : np.ndarray
        Input, shape (*, in_features).
    W : np.ndarray
        Weight matrix, shape (in_features, out_features).
    b : np.ndarray
        Bias vector, shape (out_features,).

    Returns
    -------
    np.ndarray
        Output, shape (*, out_features).
    """
    return x @ W + b


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation.

    formula: relu(x) = max(0, x)

    Parameters
    ----------
    x : np.ndarray
        Input tensor.

    Returns
    -------
    np.ndarray
        Output with all negative values set to 0.
    """
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Approximate GELU activation (used in many Transformer models).

    formula: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Parameters
    ----------
    x : np.ndarray
        Input tensor.

    Returns
    -------
    np.ndarray
        GELU-activated output.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Scaled dot-product attention.

    formula:
        scores = Q @ K^T / sqrt(d_k)
        if mask: scores = scores + (1 - mask) * (-1e9)
        weights = softmax(scores)
        output = weights @ V

    Parameters
    ----------
    Q : np.ndarray
        Queries, shape (seq_len, d_k).
    K : np.ndarray
        Keys, shape (seq_len, d_k).
    V : np.ndarray
        Values, shape (seq_len, d_v).
    mask : np.ndarray or None
        Attention mask, shape (seq_len, seq_len). 1 = attend, 0 = mask out.

    Returns
    -------
    np.ndarray
        Attention output, shape (seq_len, d_v).
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (1 - mask) * (-1e9)

    weights = softmax(scores)
    output = weights @ V
    return output
