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


def sinusoidal_position_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding (from "Attention Is All You Need").

    formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Parameters
    ----------
    seq_len : int
        Maximum sequence length.
    d_model : int
        Embedding dimension (must be even).

    Returns
    -------
    np.ndarray
        Positional encoding matrix, shape (seq_len, d_model).
    """
    assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)  # (d_model/2,)
    pe[:, 0::2] = np.sin(position / div_term)  # even indices
    pe[:, 1::2] = np.cos(position / div_term)  # odd indices
    return pe


def multi_head_attention(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    n_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Multi-head attention.

    formula:
        heads = split(X @ W_Q, X @ W_K, X @ W_V) into n_heads
        for each head h:
            attn_h = softmax(Q_h @ K_h^T / sqrt(d_k) + mask) @ V_h
        concat all attn_h, then project: output = concat @ W_O

    Parameters
    ----------
    X : np.ndarray
        Input, shape (seq_len, d_model).
    W_Q, W_K, W_V : np.ndarray
        Projection matrices, shape (d_model, d_model).
    W_O : np.ndarray
        Output projection, shape (d_model, d_model).
    n_heads : int
        Number of attention heads.
    mask : np.ndarray or None
        Attention mask, shape (seq_len, seq_len).

    Returns
    -------
    np.ndarray
        Output, shape (seq_len, d_model).
    """
    seq_len, d_model = X.shape
    d_k = d_model // n_heads
    assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

    # Project to Q, K, V: (seq_len, d_model)
    Q = X @ W_Q  # (seq_len, d_model)
    K = X @ W_K
    V = X @ W_V

    # Reshape to (n_heads, seq_len, d_k)
    Q_heads = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)  # (n_heads, seq_len, d_k)
    K_heads = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # Scaled dot-product attention per head
    head_outputs = []
    for h in range(n_heads):
        Q_h = Q_heads[h]  # (seq_len, d_k)
        K_h = K_heads[h]
        V_h = V_heads[h]

        scores = Q_h @ K_h.T / np.sqrt(d_k)  # (seq_len, seq_len)
        if mask is not None:
            scores = scores + (1 - mask) * (-1e9)
        weights = softmax(scores)
        head_out = weights @ V_h  # (seq_len, d_k)
        head_outputs.append(head_out)

    # Concatenate heads: (seq_len, d_model)
    concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, n_heads * d_k) = (seq_len, d_model)

    # Output projection
    output = concat @ W_O  # (seq_len, d_model)
    return output


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
