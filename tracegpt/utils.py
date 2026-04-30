"""
utils.py — Utility functions for TraceGPT.

Helper functions for array creation, formatting, and verification.
"""

from __future__ import annotations

import numpy as np


def tiny_matrix(rows: int, cols: int, seed: int = 42, scale: float = 1.0) -> np.ndarray:
    """
    Create a tiny hand-verifiable matrix with integer-ish values.

    Parameters
    ----------
    rows, cols : int
        Matrix dimensions.
    seed : int
        Random seed for reproducibility.
    scale : float
        Scale factor for values.

    Returns
    -------
    np.ndarray
        A (rows, cols) matrix with small values.
    """
    rng = np.random.RandomState(seed)
    return np.round(rng.randn(rows, cols) * scale, decimals=2)


def tiny_vector(dim: int, seed: int = 42) -> np.ndarray:
    """
    Create a tiny vector with small values.

    Parameters
    ----------
    dim : int
        Vector dimension.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        A (dim,) vector.
    """
    rng = np.random.RandomState(seed)
    return np.round(rng.randn(dim), decimals=2)


def one_hot(index: int, dim: int) -> np.ndarray:
    """
    Create a one-hot vector.

    Parameters
    ----------
    index : int
        Position of the 1.
    dim : int
        Total dimension.

    Returns
    -------
    np.ndarray
        One-hot vector of shape (dim,).
    """
    vec = np.zeros(dim)
    vec[index] = 1.0
    return vec


def assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float = 1e-6,
    label: str = "",
) -> None:
    """
    Assert two arrays are element-wise close.

    Parameters
    ----------
    actual, expected : np.ndarray
        Arrays to compare.
    atol : float
        Absolute tolerance.
    label : str
        Optional label for error messages.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"[{label}] Shape mismatch: actual={actual.shape}, expected={expected.shape}"
        )
    if not np.allclose(actual, expected, atol=atol):
        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        raise AssertionError(
            f"[{label}] Values differ (max diff={max_diff:.2e}, atol={atol})\n"
            f"  actual:\n{actual}\n  expected:\n{expected}"
        )


def format_array(arr: np.ndarray, precision: int = 4) -> str:
    """
    Pretty-print a numpy array for display.

    Parameters
    ----------
    arr : np.ndarray
        Array to format.
    precision : int
        Decimal places.

    Returns
    -------
    str
        Formatted string.
    """
    if arr.ndim == 0:
        return f"{float(arr):.{precision}f}"
    if arr.ndim == 1:
        return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"
    rows = []
    for row in arr:
        rows.append("  [" + ", ".join(f"{v:.{precision}f}" for v in row) + "]")
    return "[\n" + ",\n".join(rows) + "\n]"
