"""
viz.py — Beautiful text-based visualization for TraceGPT.

The key differentiator: every number has a language meaning.
Instead of raw matrices, we show word-labeled heatmaps, annotated traces,
and human-readable attention patterns.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def attention_heatmap(
    tokens: list[str],
    weights: np.ndarray,
    title: str = "Attention Weights",
) -> str:
    """
    Generate a text-based attention heatmap with word labels.

    Example output:
        Attention: "the cat sat"
        ┌─────────────────────────────────┐
        │          the   cat   sat        │
        │  the   ▓▓▓▓▓▓▓░░░░░░░░░░░░░░  │
        │  cat   ▓▓▓▓░░░▓▓▓▓▓▓░░░░░░░░  │
        │  sat   ▓▓░░░░░▓▓▓░░░▓▓▓▓▓▓▓▓  │
        └─────────────────────────────────┘

    Parameters
    ----------
    tokens : list[str]
        Token names for row/column labels.
    weights : np.ndarray
        Attention weight matrix, shape (seq_len, seq_len).
    title : str
        Title for the heatmap.

    Returns
    -------
    str
        Formatted string for terminal display.
    """
    seq_len = len(tokens)
    assert weights.shape == (seq_len, seq_len)

    # Unicode block characters for intensity
    blocks = " ░▒▓█"

    # Calculate column width
    max_token_len = max(len(t) for t in tokens)
    col_width = max(max_token_len, 8)

    lines = []
    lines.append(f"  {title}: \"{' '.join(tokens)}\"")
    lines.append("")

    # Header
    header = " " * (col_width + 2)
    for t in tokens:
        header += f"{t:>{col_width}}"
    lines.append(header)

    # Separator
    sep = " " * (col_width + 2) + "─" * (col_width * seq_len)
    lines.append(sep)

    # Rows
    for i, token in enumerate(tokens):
        row = f"  {token:>{col_width}} │"
        for j in range(seq_len):
            val = weights[i, j]
            # Map value to block intensity (0-4)
            intensity = min(int(val * len(blocks)), len(blocks) - 1)
            intensity = max(intensity, 0)
            cell = blocks[intensity] * (col_width // 2 + 1)
            row += f"{cell:<{col_width}}"
        # Add actual values on the right
        vals = "  ".join(f"{weights[i,j]:.2f}" for j in range(seq_len))
        row += f" │ {vals}"
        lines.append(row)

    lines.append(sep)

    return "\n".join(lines)


def labeled_matrix(
    row_labels: list[str],
    col_labels: list[str],
    matrix: np.ndarray,
    title: str = "",
    precision: int = 2,
    max_width: int = 10,
) -> str:
    """
    Display a matrix with row and column labels.

    Example:
        Q (Queries) for "the cat sat"
                d0     d1     d2     d3
        the   │ 1.00  0.00  0.50  0.25 │
        cat   │ 0.00  1.00  0.25  0.50 │
        sat   │ 0.50  0.25  1.00  0.00 │
    """
    rows, cols = matrix.shape
    assert len(row_labels) == rows
    assert len(col_labels) == cols

    max_row_label = max(len(str(l)) for l in row_labels)
    max_col_label = max(len(str(l)) for l in col_labels)
    col_w = max(max_col_label, max_width)

    lines = []
    if title:
        lines.append(f"  {title}")
        lines.append("")

    # Header
    header = " " * (max_row_label + 3)
    for label in col_labels:
        header += f"{label:>{col_w}}"
    lines.append(header)

    # Separator
    sep = " " * (max_row_label + 1) + "┼" + "─" * (col_w * cols)
    lines.append(sep)

    # Rows
    for i, label in enumerate(row_labels):
        row = f"  {label:>{max_row_label}} │"
        for j in range(cols):
            row += f"{matrix[i,j]:>{col_w}.{precision}f}"
        row += " │"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


def probability_bar(
    tokens: list[str],
    probs: np.ndarray,
    title: str = "Next Token Probability",
    width: int = 30,
    highlight: int | None = None,
) -> str:
    """
    Display a probability distribution with word labels and bars.

    Example:
        Next Token Prediction after "the cat sat on the"
           mat  ████████████████████████████████  0.4523  ← predicted
         happy  ████████░░░░░░░░░░░░░░░░░░░░░░░░  0.1234
           cat  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0891
          ...
    """
    lines = []
    lines.append(f"  {title}")
    lines.append("")

    # Sort by probability (descending)
    indices = np.argsort(probs)[::-1]

    for idx in indices:
        prob = probs[idx]
        bar_len = int(prob * width)
        filled = "█" * bar_len
        empty = "░" * (width - bar_len)
        marker = " ← predicted" if idx == highlight else ""
        token = tokens[idx]
        lines.append(f"    {token:>10}  {filled}{empty}  {prob:.4f}{marker}")

    return "\n".join(lines)


def trace_step_human(
    step_name: str,
    description: str,
    formula: str,
    tokens: list[str],
    before: np.ndarray | None = None,
    after: np.ndarray | None = None,
    detail: str = "",
) -> str:
    """
    Format a single trace step in human-readable form.

    Example:
        ─────────────────────────────────────
        Step: Embedding Lookup
        "the cat sat on the" → find each word's vector

        Formula: e = Embedding[token_id]

        "the" → [0.12, -0.34, 0.56, ...]
        "cat" → [-0.23, 0.45, -0.12, ...]
        "sat" → [0.67, 0.11, -0.89, ...]
        ─────────────────────────────────────
    """
    lines = []
    lines.append(f"  {'─' * 50}")
    lines.append(f"  📍 {step_name}")
    lines.append(f"     {description}")
    lines.append("")
    lines.append(f"     Formula: {formula}")
    lines.append("")

    if detail:
        lines.append(f"     {detail}")

    return "\n".join(lines)


def word_level_trace(
    tokens: list[str],
    vectors: np.ndarray,
    dim_names: list[str] | None = None,
    title: str = "",
    precision: int = 3,
) -> str:
    """
    Show each word's vector representation side by side.

    Example:
        Word Vectors (after Positional Encoding)
                d0      d1      d2      d3
        "the"   0.120  -0.340   0.560   0.789
        "cat"  -0.230   0.450  -0.120   0.334
        "sat"   0.670   0.110  -0.890   0.123
    """
    if dim_names is None:
        dim_names = [f"d{i}" for i in range(vectors.shape[1])]

    quoted_tokens = [f'"{t}"' for t in tokens]
    return labeled_matrix(quoted_tokens, dim_names, vectors, title=title, precision=precision)


def generation_trace(
    prompt: list[str],
    generated: list[str],
    step_probs: list[dict[str, float]] | None = None,
) -> str:
    """
    Show the generation process step by step with words.

    Example:
        Generation Process:
        ─────────────────────────────────────
        Step 0: "the cat" → predict "sat" (p=0.34)
        Step 1: "the cat sat" → predict "on" (p=0.52)
        Step 2: "the cat sat on" → predict "the" (p=0.71)
        Step 3: "the cat sat on the" → predict "mat" (p=0.45)
        ─────────────────────────────────────
        Final: "the cat sat on the mat"
    """
    lines = []
    lines.append("  Generation Process:")
    lines.append(f"  {'─' * 50}")

    all_tokens = list(prompt)
    for i, new_token in enumerate(generated):
        context = " ".join(all_tokens)
        prob_str = ""
        if step_probs and i < len(step_probs):
            prob = step_probs[i].get(new_token, 0)
            prob_str = f" (p={prob:.4f})"
        lines.append(f"  Step {i}: \"{context}\" → \"{new_token}\"{prob_str}")
        all_tokens.append(new_token)

    lines.append(f"  {'─' * 50}")
    lines.append(f"  Final: \"{' '.join(all_tokens)}\"")
    return "\n".join(lines)
