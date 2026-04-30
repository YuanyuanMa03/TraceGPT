"""
Level 2: Minimal Transformer Block

A single Transformer block combines:
  1. Multi-head causal self-attention (we use 1 head for clarity)
  2. Residual connection + LayerNorm
  3. Feed-forward network (linear → ReLU → linear)
  4. Residual connection + LayerNorm

Architecture:
    input
      │
      ▼
    ┌─────────────────────┐
    │  Self-Attention      │
    └────────┬────────────┘
      │      │
      │   + input (residual)
      ▼      │
    LayerNorm │
      │      │
      ▼      ▼
    ┌─────────────────────┐
    │  FFN: Linear→ReLU→Linear │
    └────────┬────────────┘
      │      │
      │   + (residual)
      ▼      │
    LayerNorm │
      │      │
      ▼      ▼
    output

Run:
    python -m levels.level2_transformer_block
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracegpt.tracer import Tracer
from tracegpt.ops import (
    softmax,
    causal_mask,
    layer_norm,
    linear,
    relu,
    scaled_dot_product_attention,
)


def main() -> None:
    tracer = Tracer()

    # ---- Hyperparameters ----
    seq_len = 3
    d_model = 4
    d_ff = 8       # feed-forward hidden dimension
    d_k = d_model  # single head, d_k = d_model
    tokens = ["I", "love", "AI"]

    print("=" * 60)
    print("  TraceGPT Level 2: Transformer Block")
    print("=" * 60)

    # ---- Input ----
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ])  # (3, 4)

    tracer.trace(
        name="input",
        formula="X ∈ R^{seq_len × d_model}",
        inputs={"tokens": np.array(tokens)},
        output=X,
        explanation=f"Input sequence: {tokens}. Shape: ({seq_len}, {d_model}).",
    )

    # =============================================
    # Part 1: Self-Attention + Residual + LayerNorm
    # =============================================

    # Q, K, V projections (single head: d_k = d_model = d_v)
    W_Q = np.eye(d_model) * 0.5
    W_K = np.eye(d_model) * 0.5
    W_V = np.eye(d_model) * 0.5
    b_Q = np.zeros(d_model)
    b_K = np.zeros(d_model)
    b_V = np.zeros(d_model)

    Q = linear(X, W_Q, b_Q)
    K = linear(X, W_K, b_K)
    V = linear(X, W_V, b_V)

    tracer.trace(
        name="attention_qkv",
        formula="Q=X@W_Q, K=X@W_K, V=X@W_V",
        inputs={"X": X, "W_Q": W_Q, "W_K": W_K, "W_V": W_V},
        output=np.stack([Q, K, V]),
        explanation="Project input into Q, K, V for single-head attention. "
                    "Using scaled identity matrices for simplicity.",
    )

    # Causal mask + scaled dot-product attention
    mask = causal_mask(seq_len)
    attn_output = scaled_dot_product_attention(Q, K, V, mask)

    tracer.trace(
        name="self_attention",
        formula="Attn(Q,K,V) = softmax(QK^T/√d_k + mask) @ V",
        inputs={"Q": Q, "K": K, "V": V, "mask": mask},
        output=attn_output,
        explanation="Scaled dot-product attention with causal mask. "
                    "Each token attends to itself and all previous tokens.",
    )

    # Residual connection
    residual1 = X + attn_output

    tracer.trace(
        name="residual_1",
        formula="residual = X + Attn(X)",
        inputs={"X": X, "attn_output": attn_output},
        output=residual1,
        explanation="Add the attention output to the original input. "
                    "Residual connections help gradients flow and stabilize training.",
    )

    # LayerNorm
    gamma1 = np.ones(d_model)
    beta1 = np.zeros(d_model)
    norm1 = layer_norm(residual1, gamma1, beta1)

    tracer.trace(
        name="layer_norm_1",
        formula="norm = LayerNorm(X + Attn(X))",
        inputs={"residual": residual1, "gamma": gamma1, "beta": beta1},
        output=norm1,
        explanation="Normalize across features for each position. Stabilizes training.",
    )

    # =============================================
    # Part 2: Feed-Forward Network + Residual + LayerNorm
    # =============================================

    # FFN: Linear1 → ReLU → Linear2
    W_ff1 = np.random.RandomState(42).randn(d_model, d_ff) * 0.5
    b_ff1 = np.zeros(d_ff)
    W_ff2 = np.random.RandomState(43).randn(d_ff, d_model) * 0.5
    b_ff2 = np.zeros(d_model)

    ff_hidden = linear(norm1, W_ff1, b_ff1)

    tracer.trace(
        name="ffn_linear1",
        formula="h = norm @ W_ff1 + b_ff1",
        inputs={"input": norm1, "W_ff1": W_ff1, "b_ff1": b_ff1},
        output=ff_hidden,
        explanation=f"First linear layer of FFN: expand from {d_model} to {d_ff} dimensions.",
    )

    ff_activated = relu(ff_hidden)

    tracer.trace(
        name="ffn_relu",
        formula="h = ReLU(h)",
        inputs={"h": ff_hidden},
        output=ff_activated,
        explanation="ReLU activation: set all negative values to 0. Introduces non-linearity.",
    )

    ff_output = linear(ff_activated, W_ff2, b_ff2)

    tracer.trace(
        name="ffn_linear2",
        formula="ff_out = h @ W_ff2 + b_ff2",
        inputs={"h": ff_activated, "W_ff2": W_ff2, "b_ff2": b_ff2},
        output=ff_output,
        explanation=f"Second linear layer: project back from {d_ff} to {d_model} dimensions.",
    )

    # Residual connection 2
    residual2 = norm1 + ff_output

    tracer.trace(
        name="residual_2",
        formula="residual = norm1 + FFN(norm1)",
        inputs={"norm1": norm1, "ff_output": ff_output},
        output=residual2,
        explanation="Second residual connection around the FFN.",
    )

    # LayerNorm 2
    gamma2 = np.ones(d_model)
    beta2 = np.zeros(d_model)
    output = layer_norm(residual2, gamma2, beta2)

    tracer.trace(
        name="layer_norm_2",
        formula="output = LayerNorm(norm1 + FFN(norm1))",
        inputs={"residual": residual2, "gamma": gamma2, "beta": beta2},
        output=output,
        explanation="Final layer normalization. This is the Transformer block output.",
    )

    # ---- Summary ----
    print(f"\n✓ Transformer Block Complete!")
    print(f"  Input shape:  {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"\n  Input:\n{X}")
    print(f"\n  Output:\n{np.round(output, 4)}")

    # ---- Export Report ----
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "level2_transformer_block.md")
    report_path = os.path.abspath(report_path)
    tracer.export_markdown(report_path, title="TraceGPT Level 2: Transformer Block")
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
