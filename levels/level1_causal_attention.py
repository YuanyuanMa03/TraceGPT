"""
Level 1: Causal Self-Attention — Q, K, V, Attention Scores, Mask, Softmax, Output

This level builds single-head causal self-attention from scratch:
  1. Project input into Query (Q), Key (K), Value (V) matrices
  2. Compute attention scores: Q @ K^T
  3. Scale by 1/sqrt(d_k)
  4. Apply causal mask (no peeking into the future)
  5. Softmax to get attention weights
  6. Weighted sum of V to get output

This is the core mechanism that lets Transformers "pay attention" to
different parts of the input sequence.

Run:
    python -m levels.level1_causal_attention
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracegpt.tracer import Tracer
from tracegpt.ops import softmax, causal_mask, linear, scaled_dot_product_attention


def main() -> None:
    tracer = Tracer()

    # ---- Hyperparameters ----
    seq_len = 3    # tiny sequence: ["I", "love", "AI"]
    d_model = 4    # embedding dimension
    d_k = 3        # key/query dimension
    d_v = 3        # value dimension
    tokens = ["I", "love", "AI"]

    print("=" * 60)
    print("  TraceGPT Level 1: Causal Self-Attention")
    print("=" * 60)

    # ---- Input Embeddings ----
    # Pretend we already have embeddings for ["I", "love", "AI"]
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],   # "I"
        [0.0, 1.0, 0.0, 1.0],   # "love"
        [1.0, 1.0, 0.0, 1.0],   # "AI"
    ])  # shape: (seq_len=3, d_model=4)

    tracer.trace(
        name="input_embeddings",
        formula="X ∈ R^{seq_len × d_model}",
        inputs={"tokens": np.array(tokens)},
        output=X,
        explanation=f"Input sequence of {seq_len} tokens, each embedded into a {d_model}-dim vector. "
                    f"Tokens: {tokens}",
    )

    # ---- Step 1: Compute Q, K, V ----
    W_Q = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ])  # (d_model, d_k)

    W_K = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.5, 0.5],
    ])  # (d_model, d_k)

    W_V = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5],
    ])  # (d_model, d_v)

    b_Q = np.zeros(d_k)
    b_K = np.zeros(d_k)
    b_V = np.zeros(d_v)

    Q = linear(X, W_Q, b_Q)
    K = linear(X, W_K, b_K)
    V = linear(X, W_V, b_V)

    tracer.trace(
        name="compute_qkv",
        formula="Q = X @ W_Q + b_Q,  K = X @ W_K + b_K,  V = X @ W_V + b_V",
        inputs={"X": X, "W_Q": W_Q, "W_K": W_K, "W_V": W_V},
        output=np.stack([Q, K, V]),
        explanation="Project each token's embedding into three vectors: "
                    "Query (what am I looking for?), Key (what do I contain?), "
                    "Value (what information do I provide?).",
    )

    print(f"\n✓ Q (Queries):\n{Q}")
    print(f"\n✓ K (Keys):\n{K}")
    print(f"\n✓ V (Values):\n{V}")

    # ---- Step 2: Attention Scores ----
    scores = Q @ K.T  # (seq_len, seq_len)

    tracer.trace(
        name="attention_scores",
        formula="scores = Q @ K^T",
        inputs={"Q": Q, "K": K},
        output=scores,
        explanation="Raw attention scores: how much each token 'attends to' every other token. "
                    "scores[i][j] = similarity between token i's query and token j's key.",
    )

    print(f"\n✓ Raw attention scores:\n{scores}")

    # ---- Step 3: Scale by 1/sqrt(d_k) ----
    scaled_scores = scores / np.sqrt(d_k)

    tracer.trace(
        name="scale_scores",
        formula="scaled_scores = scores / sqrt(d_k)",
        inputs={"scores": scores, "d_k": np.array(d_k)},
        output=scaled_scores,
        explanation=f"Scale down by sqrt(d_k)={np.sqrt(d_k):.4f}. This prevents the dot products "
                    f"from growing too large when d_k is big, which would push softmax into "
                    f"regions with tiny gradients.",
    )

    print(f"\n✓ Scaled scores:\n{scaled_scores}")

    # ---- Step 4: Causal Mask ----
    mask = causal_mask(seq_len)

    tracer.trace(
        name="causal_mask",
        formula="M[i][j] = 1 if j ≤ i, else 0",
        inputs={"seq_len": np.array(seq_len)},
        output=mask,
        explanation="Lower-triangular mask: each token can only attend to itself and earlier tokens. "
                    "This is what makes the model 'causal' — it cannot see the future during generation.",
    )

    print(f"\n✓ Causal mask:\n{mask}")

    # Apply mask
    masked_scores = scaled_scores + (1 - mask) * (-1e9)

    tracer.trace(
        name="apply_mask",
        formula="masked_scores = scaled_scores + (1 - M) * (-inf)",
        inputs={"scaled_scores": scaled_scores, "mask": mask},
        output=masked_scores,
        explanation="Add -inf to positions where mask=0. After softmax, these become ~0 probability, "
                    "preventing attention to future tokens.",
    )

    print(f"\n✓ Masked scores:\n{masked_scores}")

    # ---- Step 5: Softmax ----
    attn_weights = softmax(masked_scores)

    tracer.trace(
        name="attention_softmax",
        formula="weights = softmax(masked_scores)",
        inputs={"masked_scores": masked_scores},
        output=attn_weights,
        explanation="Convert masked scores to attention weights (probability distribution). "
                    "Each row sums to 1. Positions with -inf become 0.",
    )

    print(f"\n✓ Attention weights:\n{attn_weights}")
    print(f"  Row sums: {attn_weights.sum(axis=1)}")

    # ---- Step 6: Weighted Sum of V ----
    attn_output = attn_weights @ V

    tracer.trace(
        name="attention_output",
        formula="output = weights @ V",
        inputs={"weights": attn_weights, "V": V},
        output=attn_output,
        explanation="Take a weighted sum of value vectors, where weights are the attention probabilities. "
                    "Each output token is a mix of the values it attends to.",
    )

    print(f"\n✓ Attention output:\n{attn_output}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Attention mechanism complete!")
    print(f"  Input:  {X.shape} → Output: {attn_output.shape}")
    print(f"{'='*60}")

    # ---- Export Report ----
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "level1_causal_attention.md")
    report_path = os.path.abspath(report_path)
    tracer.export_markdown(report_path, title="TraceGPT Level 1: Causal Self-Attention")
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
