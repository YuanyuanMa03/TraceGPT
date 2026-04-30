"""
Level 4: Multi-Head Attention + Full GPT Forward Pass

Builds a complete (tiny) GPT forward pass with:
  1. Token embedding + Positional encoding
  2. Multi-head causal self-attention (2 heads)
  3. Residual + LayerNorm
  4. Feed-forward network
  5. Residual + LayerNorm
  6. Output projection → Softmax → Next token prediction

This level shows how all the pieces fit together into a working model.

Run:
    python -m levels.level4_multihead_gpt
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
    sinusoidal_position_encoding,
    multi_head_attention,
)


def main() -> None:
    tracer = Tracer()

    # ---- Hyperparameters ----
    vocab_size = 5
    seq_len = 3
    d_model = 4
    d_ff = 8
    n_heads = 2
    d_k = d_model // n_heads  # 2
    token_names = ["<PAD>", "hello", "world", "good", "bye"]

    print("=" * 60)
    print("  TraceGPT Level 4: Multi-Head Attention + Tiny GPT")
    print("=" * 60)
    print(f"  vocab_size={vocab_size}, seq_len={seq_len}, d_model={d_model}")
    print(f"  n_heads={n_heads}, d_k={d_k}, d_ff={d_ff}")

    # ==========================================
    # Step 1: Token Embedding
    # ==========================================
    embedding_table = np.array([
        [ 0.0,  0.0,  0.0,  0.0],   # <PAD> (index 0)
        [ 1.0,  0.0,  0.0,  0.0],   # "hello" (index 1)
        [ 0.0,  1.0,  0.0,  0.0],   # "world" (index 2)
        [ 0.0,  0.0,  1.0,  0.0],   # "good" (index 3)
        [ 0.0,  0.0,  0.0,  1.0],   # "bye" (index 4)
    ])  # (vocab_size, d_model)

    # Input: "hello world good" → token IDs [1, 2, 3]
    input_ids = np.array([1, 2, 3])
    X_tok = embedding_table[input_ids]  # (seq_len, d_model)

    tracer.trace(
        name="token_embedding",
        formula="X_tok = Embedding[token_ids]",
        inputs={"input_ids": input_ids, "embedding_table": embedding_table},
        output=X_tok,
        explanation=f"Look up embeddings for input tokens: {[token_names[i] for i in input_ids]}. "
                    f"Each token becomes a {d_model}-dim vector.",
    )

    print(f"\n✓ Token embeddings: shape {X_tok.shape}")
    print(f"  {[token_names[i] for i in input_ids]}")

    # ==========================================
    # Step 2: Positional Encoding
    # ==========================================
    PE = sinusoidal_position_encoding(seq_len, d_model)
    X = X_tok + PE

    tracer.trace(
        name="add_positional_encoding",
        formula="X = X_tok + PE",
        inputs={"X_tok": X_tok, "PE": PE},
        output=X,
        explanation="Add sinusoidal positional encodings so the model knows "
                    "the ORDER of tokens in the sequence.",
    )

    print(f"\n✓ Embeddings + PE: shape {X.shape}")

    # ==========================================
    # Step 3: Multi-Head Causal Self-Attention
    # ==========================================
    # Weight matrices for multi-head attention
    # Using simple structured matrices for hand-verifiability
    W_Q = np.array([
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.5, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0, 1.0],
    ])  # (d_model, d_model)

    W_K = np.array([
        [ 0.8, 0.0,  0.3, 0.0],
        [ 0.0, 0.8,  0.0, 0.3],
        [ 0.3, 0.0,  0.8, 0.0],
        [ 0.0, 0.3,  0.0, 0.8],
    ])

    W_V = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    W_O = np.array([
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
    ])  # (d_model, d_model)

    mask = causal_mask(seq_len)

    # Step-by-step: show what each head sees
    Q_all = X @ W_Q
    K_all = X @ W_K
    V_all = X @ W_V

    tracer.trace(
        name="mha_qkv_projection",
        formula="Q = X @ W_Q, K = X @ W_K, V = X @ W_V",
        inputs={"X": X, "W_Q": W_Q, "W_K": W_K, "W_V": W_V},
        output=np.stack([Q_all, K_all, V_all]),
        explanation=f"Project input into Q, K, V. Total dim={d_model}, "
                    f"split into {n_heads} heads × {d_k} dims each.",
    )

    # Show head splitting
    Q_heads = Q_all.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K_heads = K_all.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V_all.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    for h in range(n_heads):
        print(f"\n  Head {h}:")
        print(f"    Q_h: {Q_heads[h].shape}\n{Q_heads[h]}")
        print(f"    K_h: {K_heads[h].shape}\n{K_heads[h]}")

    attn_output = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask)

    tracer.trace(
        name="multi_head_attention",
        formula="MHA(X) = Concat(head_1,...,head_h) @ W_O\n  where head_i = softmax(Q_i @ K_i^T / √d_k + mask) @ V_i",
        inputs={"X": X, "W_Q": W_Q, "W_K": W_K, "W_V": W_V, "W_O": W_O, "mask": mask, "n_heads": np.array(n_heads)},
        output=attn_output,
        explanation=f"Multi-head attention with {n_heads} heads. "
                    f"Each head independently attends to the sequence in {d_k}-dim subspaces. "
                    f"Outputs are concatenated and projected back to {d_model} dims. "
                    f"Causal mask prevents attending to future tokens.",
    )

    print(f"\n✓ Multi-head attention output: shape {attn_output.shape}")
    print(f"  {np.round(attn_output, 4)}")

    # ==========================================
    # Step 4: Residual + LayerNorm
    # ==========================================
    residual1 = X + attn_output
    gamma1 = np.ones(d_model)
    beta1 = np.zeros(d_model)
    norm1 = layer_norm(residual1, gamma1, beta1)

    tracer.trace(
        name="residual_layernorm_1",
        formula="norm1 = LayerNorm(X + MHA(X))",
        inputs={"X": X, "attn_output": attn_output},
        output=norm1,
        explanation="First residual connection + layer normalization. "
                    "The residual lets gradients flow directly, LayerNorm stabilizes.",
    )

    # ==========================================
    # Step 5: Feed-Forward Network
    # ==========================================
    rng = np.random.RandomState(42)
    W_ff1 = rng.randn(d_model, d_ff) * 0.3
    b_ff1 = np.zeros(d_ff)
    W_ff2 = rng.randn(d_ff, d_model) * 0.3
    b_ff2 = np.zeros(d_model)

    ff_hidden = linear(norm1, W_ff1, b_ff1)
    ff_activated = relu(ff_hidden)
    ff_output = linear(ff_activated, W_ff2, b_ff2)

    tracer.trace(
        name="feed_forward",
        formula="FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2",
        inputs={"input": norm1, "W_ff1": W_ff1, "W_ff2": W_ff2},
        output=ff_output,
        explanation=f"Position-wise feed-forward: expand {d_model}→{d_ff}→{d_model}. "
                    f"Applied independently to each position.",
    )

    # ==========================================
    # Step 6: Residual + LayerNorm
    # ==========================================
    residual2 = norm1 + ff_output
    gamma2 = np.ones(d_model)
    beta2 = np.zeros(d_model)
    output = layer_norm(residual2, gamma2, beta2)

    tracer.trace(
        name="residual_layernorm_2",
        formula="output = LayerNorm(norm1 + FFN(norm1))",
        inputs={"norm1": norm1, "ff_output": ff_output},
        output=output,
        explanation="Second residual + LayerNorm. This is the Transformer block output.",
    )

    print(f"\n✓ Transformer block output: shape {output.shape}")
    print(f"  {np.round(output, 4)}")

    # ==========================================
    # Step 7: Output Projection → Next Token Prediction
    # ==========================================
    # Project last token's output to vocab logits
    W_out = embedding_table.T  # weight tying: reuse embedding table
    b_out = np.zeros(vocab_size)
    last_hidden = output[-1:]  # (1, d_model) — last position
    logits = linear(last_hidden, W_out, b_out)[0]  # (vocab_size,)

    tracer.trace(
        name="output_projection",
        formula="logits = output[-1] @ W_out + b_out  (weight-tied with embeddings)",
        inputs={"last_hidden": last_hidden, "W_out": W_out},
        output=logits,
        explanation="Project the last position's hidden state to vocabulary-size logits. "
                    "Using weight tying: the output projection shares weights with the embedding table.",
    )

    probs = softmax(logits)

    tracer.trace(
        name="output_softmax",
        formula="P(next_token) = softmax(logits)",
        inputs={"logits": logits},
        output=probs,
        explanation="Convert logits to probabilities over the vocabulary.",
    )

    predicted_id = int(np.argmax(probs))

    print(f"\n{'='*60}")
    print(f"  🎯 Next Token Prediction")
    print(f"  Input:  {[token_names[i] for i in input_ids]}")
    print(f"  Predicted: '{token_names[predicted_id]}' (prob={probs[predicted_id]:.4f})")
    print(f"\n  Probability distribution:")
    for i, name in enumerate(token_names):
        bar = "█" * int(probs[i] * 40)
        marker = " ← predicted" if i == predicted_id else ""
        print(f"    {name:>8}: {probs[i]:.4f} {bar}{marker}")
    print(f"{'='*60}")

    # ---- Export Report ----
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "level4_multihead_gpt.md")
    report_path = os.path.abspath(report_path)
    tracer.export_markdown(report_path, title="TraceGPT Level 4: Multi-Head Attention + Tiny GPT")
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
