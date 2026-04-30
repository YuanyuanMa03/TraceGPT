"""
Level 0: Next Token Prediction — Embedding → Linear → Softmax → Prediction

This is the simplest possible language model:
  1. Look up a token's embedding vector
  2. Project it to vocabulary-size logits
  3. Apply softmax to get probabilities
  4. Pick the most likely next token

This level teaches the fundamental "prediction head" that sits on top of
every language model, before any attention or Transformer architecture.

Run:
    python -m levels.level0_next_token
"""

from __future__ import annotations

import sys
import os

import numpy as np

# Add parent to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracegpt.tracer import Tracer
from tracegpt.ops import softmax, linear


def main() -> None:
    tracer = Tracer()

    # ---- Hyperparameters (tiny for hand verification) ----
    vocab_size = 4  # tokens: ["hello", "world", "good", "bye"]
    embed_dim = 3   # embedding dimension
    token_names = ["hello", "world", "good", "bye"]

    print("=" * 60)
    print("  TraceGPT Level 0: Next Token Prediction")
    print("=" * 60)

    # ---- Step 1: Embedding Table ----
    # Each row is a learnable vector for one token
    embedding_table = np.array([
        [1.0, 0.0, 0.0],   # "hello"
        [0.0, 1.0, 0.0],   # "world"
        [0.0, 0.0, 1.0],   # "good"
        [1.0, 1.0, 0.0],   # "bye"
    ])

    # Input: token "hello" (index 0)
    input_token_id = 0
    embedding = embedding_table[input_token_id]

    tracer.trace(
        name="embedding_lookup",
        formula="e = E[token_id]",
        inputs={"embedding_table": embedding_table, "token_id": np.array(input_token_id)},
        output=embedding,
        explanation=f"Look up the embedding vector for token '{token_names[input_token_id]}' (id={input_token_id}). "
                    f"The embedding table has {vocab_size} rows (one per token) and {embed_dim} columns (embedding dim).",
    )

    print(f"\n✓ Step 1: Embedding lookup for '{token_names[input_token_id]}'")
    print(f"  embedding = {embedding}")

    # ---- Step 2: Output Projection (Linear) ----
    # Project from embed_dim → vocab_size to get logits for each token
    W_out = np.array([
        [2.0, 0.5, 0.0, -1.0],
        [-0.5, 1.5, 0.5, 0.0],
        [0.0, -1.0, 2.0, 1.0],
    ])  # shape: (embed_dim, vocab_size)

    b_out = np.array([0.1, -0.2, 0.3, -0.1])  # shape: (vocab_size,)

    logits = linear(embedding, W_out, b_out)

    tracer.trace(
        name="linear_projection",
        formula="logits = e @ W_out + b_out",
        inputs={"embedding": embedding, "W_out": W_out, "b_out": b_out},
        output=logits,
        explanation="Project the embedding to vocabulary-size logits. Each logit is an unnormalized "
                    "score for how likely each token is to be the next token.",
    )

    print(f"\n✓ Step 2: Linear projection → logits")
    print(f"  logits = {logits}")

    # ---- Step 3: Softmax ----
    probs = softmax(logits)

    tracer.trace(
        name="softmax",
        formula="P(token) = exp(logit_i) / Σ exp(logit_j)",
        inputs={"logits": logits},
        output=probs,
        explanation="Convert logits to a probability distribution over the vocabulary. "
                    "The softmax ensures all probabilities are positive and sum to 1.",
    )

    print(f"\n✓ Step 3: Softmax → probabilities")
    print(f"  probs = {probs}")
    print(f"  sum   = {np.sum(probs):.6f} (should be 1.0)")

    # ---- Step 4: Prediction ----
    predicted_id = int(np.argmax(probs))

    tracer.trace(
        name="argmax_prediction",
        formula="predicted_token = argmax(P)",
        inputs={"probabilities": probs},
        output=np.array(predicted_id),
        explanation=f"Pick the token with highest probability. "
                    f"Predicted: '{token_names[predicted_id]}' (id={predicted_id}) "
                    f"with probability {probs[predicted_id]:.4f}",
    )

    print(f"\n✓ Step 4: Prediction")
    print(f"  Predicted next token: '{token_names[predicted_id]}' (prob={probs[predicted_id]:.4f})")
    print(f"\n  Full probability distribution:")
    for i, name in enumerate(token_names):
        bar = "█" * int(probs[i] * 40)
        marker = " ← predicted" if i == predicted_id else ""
        print(f"    {name:>8}: {probs[i]:.4f} {bar}{marker}")

    # ---- Export Report ----
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "level0_next_token.md")
    report_path = os.path.abspath(report_path)
    tracer.export_markdown(report_path, title="TraceGPT Level 0: Next Token Prediction")
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
