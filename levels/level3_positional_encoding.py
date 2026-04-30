"""
Level 3: Sinusoidal Positional Encoding

Implements the sinusoidal positional encoding from "Attention Is All You Need":
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Positional encoding gives the model a sense of token ORDER, since self-attention
is permutation-invariant by itself. Different positions get unique encodings that
the model can use to distinguish "dog bites man" from "man bites dog".

Run:
    python -m levels.level3_positional_encoding
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracegpt.tracer import Tracer
from tracegpt.ops import sinusoidal_position_encoding


def main() -> None:
    tracer = Tracer()

    seq_len = 4
    d_model = 4
    tokens = ["I", "love", "AI", "!"]

    print("=" * 60)
    print("  TraceGPT Level 3: Sinusoidal Positional Encoding")
    print("=" * 60)

    # ---- Step 1: What problem does PE solve? ----
    # Self-attention is permutation-invariant:
    # attention(["I","love","AI"]) == attention(["AI","love","I"]) without PE!
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],   # token 0
        [0.0, 1.0, 0.0, 1.0],   # token 1
        [1.0, 1.0, 0.0, 1.0],   # token 2
        [0.0, 0.0, 1.0, 1.0],   # token 3
    ])

    tracer.trace(
        name="input_without_position",
        formula="X ∈ R^{seq_len × d_model} (no position info)",
        inputs={"tokens": np.array(tokens)},
        output=X,
        explanation="Raw token embeddings contain no position information. "
                    "Self-attention treats them as a SET, not a sequence. "
                    "We need to INJECT position information.",
    )

    # ---- Step 2: Generate sinusoidal PE ----
    PE = sinusoidal_position_encoding(seq_len, d_model)

    tracer.trace(
        name="sinusoidal_pe",
        formula="PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d))",
        inputs={"seq_len": np.array(seq_len), "d_model": np.array(d_model)},
        output=PE,
        explanation="Each position gets a unique sinusoidal encoding. "
                    "Even indices use sin, odd indices use cos. "
                    "The frequencies decrease across dimensions, so each "
                    "dimension corresponds to a different scale of position.",
    )

    print(f"\n✓ Positional Encoding Matrix ({seq_len}×{d_model}):")
    for pos in range(seq_len):
        row = "  ".join(f"{v:8.4f}" for v in PE[pos])
        print(f"  pos {pos} ({tokens[pos]:>4}): [{row}]")

    # ---- Step 3: Verify uniqueness — each position has a different encoding ----
    tracer.trace(
        name="verify_uniqueness",
        formula="PE[i] ≠ PE[j] for all i ≠ j",
        inputs={"PE": PE},
        output=np.array([not np.allclose(PE[i], PE[j])
                         for i in range(seq_len) for j in range(i+1, seq_len)]),
        explanation="Verify that each position gets a DIFFERENT encoding vector. "
                    "This is essential — if two positions had the same encoding, "
                    "the model couldn't distinguish them.",
    )

    # Check pairwise differences
    all_unique = True
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if np.allclose(PE[i], PE[j]):
                all_unique = False
                print(f"  ⚠ pos {i} ≈ pos {j}")
    print(f"\n✓ All positions unique: {all_unique}")

    # ---- Step 4: Visualize the sinusoidal patterns ----
    print(f"\n  Sinusoidal pattern per dimension:")
    print(f"  {'pos':>4}", end="")
    for d in range(d_model):
        fn = "sin" if d % 2 == 0 else "cos"
        print(f"  d{d}({fn})  ", end="")
    print()
    for pos in range(seq_len):
        print(f"  {pos:>4}", end="")
        for d in range(d_model):
            print(f"  {PE[pos, d]:+7.4f}", end="")
        print()

    # ---- Step 5: Add PE to embeddings ----
    X_with_pe = X + PE

    tracer.trace(
        name="add_pe_to_embeddings",
        formula="X_pe = X + PE",
        inputs={"X": X, "PE": PE},
        output=X_with_pe,
        explanation="Add positional encodings to token embeddings. "
                    "Now each vector encodes BOTH what the token is AND where it is. "
                    "This is the standard approach from the original Transformer paper.",
    )

    print(f"\n✓ Embeddings + PE:")
    for pos in range(seq_len):
        row = "  ".join(f"{v:8.4f}" for v in X_with_pe[pos])
        print(f"  pos {pos} ({tokens[pos]:>4}): [{row}]")

    # ---- Step 6: Demonstrate why this matters ----
    # Show that same tokens at different positions get different representations
    same_token_diff_pos = np.allclose(X_with_pe[0], X_with_pe[2])
    print(f"\n✓ Token at pos 0 ≠ Token at pos 2 (even if same embedding): "
          f"{'SAME (bad!)' if same_token_diff_pos else 'DIFFERENT (good!)'}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Key insight: Without PE, Transformers are permutation-invariant.")
    print(f"  With PE, position information is baked into the embeddings.")
    print(f"  Different frequencies capture different position scales:")
    print(f"    - Low dims: fast oscillation → distinguish nearby positions")
    print(f"    - High dims: slow oscillation → capture long-range order")
    print(f"{'='*60}")

    # ---- Export Report ----
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "level3_positional_encoding.md")
    report_path = os.path.abspath(report_path)
    tracer.export_markdown(report_path, title="TraceGPT Level 3: Sinusoidal Positional Encoding")
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
