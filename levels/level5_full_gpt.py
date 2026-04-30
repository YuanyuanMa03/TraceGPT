"""
Level 5: Full GPT-2 Style Forward Pass + Autoregressive Generation

A complete tiny GPT model:
  - Token Embedding + Sinusoidal Positional Encoding
  - 2-layer Transformer with Multi-Head Attention
  - Weight-tied output projection
  - Greedy and temperature-based generation

Run:
    python -m levels.level5_full_gpt
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracegpt.tracer import Tracer
from tracegpt.model import TinyGPT, GPTConfig
from tracegpt.ops import softmax


# A tiny vocabulary for demonstration
VOCAB = {
    "<PAD>": 0, "<EOS>": 1,
    "hello": 2, "world": 3, "good": 4, "bye": 5,
    "the": 6, "cat": 7, "sat": 8, "on": 9, "mat": 10,
    "is": 11, "happy": 12, "sad": 13, "and": 14, "dog": 15,
}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}


def decode(ids):
    return [ID_TO_TOKEN.get(i, f"<{i}>") for i in ids]


def main() -> None:
    tracer = Tracer()

    print("=" * 60)
    print("  TraceGPT Level 5: Full GPT Forward Pass + Generation")
    print("=" * 60)

    # ---- Create Model ----
    config = GPTConfig(
        vocab_size=len(VOCAB),
        max_seq_len=8,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        weight_tying=True,
        activation="gelu",
    )
    model = TinyGPT(config)
    print(f"\n✓ Model created: {model}")

    # ---- Part 1: Single Forward Pass ----
    print(f"\n{'='*60}")
    print(f"  Part 1: Forward Pass")
    print(f"{'='*60}")

    prompt = ["hello", "world"]
    prompt_ids = np.array([VOCAB[t] for t in prompt])
    print(f"\n  Input: {prompt} → IDs: {prompt_ids.tolist()}")

    logits = model.forward(prompt_ids, tracer=tracer)
    print(f"\n  Logits shape: {logits.shape}")
    print(f"  Last position logits: {logits[-1][:6].round(3)}... (truncated)")

    probs = softmax(logits[-1])
    top5 = np.argsort(probs)[-5:][::-1]
    print(f"\n  Top-5 predictions after '{prompt[-1]}':")
    for idx in top5:
        bar = "█" * int(probs[idx] * 30)
        print(f"    {ID_TO_TOKEN[idx]:>8}: {probs[idx]:.4f} {bar}")

    predicted = int(np.argmax(probs))
    print(f"\n  → Predicted: '{ID_TO_TOKEN[predicted]}' (prob={probs[predicted]:.4f})")

    # ---- Part 2: Greedy Generation ----
    print(f"\n{'='*60}")
    print(f"  Part 2: Greedy Generation")
    print(f"{'='*60}")

    np.random.seed(42)  # reproducibility for weight init, not sampling
    gen_tracer = Tracer()

    prompt = ["the", "cat"]
    prompt_ids = np.array([VOCAB[t] for t in prompt])
    print(f"\n  Prompt: {prompt} → IDs: {prompt_ids.tolist()}")

    generated_ids = model.generate(prompt_ids, max_new_tokens=6, temperature=0.0, tracer=gen_tracer)
    generated_tokens = decode(generated_ids)

    print(f"\n  Generation trace:")
    print(f"    Step 0: {prompt}")
    for i in range(6):
        step = i + 1
        new_token = generated_tokens[len(prompt) + i]
        so_far = generated_tokens[:len(prompt) + step]
        print(f"    Step {step}: + '{new_token}' → {so_far}")

    print(f"\n  Final: {generated_tokens}")

    # ---- Part 3: Temperature Sampling ----
    print(f"\n{'='*60}")
    print(f"  Part 3: Temperature Sampling")
    print(f"{'='*60}")

    for temp in [0.0, 0.5, 1.0, 1.5]:
        np.random.seed(123)
        gen_ids = model.generate(prompt_ids, max_new_tokens=6, temperature=temp)
        tokens = decode(gen_ids)
        print(f"\n  T={temp:.1f}: {' '.join(tokens)}")

    # ---- Part 4: Top-k Sampling ----
    print(f"\n{'='*60}")
    print(f"  Part 4: Top-k Sampling (k=5, T=1.0)")
    print(f"{'='*60}")

    for trial in range(3):
        np.random.seed(trial * 100)
        gen_ids = model.generate(prompt_ids, max_new_tokens=6, temperature=1.0, top_k=5)
        tokens = decode(gen_ids)
        print(f"  Trial {trial+1}: {' '.join(tokens)}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Model Summary")
    print(f"  {model}")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Config: {config}")
    print(f"{'='*60}")

    # ---- Export Reports ----
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")

    path1 = os.path.abspath(os.path.join(report_dir, "level5_forward_pass.md"))
    tracer.export_markdown(path1, title="TraceGPT Level 5: Forward Pass Trace")
    print(f"\n📄 Forward pass report → {path1}")

    path2 = os.path.abspath(os.path.join(report_dir, "level5_generation.md"))
    gen_tracer.export_markdown(path2, title="TraceGPT Level 5: Greedy Generation Trace")
    print(f"📄 Generation report → {path2}")


if __name__ == "__main__":
    main()
