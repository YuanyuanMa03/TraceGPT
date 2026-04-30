"""
Level 6: The Showcase — "One Sentence, Fully Traced"

This is TraceGPT's killer demo. We trace the sentence
"the cat sat on the mat" through every single operation,
showing how each word's meaning transforms at each step.

Every number has a story. Every matrix has word labels.
Every attention weight shows which word cares about which.

This is what makes TraceGPT different from nanoGPT, minGPT,
and every other tutorial: you can SEE and UNDERSTAND every
computation, connected to actual language.

Run:
    python -m levels.level6_showcase
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracegpt.tracer import Tracer
from tracegpt.model import TinyGPT, GPTConfig
from tracegpt.ops import softmax, causal_mask
from tracegpt.viz import (
    attention_heatmap,
    labeled_matrix,
    probability_bar,
    word_level_trace,
    generation_trace,
)


# ── A real sentence vocabulary ──────────────────────────────
VOCAB = [
    "<PAD>", "<EOS>", "the", "cat", "sat", "on", "mat",
    "is", "happy", "a", "dog", "ran", "to", "park", "big", "small",
]
VOCAB_MAP = {w: i for i, w in enumerate(VOCAB)}
DIM_NAMES = ["animal", "action", "location", "size", "emotion", "grammar", "time", "concrete"]

SENTENCE = ["the", "cat", "sat"]


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  TraceGPT Showcase: One Sentence, Fully Traced          ║")
    print("║  \"the cat sat\" → What comes next?                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Step 1: The Words ──────────────────────────────────────
    print("═" * 60)
    print("  Step 1: The Words")
    print("═" * 60)
    print()
    print(f'  Input sentence: "{" ".join(SENTENCE)}"')
    print(f"  Token IDs: {[VOCAB_MAP[w] for w in SENTENCE]}")
    print()
    print('  Question: After "the cat sat", what word comes next?')
    print('  A human might guess "on" or "down" or "there".')
    print("  Let's trace how a Transformer figures it out.")
    print()

    # ── Step 2: Word Embeddings ────────────────────────────────
    print("═" * 60)
    print("  Step 2: Word Embeddings — Words Become Numbers")
    print("═" * 60)
    print()
    print('  Each word gets a vector. Think of it as a "fingerprint".')
    print('  Each dimension captures a different aspect of meaning:')
    print()

    # Hand-crafted embeddings for educational clarity
    # Each row is a word's embedding, each column is a semantic dimension
    embeddings = np.array([
        #     animal  action  location  size  emotion  grammar  time  concrete
        [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # <PAD>
        [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # <EOS>
        [0.10,  0.00,  0.00,  0.00,  0.00,  0.90,  0.00,  0.00],  # "the" — grammar word
        [0.90,  0.10,  0.00,  0.30,  0.20,  0.00,  0.00,  0.80],  # "cat" — animal, small, concrete
        [0.10,  0.90,  0.10,  0.00,  0.00,  0.00,  0.30,  0.10],  # "sat" — action, past
        [0.00,  0.10,  0.10,  0.00,  0.00,  0.90,  0.00,  0.00],  # "on" — grammar, location-ish
        [0.00,  0.00,  0.90,  0.70,  0.00,  0.00,  0.00,  0.90],  # "mat" — location, big, concrete
        [0.10,  0.00,  0.00,  0.00,  0.80,  0.00,  0.00,  0.00],  # "is"
        [0.10,  0.00,  0.00,  0.00,  0.90,  0.00,  0.00,  0.10],  # "happy" — emotion
        [0.10,  0.00,  0.00,  0.00,  0.00,  0.90,  0.00,  0.00],  # "a" — grammar
        [0.90,  0.10,  0.00,  0.70,  0.20,  0.00,  0.00,  0.80],  # "dog" — animal, big, concrete
        [0.20,  0.90,  0.20,  0.00,  0.00,  0.00,  0.50,  0.10],  # "ran" — action, past
        [0.00,  0.10,  0.10,  0.00,  0.00,  0.90,  0.00,  0.00],  # "to" — grammar
        [0.00,  0.00,  0.90,  0.80,  0.00,  0.00,  0.00,  0.90],  # "park" — location, big
        [0.00,  0.00,  0.00,  0.90,  0.00,  0.00,  0.00,  0.80],  # "big" — size
        [0.00,  0.00,  0.00,  0.20,  0.00,  0.00,  0.00,  0.80],  # "small" — size
    ])

    token_ids = [VOCAB_MAP[w] for w in SENTENCE]
    word_vecs = embeddings[token_ids]

    print(word_level_trace(
        tokens=SENTENCE,
        vectors=word_vecs,
        dim_names=DIM_NAMES,
        title="Word Embeddings (hand-crafted for clarity)",
    ))
    print()
    print('  💡 Reading this: "cat" has high values in animal (0.9) and concrete (0.8)')
    print('     "sat" has high values in action (0.9) and time (0.3, past tense)')
    print('     "the" is mostly grammar (0.9)')
    print()

    # ── Step 3: Positional Encoding ────────────────────────────
    print("═" * 60)
    print("  Step 3: Positional Encoding — Words Get an Address")
    print("═" * 60)
    print()
    print('  "cat sat the" and "the cat sat" have the same words')
    print('  but different meanings. Positional encoding tells the model')
    print('  WHERE each word sits in the sentence.')
    print()

    from tracegpt.ops import sinusoidal_position_encoding

    pe = sinusoidal_position_encoding(len(SENTENCE), 8)
    vecs_with_pe = word_vecs + pe

    print(word_level_trace(
        tokens=SENTENCE,
        vectors=pe,
        dim_names=DIM_NAMES,
        title="Positional Encoding (unique address for each position)",
    ))
    print()
    print(word_level_trace(
        tokens=SENTENCE,
        vectors=vecs_with_pe,
        dim_names=DIM_NAMES,
        title="Word Vectors + Position = Full Input",
    ))
    print()

    # ── Step 4: Computing Q, K, V ──────────────────────────────
    print("═" * 60)
    print("  Step 4: Query, Key, Value — What Does Each Word Want/Know?")
    print("═" * 60)
    print()
    print('  Three questions for each word:')
    print('    Query (Q): "What am I looking for?"')
    print('    Key (K):   "What do I contain?"')
    print('    Value (V): "What information do I offer?"')
    print()

    # Simple projection: mix the semantic dimensions
    W_Q = np.eye(8) * 0.5 + np.roll(np.eye(8), 1, axis=1) * 0.3
    W_K = np.eye(8) * 0.4 + np.roll(np.eye(8), -1, axis=1) * 0.3
    W_V = np.eye(8) * 0.6

    Q = vecs_with_pe @ W_Q
    K = vecs_with_pe @ W_K
    V = vecs_with_pe @ W_V

    print(word_level_trace(tokens=SENTENCE, vectors=Q, dim_names=DIM_NAMES,
                           title='Queries — "What am I looking for?"'))
    print()
    print(word_level_trace(tokens=SENTENCE, vectors=K, dim_names=DIM_NAMES,
                           title='Keys — "What do I contain?"'))
    print()

    # ── Step 5: Attention Scores ────────────────────────────────
    print("═" * 60)
    print("  Step 5: Attention Scores — How Much Does Each Word Care?")
    print("═" * 60)
    print()
    print('  Attention score = how well Query matches Key')
    print('  "the" asks its question, and "cat"/"sat" answer...')
    print()

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    print(labeled_matrix(
        row_labels=[f'"{t}" wants' for t in SENTENCE],
        col_labels=[f'"{t}" has' for t in SENTENCE],
        matrix=scores,
        title="Raw Attention Scores (before mask & softmax)",
    ))
    print()

    # ── Step 6: Causal Mask ─────────────────────────────────────
    print("═" * 60)
    print("  Step 6: Causal Mask — No Peeking Into the Future!")
    print("═" * 60)
    print()
    print('  "the" can only see "the" (position 0)')
    print('  "cat" can see "the" and "cat" (positions 0-1)')
    print('  "sat" can see all three (positions 0-2)')
    print()
    print('  This is what makes GPT autoregressive:')
    print('  each word only knows what came BEFORE it.')
    print()

    mask = causal_mask(len(SENTENCE))
    print(labeled_matrix(
        row_labels=[f'"{t}"' for t in SENTENCE],
        col_labels=[f'"{t}"' for t in SENTENCE],
        matrix=mask,
        title="Causal Mask (1=can attend, 0=blocked)",
        precision=0,
    ))
    print()

    masked_scores = scores + (1 - mask) * (-1e9)
    attn_weights = softmax(masked_scores)

    print(attention_heatmap(SENTENCE, attn_weights,
                           title="Attention Weights (after mask + softmax)"))
    print()
    print('  💡 Reading this:')
    print('     Row 1 ("the"): only attends to itself (1.00) — it\'s first!')
    print('     Row 2 ("cat"): attends to both "the" and "cat"')
    print('     Row 3 ("sat"): attends to all three words')
    print()
    print('  The higher the number, the more "cat" pays attention')
    print('  to that word when building its understanding.')
    print()

    # ── Step 7: Attention Output ────────────────────────────────
    print("═" * 60)
    print("  Step 7: Attention Output — Words Gain Context")
    print("═" * 60)
    print()
    print('  Each word\'s output is a BLEND of all words it attends to.')
    print('  "sat" becomes a mix of "the" + "cat" + "sat" —')
    print('  now it "knows" a cat is involved!')
    print()

    attn_output = attn_weights @ V

    print(word_level_trace(
        tokens=SENTENCE,
        vectors=attn_output,
        dim_names=DIM_NAMES,
        title="Attention Output — each word now carries context from other words",
    ))
    print()

    # ── Step 8: Prediction ──────────────────────────────────────
    print("═" * 60)
    print("  Step 8: Prediction — What Word Comes Next?")
    print("═" * 60)
    print()

    # Simple projection to vocab (use embedding similarity)
    last_vec = attn_output[-1:]  # "sat"'s contextualized vector
    logits = (last_vec @ embeddings.T)[0]  # similarity with all words
    probs = softmax(logits / 0.5)  # temperature for sharper distribution

    predicted_id = int(np.argmax(probs))
    predicted_word = VOCAB[predicted_id]

    print(probability_bar(
        tokens=VOCAB,
        probs=probs,
        title=f'After "the cat sat" → What comes next?',
        highlight=predicted_id,
    ))
    print()
    print(f'  🎯 The model predicts: "{predicted_word}" (probability={probs[predicted_id]:.4f})')
    print()

    # ── Step 9: Why? ────────────────────────────────────────────
    print("═" * 60)
    print("  Step 9: Why This Prediction?")
    print("═" * 60)
    print()
    print(f'  The last word\'s attention output (for "sat") is most similar')
    print(f'  to the embedding of "{predicted_word}" in our vocabulary.')
    print()
    print('  Let\'s see the similarity scores:')
    print()

    for i, word in enumerate(VOCAB):
        sim = float(logits[i])
        if abs(sim) > 0.01:
            marker = " ← WINNER" if i == predicted_id else ""
            print(f'    "sat" output vs "{word:>8}": {sim:+.4f}{marker}')

    print()
    print('  ┌──────────────────────────────────────────────────────┐')
    print('  │  Full trace complete!                                 │')
    print(f'  │  "the cat sat" → "{predicted_word}"                         │')
    print('  │                                                        │')
    print('  │  Every number above is connected to a word.            │')
    print('  │  Every operation is visible and verifiable.            │')
    print('  │  This is TraceGPT: formula → trace → understanding    │')
    print('  └──────────────────────────────────────────────────────┘')
    print()

    # ── Export Report ───────────────────────────────────────────
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "level6_showcase.md")
    report_path = os.path.abspath(report_path)

    tracer = Tracer()
    tracer.trace("showcase", "Complete trace of 'the cat sat'", {"tokens": np.array(SENTENCE)}, attn_output,
                 "Showcase: every number has a story")
    tracer.export_markdown(report_path, title="TraceGPT Showcase: One Sentence, Fully Traced")
    print(f"📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
