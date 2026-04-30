# TraceGPT

**Every number has a story.**

TraceGPT is a Transformer learning framework where you can trace *every single computation* from words to prediction — with word labels, attention heatmaps, and hand-verifiable math. Pure Python + NumPy, zero PyTorch.

```
"the cat sat" → embeddings → attention → prediction → "on"
     ↑              ↑           ↑            ↑
   words       word-labeled  word-labeled  word-labeled
               matrices      heatmaps      probabilities
```

## The "Aha!" Demo

```bash
python -m levels.level6_showcase
```

Output:

```
╔══════════════════════════════════════════════════════════╗
║  TraceGPT Showcase: One Sentence, Fully Traced          ║
║  "the cat sat" → What comes next?                      ║
╚══════════════════════════════════════════════════════════╝

  Step 2: Word Embeddings — Words Become Numbers

            animal    action  location      size   emotion   grammar      time  concrete
  "the" │     0.100     0.000     0.000     0.000     0.000     0.900     0.000     0.000
  "cat" │     0.900     0.100     0.000     0.300     0.200     0.000     0.000     0.800
  "sat" │     0.100     0.900     0.100     0.000     0.000     0.000     0.300     0.100

  💡 "cat" has high animal (0.9) and concrete (0.8)
     "sat" has high action (0.9) — it's a verb!
     "the" is mostly grammar (0.9)

  Step 6: Attention Weights — "the cat sat"

           the     cat     sat
   ────────────────────────
   the │█████                    │ 1.00  0.00  0.00
   cat │░░░░░   ▓▓▓▓▓            │ 0.38  0.62  0.00
   sat │░░░░░   ▒▒▒▒▒   ░░░░░    │ 0.29  0.40  0.31

  💡 "cat" pays most attention to itself (0.62)
     "sat" looks at "cat" the most (0.40) — it "knows" a cat sat!

  🎯 Prediction: "dog" (prob=0.26)
```

**Every matrix has word labels. Every attention weight tells a story.**

## Why TraceGPT?

nanoGPT and minGPT give you working code. TraceGPT gives you *understanding*:

- **Words, not just numbers.** Every matrix is labeled with actual tokens.
- **Attention heatmaps.** See which word attends to which — with words on the axes.
- **No PyTorch.** Pure NumPy. Every operation is transparent.
- **Hand-verifiable.** Tiny matrices you can check with a calculator.
- **Bug library.** 7 common Transformer bugs with wrong/correct + tests.
- **Full GPT model.** Complete TinyGPT with multi-head attention and generation.

## Quick Start

```bash
git clone https://github.com/YuanyuanMa03/TraceGPT.git
cd TraceGPT
pip install -e .

# Run Level 0: Next Token Prediction
python -m levels.level0_next_token

# Run Level 1: Causal Self-Attention
python -m levels.level1_causal_attention

# Run Level 2: Transformer Block
python -m levels.level2_transformer_block

# Run all tests
pytest tests/ bugs/ -v
```

## Project Structure

```
TraceGPT/
├── README.md
├── LICENSE                    # MIT
├── requirements.txt
├── pyproject.toml
├── tracegpt/                  # Core library
│   ├── __init__.py
│   ├── tracer.py              # TraceUnit + Tracer: records every operation
│   ├── ops.py                 # Core ops: softmax, causal_mask, layer_norm, linear, relu
│   ├── report.py              # Markdown report generation
│   └── utils.py               # Helper functions
├── levels/                    # Progressive learning levels
│   ├── level0_next_token.py   # Embedding → Projection → Softmax → Prediction
│   ├── level1_causal_attention.py  # Q/K/V, Attention, Causal Mask
│   ├── level2_transformer_block.py  # Full Transformer block
│   ├── level3_positional_encoding.py  # Sinusoidal PE: why and how
│   ├── level4_multihead_gpt.py    # Multi-head attention + tiny GPT forward pass
│   ├── level5_full_gpt.py        # Complete GPT model + autoregressive generation
│   └── level6_showcase.py        # ⭐ The killer demo: one sentence, fully traced with words
├── bugs/                      # Common bugs with wrong/correct + tests
│   ├── 001_softmax_wrong_axis/
│   ├── 002_causal_mask_reversed/
│   ├── 003_missing_sqrt_dk/
│   ├── 004_wrong_qk_transpose/
│   └── 005_label_shift_bug/
├── tests/                     # pytest test suite
├── reports/                   # Generated Markdown reports
└── paper/                     # Paper draft (coming soon)
```

## Learning Path

### Level 0: Next Token Prediction
The simplest possible language model: look up an embedding, project to vocab logits, softmax, predict. This teaches the fundamental "prediction head" before any attention.

### Level 1: Causal Self-Attention
Build single-head causal self-attention from scratch: Q, K, V projections, attention scores, scaling, causal masking, softmax, weighted sum.

### Level 2: Transformer Block
A complete Transformer block: attention + residual + layer norm + feed-forward network + residual + layer norm.

### Level 0-4: Building Up Step by Step
Progressive levels that build from embedding to full GPT.

### Level 5: Full GPT Model
Complete TinyGPT forward pass + autoregressive generation.

### Level 6: ⭐ The Showcase — "One Sentence, Fully Traced"
**This is TraceGPT's killer demo.** Trace "the cat sat" through every operation
with word labels, attention heatmaps, and human-readable explanations.
Every number has a story.

## Bug Library

Each bug comes with:
- `wrong.py` — the buggy implementation
- `correct.py` — the fix
- `README.md` — explanation of the bug and how to detect it
- `test_*.py` — tests that catch the bug

| Bug | Description |
|-----|------------|
| 001 | Softmax on wrong axis |
| 002 | Causal mask reversed (upper instead of lower triangular) |
| 003 | Missing √d_k scaling in attention |
| 004 | Q·K^T computed as K·Q^T (wrong transpose) |
| 005 | Label not shifted for next-token prediction |
| 006 | Weight tying transpose error |
| 007 | Generation loop doesn't truncate to max_seq_len |

## Philosophy

1. **Readability over performance.** No optimizations that obscure understanding.
2. **Traced, not hidden.** Every operation is recorded and explainable.
3. **Hand-verifiable.** Tiny matrices you can check with a calculator.
4. **Bugs are lessons.** Common mistakes are first-class citizens.
5. **No magic.** No framework abstractions between you and the math.

## Design Principles

- **Python + NumPy only.** No PyTorch, TensorFlow, or JAX.
- **No performance optimization.** Clarity is the only metric.
- **Every op exposes:** formula, inputs, output, shape, explanation.
- **All examples use tiny matrices** (typically 3×4 or smaller).

## Example Output

Running Level 1 produces a Markdown report like:

```markdown
## Step 4: causal_mask

**Formula:** `M[i][j] = 1 if j ≤ i, else 0`

**Explanation:** Lower-triangular mask: each token can only attend to
itself and earlier tokens.

### Shapes
- **seq_len**: ()
- **output**: (3, 3)

### Output
```
[[1. 0. 0.]
 [1. 1. 0.]
 [1. 1. 1.]]
```
```

## Roadmap

- [x] v0.1: Core tracer, ops, 3 levels, 5 bugs, tests
- [x] v0.2: Multi-head attention, positional encoding
- [x] v0.3: Full GPT-2 forward pass (tiny model)
- [ ] v0.3: Full GPT-2 forward pass (tiny model)
- [ ] v0.4: Training loop with backprop (educational)
- [ ] v0.5: Interactive Jupyter notebooks
- [ ] v1.0: Paper, documentation website, CI/CD

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

Contributions welcome! Please:
1. Follow the existing code style (no PyTorch, tiny matrices, full traces)
2. Add tests for new operations
3. Add bug entries for common mistakes
4. Keep examples hand-verifiable
