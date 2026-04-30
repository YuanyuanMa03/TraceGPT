# TraceGPT

**A calculator-verifiable Transformer learning framework.**

TraceGPT turns every hidden tensor operation in GPT-style models into a readable, auditable, and reproducible trace.

```
formula → hand calculation → NumPy code → tensor trace → Markdown report
```

## Why TraceGPT?

Most Transformer tutorials give you PyTorch code that works, but you can't *see* what's happening inside. TraceGPT is different:

- **No PyTorch.** Pure Python + NumPy. Every operation is transparent.
- **Tiny matrices.** Every example uses 3×4 matrices you can verify by hand.
- **Full traces.** Every step records its formula, inputs, outputs, shapes, and explanation.
- **Bug library.** Common Transformer bugs with wrong/correct implementations and tests.
- **Markdown reports.** Beautiful, human-readable execution reports.

TraceGPT is for students, researchers, and anyone who wants to *truly understand* Transformers — not just use them.

## Quick Start

```bash
# Clone
git clone https://github.com/your-username/TraceGPT.git
cd TraceGPT

# Install (no dependencies beyond NumPy)
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
│   └── level2_transformer_block.py  # Full Transformer block
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
- [ ] v0.2: Multi-head attention, positional encoding
- [ ] v0.3: Full GPT-2 forward pass (tiny model)
- [ ] v0.4: Training loop with backprop (educational)
- [ ] v0.5: Interactive Jupyter notebooks
- [ ] v1.0: Paper, documentation website, CI/CD

## Citation

If you use TraceGPT in your research or teaching, please cite:

```bibtex
@software{tracegpt2026,
  title = {TraceGPT: A Calculator-Verifiable Transformer Learning Framework},
  author = {Yuanyuan Ma},
  year = {2026},
  url = {https://github.com/your-username/TraceGPT}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

Contributions welcome! Please:
1. Follow the existing code style (no PyTorch, tiny matrices, full traces)
2. Add tests for new operations
3. Add bug entries for common mistakes
4. Keep examples hand-verifiable
