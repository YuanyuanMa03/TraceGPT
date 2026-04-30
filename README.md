# <img src="https://yuanyuanma03.github.io/TraceGPT/banner.svg" width="100%" alt="TraceGPT Banner">

**I traced one sentence through a Transformer and showed every number.**

You've seen [nanoGPT](https://github.com/karpathy/nanoGPT). You've seen [minGPT](https://github.com/karpathy/minGPT). You've even seen [picoGPT](https://github.com/jaymody/picoGPT).

But have you seen every number in a Transformer connected to the **actual words**? 🤔

TraceGPT is a Transformer where every matrix has word labels, every attention weight shows which word cares about which, and every prediction traces back to meaning. Pure Python + NumPy, zero PyTorch.

```
"the cat sat" → embeddings → attention → prediction → "on"
     ↑              ↑           ↑            ↑
   words       "cat"=0.9   "cat"→"sat"  "on"=0.45
               animal      attention=0.40
```

**What if every number in a Transformer had a name?**

## 🌐 Try It Live

**👉 [Interactive Demo](https://yuanyuanma03.github.io/TraceGPT/)** — Trace a sentence in your browser, no install needed.

## The "Aha! Demo"

```bash
python -m levels.level6_showcase
```

**Key Output:**

### Word Embeddings
```
            animal    action  location      size   emotion   grammar      time  concrete
  "the" │     0.100     0.000     0.000     0.000     0.000     0.900     0.000     0.000 │
  "cat" │     0.900     0.100     0.000     0.300     0.200     0.000     0.000     0.800 │
  "sat" │     0.100     0.900     0.100     0.000     0.000     0.000     0.300     0.100 │
```

**💡 Reading this:**
- "cat" = high animal (0.9) + concrete (0.8) — it's an animal!
- "sat" = high action (0.9) — it's a verb!
- "the" = high grammar (0.9) — it's a grammar word!

### Attention Weights
```
Attention Weights: "the cat sat"

           the     cat     sat
  ────────────────────────
  the  │█████                    │ 1.00  0.00  0.00  ← "the" sees only itself
  cat  │░░░░░   ▓▓▓▓▓            │ 0.38  0.62  0.00  ← "cat" looks at itself most
  sat  │░░░░░   ▒▒▒▒▒   ░░░░░    │ 0.29  0.40  0.31  ← "sat" looks at "cat"! 🤯
```

**💡 The model "knows" a cat is sitting!** "sat" pays most attention to "cat" (40%).

### Prediction
```
After "the cat sat" → What comes next?

     dog  ███████░░░░░░░░░░░░░░░░░░░░░░░  0.2585 ← predicted
     cat  ████░░░░░░░░░░░░░░░░░░░░░░░░░░  0.1516
    park  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.1173
     mat  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.1027
```

The last word's output is most similar to **"dog"** (1.7305) — both are animals!

📄 Full trace: [reports/level6_showcase.md](reports/level6_showcase.md)

[**Try the demo →**](https://yuanyuanma03.github.io/TraceGPT/)

## Why TraceGPT?

nanoGPT gives you working code. picoGPT gives you tiny code. TraceGPT gives you **understanding**:

- **Words, not just numbers.** Every matrix is labeled with actual tokens. You see "cat" = high animal, not just `[0.9, 0.1, 0, ...]`.
- **Attention heatmaps with words.** See which word attends to which — "sat" looks at "cat" the most.
- **No PyTorch.** Pure NumPy. Every operation is transparent.
- **Interactive web demo.** Try it in your browser, zero install.
- **Hand-verifiable.** Tiny matrices you can check with a calculator.
- **Bug library.** 7 common Transformer bugs with wrong/correct implementations and tests.
- **Full GPT model.** Complete TinyGPT with multi-head attention and autoregressive generation.

TraceGPT is for anyone who has stared at a matrix multiplication and thought: *"but what do these numbers actually MEAN?"*

## Quick Start

```bash
git clone https://github.com/YuanyuanMa03/TraceGPT.git
cd TraceGPT
pip install -e .

# Run Level 6 (showcase)
python -m levels.level6_showcase

# Run all tests
pytest tests/ bugs/ -v
```

## Project Structure

```
TraceGPT/
├── tracegpt/          # Core library (pure NumPy)
│   ├── tracer.py      # Traces every operation
│   ├── ops.py         # Core ops with explanations
│   └── report.py      # Markdown report generation
├── levels/            # 6 progressive learning levels
│   ├── level0...6     # From embedding → full GPT
│   └── level6_showcase.py  # ⭐ The killer demo
├── bugs/              # 7 common Transformer bugs
└── tests/             # 87 tests, all passing
```

## Learning Path

- **Level 0:** Embedding → prediction
- **Level 1:** Causal self-attention
- **Level 2:** Transformer block
- **Level 3:** Positional encoding
- **Level 4:** Multi-head attention
- **Level 5:** Full GPT + generation
- **Level 6:** ⭐ **Showcase** — word-level traces

## What's Different?

| Feature | nanoGPT | picoGPT | **TraceGPT** |
|---------|---------|---------|-------------|
| Pure NumPy | ❌ PyTorch | ✅ | ✅ |
| Word-labeled matrices | ❌ | ❌ | ✅ |
| Attention heatmaps with words | ❌ | ❌ | ✅ |
| Interactive demo | ❌ | ❌ | ✅ |
| Bug library | ❌ | ❌ | ✅ |
| Hand-verifiable | ✅ | ✅ | ✅ |

## Bug Library

7 common Transformer bugs with wrong/correct code + tests:
- Softmax on wrong axis
- Causal mask reversed
- Missing √d_k scaling
- Wrong Q·K transpose
- Label shift bug
- Weight tying transpose error
- Generation loop truncation

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

## Roadmap

- [x] v0.4: Word-level traces, attention heatmaps, interactive demo
- [ ] v0.5: Training loop with backprop (educational)
- [ ] v1.0: Paper, documentation website

## License

MIT License. See [LICENSE](LICENSE).

## ⭐ Star on GitHub

If you've ever stared at a Transformer and thought *"but what do these numbers actually MEAN?"* —

**[Star TraceGPT →](https://github.com/YuanyuanMa03/TraceGPT)**
