# Launch Posts for TraceGPT

## Hacker News: Show HN

**Title:** Show HN: TraceGPT – Trace a Transformer word by word, see what every number means

**Text:**
I built TraceGPT because I kept staring at Transformer tutorials full of number matrices and thinking "but what do these numbers actually MEAN?"

TraceGPT is different: every matrix has word labels, every attention weight shows which word cares about which, and every prediction traces back to meaning.

For example, when processing "the cat sat":
- The embedding matrix shows "cat" = high animal (0.9), concrete (0.8)
- The attention heatmap shows "sat" pays most attention to "cat" (0.40) — the model "knows" a cat is sitting!
- The prediction traces back to which words influenced it

Pure Python + NumPy, zero PyTorch. Interactive web demo (no install): https://yuanyuanma03.github.io/TraceGPT/

Also includes: 6 progressive learning levels, 7 common Transformer bugs with tests, a full TinyGPT model with autoregressive generation.

GitHub: https://github.com/YuanyuanMa03/TraceGPT

---

## Reddit: r/MachineLearning

**Title:** [P] TraceGPT — A Transformer where every number has a word label

**Text:**
Most Transformer tutorials show you this:

```
[[0.12, -0.34, 0.56],
 [0.78,  0.23, -0.91]]
```

But what if you could see this:

```
            animal  action  grammar
  "the"   │  0.10    0.00    0.90  │
  "cat"   │  0.90    0.10    0.00  │
  "sat"   │  0.10    0.90    0.00  │
```

TraceGPT is a NumPy-only Transformer where every matrix is labeled with words, every attention weight shows which word attends to which, and every prediction is traceable.

Live demo (works in browser, zero install): https://yuanyuanma03.github.io/TraceGPT/

Key differentiator from nanoGPT/picoGPT: word-labeled attention heatmaps where you can SEE "sat" paying attention to "cat" because it "knows" a cat is sitting.

Includes 6 progressive levels (embedding → attention → full GPT), 7 common Transformer bugs with wrong/correct + tests, and 87 unit tests.

GitHub: https://github.com/YuanyuanMa03/TraceGPT

---

## Reddit: r/learnmachinelearning

**Title:** I built an interactive Transformer you can trace word-by-word in your browser

**Text:**
I got frustrated that every Transformer tutorial shows you matrices of numbers without explaining what they mean in terms of actual language.

So I built TraceGPT — a Transformer where you can see:

1. **Word embeddings with meaning:** "cat" = high animal (0.9), concrete (0.8) — you can read what each dimension means
2. **Attention heatmaps with words:** See "sat" attend to "cat" (40%) — the model knows a cat is sitting!
3. **Predictions that trace back:** See which words influenced the final prediction

Try it right now in your browser (no install): https://yuanyuanma03.github.io/TraceGPT/

It's pure NumPy, no PyTorch. Every step is hand-verifiable. There are also 6 progressive learning levels if you want to build up from scratch.

GitHub: https://github.com/YuanyuanMa03/TraceGPT

---

## Twitter/X Thread

🧵 I traced one sentence through a Transformer and showed every number.

You've seen nanoGPT. You've seen picoGPT.

But have you seen a Transformer where every number has a WORD LABEL? 👇

1/7

Here's "the cat sat" becoming vectors:
            animal  action  grammar
  "the"   │  0.10    0.00    0.90  │  ← grammar word!
  "cat"   │  0.90    0.10    0.00  │  ← animal!
  "sat"   │  0.10    0.90    0.00  │  ← action!

Every dimension has a NAME.

2/7

Now attention — which word cares about which?

           the     cat     sat
  the  │  1.00    0.00    0.00  │
  cat  │  0.38    0.62    0.00  │
  sat  │  0.29    0.40    0.31  │  ← "sat" looks at "cat" most!

The model KNOWS a cat is sitting 🤯

3/7

This is TraceGPT — a Transformer where every matrix has word labels, every attention weight shows which word cares about which.

Pure NumPy. Zero PyTorch. Try it in your browser: https://yuanyuanma03.github.io/TraceGPT/

4/7

6 progressive levels take you from embedding → attention → full GPT.

Plus 7 common Transformer bugs (with wrong/correct code + tests).

87 unit tests, all passing.

5/7

What's different from nanoGPT/picoGPT?

- nanoGPT: PyTorch, training focused
- picoGPT: tiny, but no visualization
- TraceGPT: WORD-LABELED MATRICES + attention heatmaps + interactive demo

Every number has a story.

6/7

If you've ever stared at a Transformer and thought "but what do these numbers actually MEAN?" —

Star TraceGPT on GitHub ⭐:
https://github.com/YuanyuanMa03/TraceGPT

Try the interactive demo:
https://yuanyuanma03.github.io/TraceGPT/

7/7
