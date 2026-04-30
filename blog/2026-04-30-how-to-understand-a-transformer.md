# How to Understand a Transformer, One Word at a Time

*Or: What if every number in a Transformer had a name?*

---

You've probably seen the "Attention is All You Need" paper. You may have even read Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) or [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) video.

But here's the thing: **every tutorial shows you matrices of numbers, and nobody tells you what those numbers mean in terms of actual language.**

You see:
```
[[0.12, -0.34, 0.56, ...],
 [0.78,  0.23, -0.91, ...],
 ...]
```

But what you *want* to see is:
```
            animal  action  location  grammar
  "the"   │  0.10    0.00    0.00     0.90   │  ← it's a grammar word!
  "cat"   │  0.90    0.10    0.00     0.00   │  ← it's an animal!
  "sat"   │  0.10    0.90    0.10     0.00   │  ← it's an action!
```

**Every number tells a story. You just need the labels.**

That's why I built [TraceGPT](https://github.com/YuanyuanMa03/TraceGPT).

## What is TraceGPT?

TraceGPT is a Transformer learning framework where you can trace **every single computation** from input words to output prediction — with word labels on every matrix, attention heatmaps showing which word cares about which, and hand-verifiable math.

It's pure Python + NumPy. Zero PyTorch. Zero magic.

[Try it in your browser right now →](https://yuanyuanma03.github.io/TraceGPT/)

## The "Aha!" Moment

Let me walk you through what happens when TraceGPT processes the sentence **"the cat sat"** and tries to predict what comes next.

### Step 1: Words Become Vectors

Each word gets an embedding — a fingerprint that captures its meaning:

```
            animal  action  location  size  emotion  grammar  time  concrete
  "the"   │  0.10    0.00    0.00    0.00   0.00     0.90    0.00   0.00   │
  "cat"   │  0.90    0.10    0.00    0.30   0.20     0.00    0.00   0.80   │
  "sat"   │  0.10    0.90    0.10    0.00   0.00     0.00    0.30   0.10   │
```

💡 **"cat" has high animal (0.9) and concrete (0.8).** This vector IS what the model knows about "cat".

### Step 2: Words Ask Questions

Each word creates a Query: "What am I looking for?"

- **"the"** asks: "I'm a grammar word — what's the main noun?"
- **"cat"** asks: "I'm an animal — is anything happening to me?"
- **"sat"** asks: "I'm an action — who's doing me?"

### Step 3: Attention — Who Cares About Whom?

```
Attention Weights: "the cat sat"

           the     cat     sat
  ──────────────────────────────
  the  │  1.00    0.00    0.00  │  ← "the" can only see itself
  cat  │  0.38    0.62    0.00  │  ← "cat" looks at itself most
  sat  │  0.29    0.40    0.31  │  ← "sat" looks at "cat" most!
```

💡 **"sat" pays the most attention to "cat" (0.40)** — the model "knows" that a cat is the one doing the sitting!

This is the magic of self-attention. Nobody programmed this. The model learned it from the structure of language.

### Step 4: Prediction

After attention, each word's vector is now a *blend* of all the words it attended to. "sat" now carries information about "cat" — it "knows" a cat sat.

We compare this blended vector against all words in the vocabulary:

```
  After "the cat sat" → What comes next?

     dog   ████████████████████░░░░░░░░░░  25.9%  ← predicted
     cat   ████████████░░░░░░░░░░░░░░░░░░  15.2%
     park  █████████░░░░░░░░░░░░░░░░░░░░░  11.7%
     mat   ████████░░░░░░░░░░░░░░░░░░░░░░  10.3%
```

The model predicts "dog"! Why? Because the blended "sat" vector is most similar to "dog" in the embedding space — both are animals, both are concrete, similar size.

(With real trained weights, it would predict "on" or "down". We're using hand-crafted embeddings here for clarity.)

## Why Does This Matter?

Most people learn Transformers like this:

1. Read the paper → confused
2. Read a blog post → still confused
3. Look at PyTorch code → `torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)` → very confused
4. Give up and just use the API

TraceGPT exists because **understanding should not require giving up**. If every number has a word label, if every matrix tells a story, if you can verify each step with a calculator — then anyone can truly understand Transformers.

## How is This Different from nanoGPT / picoGPT?

| Feature | nanoGPT | picoGPT | **TraceGPT** |
|---------|---------|---------|-------------|
| Pure NumPy | ❌ PyTorch | ✅ | ✅ |
| Word-labeled matrices | ❌ | ❌ | ✅ |
| Attention heatmaps | ❌ | ❌ | ✅ |
| Interactive web demo | ❌ | ❌ | ✅ |
| Bug library | ❌ | ❌ | ✅ |
| Progressive levels | ❌ | ❌ | ✅ |
| Full GPT model | ✅ | ✅ | ✅ |

## Try It

**Browser (zero install):** [yuanyuanma03.github.io/TraceGPT](https://yuanyuanma03.github.io/TraceGPT/)

**Local:**
```bash
git clone https://github.com/YuanyuanMa03/TraceGPT.git
cd TraceGPT
python -m levels.level6_showcase  # The killer demo
python -m levels.level0_next_token  # Start from scratch
pytest tests/  # 87 tests, all passing
```

## What's Next?

TraceGPT is early (v0.4). Here's the roadmap:

- [x] Word-labeled traces with attention heatmaps
- [x] Interactive web demo
- [x] Full TinyGPT model with generation
- [x] 7 common Transformer bugs
- [ ] Training loop with backprop (educational)
- [ ] Jupyter notebooks (Colab-ready)
- [ ] Paper

If you've ever stared at a Transformer and thought *"but what do these numbers actually MEAN?"* — give TraceGPT a ⭐.

**[Star on GitHub →](https://github.com/YuanyuanMa03/TraceGPT)**

---

*TraceGPT is built by [Yuanyuan Ma](https://github.com/YuanyuanMa03), a graduate student at Nanjing Agricultural University studying Agricultural AI and Multi-Agent systems.*
