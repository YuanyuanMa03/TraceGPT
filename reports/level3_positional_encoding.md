# TraceGPT Level 3: Sinusoidal Positional Encoding

**Generated:** 2026-04-30 14:56:58
**Total operations:** 4

---

## Step 1: input_without_position

**Formula:** `X ∈ R^{seq_len × d_model} (no position info)`

**Explanation:** Raw token embeddings contain no position information. Self-attention treats them as a SET, not a sequence. We need to INJECT position information.

### Shapes

- **tokens**: (4,)
- **output**: (4, 4)

### Inputs

**tokens:**
```
['I', 'love', 'AI', '!']
```

### Output

```
[
  [1.000000, 0.000000, 1.000000, 0.000000],
  [0.000000, 1.000000, 0.000000, 1.000000],
  [1.000000, 1.000000, 0.000000, 1.000000],
  [0.000000, 0.000000, 1.000000, 1.000000]
]
```

---

## Step 2: sinusoidal_pe

**Formula:** `PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d))`

**Explanation:** Each position gets a unique sinusoidal encoding. Even indices use sin, odd indices use cos. The frequencies decrease across dimensions, so each dimension corresponds to a different scale of position.

### Shapes

- **seq_len**: ()
- **d_model**: ()
- **output**: (4, 4)

### Inputs

**seq_len:**
```
4.0
```

**d_model:**
```
4.0
```

### Output

```
[
  [0.000000, 1.000000, 0.000000, 1.000000],
  [0.841471, 0.540302, 0.010000, 0.999950],
  [0.909297, -0.416147, 0.019999, 0.999800],
  [0.141120, -0.989992, 0.029996, 0.999550]
]
```

---

## Step 3: verify_uniqueness

**Formula:** `PE[i] ≠ PE[j] for all i ≠ j`

**Explanation:** Verify that each position gets a DIFFERENT encoding vector. This is essential — if two positions had the same encoding, the model couldn't distinguish them.

### Shapes

- **PE**: (4, 4)
- **output**: (6,)

### Inputs

**PE:**
```
[
  [0.000000, 1.000000, 0.000000, 1.000000],
  [0.841471, 0.540302, 0.010000, 0.999950],
  [0.909297, -0.416147, 0.019999, 0.999800],
  [0.141120, -0.989992, 0.029996, 0.999550]
]
```

### Output

```
[1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
```

---

## Step 4: add_pe_to_embeddings

**Formula:** `X_pe = X + PE`

**Explanation:** Add positional encodings to token embeddings. Now each vector encodes BOTH what the token is AND where it is. This is the standard approach from the original Transformer paper.

### Shapes

- **X**: (4, 4)
- **PE**: (4, 4)
- **output**: (4, 4)

### Inputs

**X:**
```
[
  [1.000000, 0.000000, 1.000000, 0.000000],
  [0.000000, 1.000000, 0.000000, 1.000000],
  [1.000000, 1.000000, 0.000000, 1.000000],
  [0.000000, 0.000000, 1.000000, 1.000000]
]
```

**PE:**
```
[
  [0.000000, 1.000000, 0.000000, 1.000000],
  [0.841471, 0.540302, 0.010000, 0.999950],
  [0.909297, -0.416147, 0.019999, 0.999800],
  [0.141120, -0.989992, 0.029996, 0.999550]
]
```

### Output

```
[
  [1.000000, 1.000000, 1.000000, 1.000000],
  [0.841471, 1.540302, 0.010000, 1.999950],
  [1.909297, 0.583853, 0.019999, 1.999800],
  [0.141120, -0.989992, 1.029996, 1.999550]
]
```

---
