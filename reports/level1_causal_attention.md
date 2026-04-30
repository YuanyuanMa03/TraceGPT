# TraceGPT Level 1: Causal Self-Attention

**Generated:** 2026-04-30 14:21:01
**Total operations:** 8

---

## Step 1: input_embeddings

**Formula:** `X ∈ R^{seq_len × d_model}`

**Explanation:** Input sequence of 3 tokens, each embedded into a 4-dim vector. Tokens: ['I', 'love', 'AI']

### Shapes

- **tokens**: (3,)
- **output**: (3, 4)

### Inputs

**tokens:**
```
['I', 'love', 'AI']
```

### Output

```
[
  [1.000000, 0.000000, 1.000000, 0.000000],
  [0.000000, 1.000000, 0.000000, 1.000000],
  [1.000000, 1.000000, 0.000000, 1.000000]
]
```

---

## Step 2: compute_qkv

**Formula:** `Q = X @ W_Q + b_Q,  K = X @ W_K + b_K,  V = X @ W_V + b_V`

**Explanation:** Project each token's embedding into three vectors: Query (what am I looking for?), Key (what do I contain?), Value (what information do I provide?).

### Shapes

- **X**: (3, 4)
- **W_Q**: (4, 3)
- **W_K**: (4, 3)
- **W_V**: (4, 3)
- **output**: (3, 3, 3)

### Inputs

**X:**
```
[
  [1.000000, 0.000000, 1.000000, 0.000000],
  [0.000000, 1.000000, 0.000000, 1.000000],
  [1.000000, 1.000000, 0.000000, 1.000000]
]
```

**W_Q:**
```
[
  [1.000000, 0.000000, 0.000000],
  [0.000000, 1.000000, 0.000000],
  [0.000000, 0.000000, 1.000000],
  [0.500000, 0.500000, 0.000000]
]
```

**W_K:**
```
[
  [0.000000, 1.000000, 0.000000],
  [1.000000, 0.000000, 0.000000],
  [0.000000, 0.000000, 1.000000],
  [0.000000, 0.500000, 0.500000]
]
```

**W_V:**
```
[
  [1.000000, 0.000000, 0.000000],
  [0.000000, 1.000000, 0.000000],
  [0.000000, 0.000000, 1.000000],
  [0.000000, 0.000000, 0.500000]
]
```

### Output

```
[
  [
  [1.000000, 0.000000, 1.000000],
  [0.500000, 1.500000, 0.000000],
  [1.500000, 1.500000, 0.000000]
],
  [
  [0.000000, 1.000000, 1.000000],
  [1.000000, 0.500000, 0.500000],
  [1.000000, 1.500000, 0.500000]
],
  [
  [1.000000, 0.000000, 1.000000],
  [0.000000, 1.000000, 0.500000],
  [1.000000, 1.000000, 0.500000]
]
]
```

---

## Step 3: attention_scores

**Formula:** `scores = Q @ K^T`

**Explanation:** Raw attention scores: how much each token 'attends to' every other token. scores[i][j] = similarity between token i's query and token j's key.

### Shapes

- **Q**: (3, 3)
- **K**: (3, 3)
- **output**: (3, 3)

### Inputs

**Q:**
```
[
  [1.000000, 0.000000, 1.000000],
  [0.500000, 1.500000, 0.000000],
  [1.500000, 1.500000, 0.000000]
]
```

**K:**
```
[
  [0.000000, 1.000000, 1.000000],
  [1.000000, 0.500000, 0.500000],
  [1.000000, 1.500000, 0.500000]
]
```

### Output

```
[
  [1.000000, 1.500000, 1.500000],
  [1.500000, 1.250000, 2.750000],
  [1.500000, 2.250000, 3.750000]
]
```

---

## Step 4: scale_scores

**Formula:** `scaled_scores = scores / sqrt(d_k)`

**Explanation:** Scale down by sqrt(d_k)=1.7321. This prevents the dot products from growing too large when d_k is big, which would push softmax into regions with tiny gradients.

### Shapes

- **scores**: (3, 3)
- **d_k**: ()
- **output**: (3, 3)

### Inputs

**scores:**
```
[
  [1.000000, 1.500000, 1.500000],
  [1.500000, 1.250000, 2.750000],
  [1.500000, 2.250000, 3.750000]
]
```

**d_k:**
```
3.0
```

### Output

```
[
  [0.577350, 0.866025, 0.866025],
  [0.866025, 0.721688, 1.587713],
  [0.866025, 1.299038, 2.165064]
]
```

---

## Step 5: causal_mask

**Formula:** `M[i][j] = 1 if j ≤ i, else 0`

**Explanation:** Lower-triangular mask: each token can only attend to itself and earlier tokens. This is what makes the model 'causal' — it cannot see the future during generation.

### Shapes

- **seq_len**: ()
- **output**: (3, 3)

### Inputs

**seq_len:**
```
3.0
```

### Output

```
[
  [1.000000, 0.000000, 0.000000],
  [1.000000, 1.000000, 0.000000],
  [1.000000, 1.000000, 1.000000]
]
```

---

## Step 6: apply_mask

**Formula:** `masked_scores = scaled_scores + (1 - M) * (-inf)`

**Explanation:** Add -inf to positions where mask=0. After softmax, these become ~0 probability, preventing attention to future tokens.

### Shapes

- **scaled_scores**: (3, 3)
- **mask**: (3, 3)
- **output**: (3, 3)

### Inputs

**scaled_scores:**
```
[
  [0.577350, 0.866025, 0.866025],
  [0.866025, 0.721688, 1.587713],
  [0.866025, 1.299038, 2.165064]
]
```

**mask:**
```
[
  [1.000000, 0.000000, 0.000000],
  [1.000000, 1.000000, 0.000000],
  [1.000000, 1.000000, 1.000000]
]
```

### Output

```
[
  [0.577350, -999999999.133975, -999999999.133975],
  [0.866025, 0.721688, -999999998.412287],
  [0.866025, 1.299038, 2.165064]
]
```

---

## Step 7: attention_softmax

**Formula:** `weights = softmax(masked_scores)`

**Explanation:** Convert masked scores to attention weights (probability distribution). Each row sums to 1. Positions with -inf become 0.

### Shapes

- **masked_scores**: (3, 3)
- **output**: (3, 3)

### Inputs

**masked_scores:**
```
[
  [0.577350, -999999999.133975, -999999999.133975],
  [0.866025, 0.721688, -999999998.412287],
  [0.866025, 1.299038, 2.165064]
]
```

### Output

```
[
  [1.000000, 0.000000, 0.000000],
  [0.536022, 0.463978, 0.000000],
  [0.161091, 0.248386, 0.590523]
]
```

---

## Step 8: attention_output

**Formula:** `output = weights @ V`

**Explanation:** Take a weighted sum of value vectors, where weights are the attention probabilities. Each output token is a mix of the values it attends to.

### Shapes

- **weights**: (3, 3)
- **V**: (3, 3)
- **output**: (3, 3)

### Inputs

**weights:**
```
[
  [1.000000, 0.000000, 0.000000],
  [0.536022, 0.463978, 0.000000],
  [0.161091, 0.248386, 0.590523]
]
```

**V:**
```
[
  [1.000000, 0.000000, 1.000000],
  [0.000000, 1.000000, 0.500000],
  [1.000000, 1.000000, 0.500000]
]
```

### Output

```
[
  [1.000000, 0.000000, 1.000000],
  [0.536022, 0.463978, 0.768011],
  [0.751614, 0.838909, 0.580546]
]
```

---
