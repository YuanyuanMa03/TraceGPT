# TraceGPT Level 0: Next Token Prediction

**Generated:** 2026-04-30 14:19:46
**Total operations:** 4

---

## Step 1: embedding_lookup

**Formula:** `e = E[token_id]`

**Explanation:** Look up the embedding vector for token 'hello' (id=0). The embedding table has 4 rows (one per token) and 3 columns (embedding dim).

### Shapes

- **embedding_table**: (4, 3)
- **token_id**: ()
- **output**: (3,)

### Inputs

**embedding_table:**
```
[
  [1.000000, 0.000000, 0.000000],
  [0.000000, 1.000000, 0.000000],
  [0.000000, 0.000000, 1.000000],
  [1.000000, 1.000000, 0.000000]
]
```

**token_id:**
```
0.0
```

### Output

```
[1.000000, 0.000000, 0.000000]
```

---

## Step 2: linear_projection

**Formula:** `logits = e @ W_out + b_out`

**Explanation:** Project the embedding to vocabulary-size logits. Each logit is an unnormalized score for how likely each token is to be the next token.

### Shapes

- **embedding**: (3,)
- **W_out**: (3, 4)
- **b_out**: (4,)
- **output**: (4,)

### Inputs

**embedding:**
```
[1.000000, 0.000000, 0.000000]
```

**W_out:**
```
[
  [2.000000, 0.500000, 0.000000, -1.000000],
  [-0.500000, 1.500000, 0.500000, 0.000000],
  [0.000000, -1.000000, 2.000000, 1.000000]
]
```

**b_out:**
```
[0.100000, -0.200000, 0.300000, -0.100000]
```

### Output

```
[2.100000, 0.300000, 0.300000, -1.100000]
```

---

## Step 3: softmax

**Formula:** `P(token) = exp(logit_i) / Σ exp(logit_j)`

**Explanation:** Convert logits to a probability distribution over the vocabulary. The softmax ensures all probabilities are positive and sum to 1.

### Shapes

- **logits**: (4,)
- **output**: (4,)

### Inputs

**logits:**
```
[2.100000, 0.300000, 0.300000, -1.100000]
```

### Output

```
[0.729203, 0.120536, 0.120536, 0.029724]
```

---

## Step 4: argmax_prediction

**Formula:** `predicted_token = argmax(P)`

**Explanation:** Pick the token with highest probability. Predicted: 'hello' (id=0) with probability 0.7292

### Shapes

- **probabilities**: (4,)
- **output**: ()

### Inputs

**probabilities:**
```
[0.729203, 0.120536, 0.120536, 0.029724]
```

### Output

```
0.0
```

---
