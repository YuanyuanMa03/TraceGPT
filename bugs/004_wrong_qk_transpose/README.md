# Bug 004: Wrong QK Transpose

## The Bug

Computing `K @ Q^T` instead of `Q @ K^T` for attention scores.

### What happens

The attention matrix is transposed: `scores[i][j]` now represents "how much key i
matches query j" instead of "how much query i matches key j". The attention pattern
is backwards.

### How to detect

- The attention score matrix shape should still be (seq_len, seq_len).
- But token 0's attention pattern will match what should be the LAST token's pattern.
- Compare with the known correct computation step by step.

### The Fix

Always compute `scores = Q @ K^T`, where Q is (seq_len, d_k) and K^T is (d_k, seq_len).
