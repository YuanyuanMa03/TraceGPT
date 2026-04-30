# Bug 003: Missing √d_k Scaling in Attention

## The Bug

Forgetting to divide attention scores by √d_k before softmax.

### What happens

Without the 1/√d_k scaling, the dot products Q·K^T grow with the dimension d_k.
For large d_k, the scores become very large, pushing softmax into saturation
where gradients are near zero. This destabilizes training.

### How to detect

- For d_k > 1, attention weights become very peaked (one value near 1, rest near 0).
- Compare with the scaled version: weights should be more evenly distributed.

### The Fix

Always scale by 1/√d_k: `scores = Q @ K^T / sqrt(d_k)`.
