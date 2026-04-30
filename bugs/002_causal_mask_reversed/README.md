# Bug 002: Causal Mask Reversed

## The Bug

The causal mask is flipped — using `np.triu` instead of `np.tril`, or applying
the mask in the wrong direction. This allows tokens to attend to **future** tokens
but **not** to past tokens.

### What happens

Instead of a lower-triangular mask (attend to past), you get an upper-triangular
mask (attend to future). During autoregressive generation, this means the model
"cheats" by looking ahead.

### How to detect

- For position 0, the first row of the attention weights should only have a
  non-zero value at index 0.
- If position 0 attends to later positions, the mask is reversed.

### The Fix

Use `np.tril` for causal mask (lower-triangular = attend to current and past).
