# Bug 006: Weight Tying Transpose Error

## The Bug

When using weight tying (sharing the embedding table as the output projection),
the correct formula is `logits = X @ E^T`, not `logits = E @ X`.

### What happens

Shapes will be wrong, or the model produces garbage predictions. The embedding table
has shape `(vocab_size, d_model)`, so we need its transpose `(d_model, vocab_size)`
to project from `d_model` → `vocab_size`.

### How to detect

- Check output logits shape: should be `(seq_len, vocab_size)`.
- If you get a shape error, the transpose is missing.
- If logits shape is correct but predictions are garbage, you might have `E @ X^T`.

### The Fix

`logits = X @ E.T` (project d_model → vocab_size), NOT `E @ X.T`.
