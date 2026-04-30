# Bug 007: Generation Loop Doesn't Handle Sequence Length Limit

## The Bug

During autoregressive generation, the model keeps appending tokens to the input
without ever truncating to `max_seq_len`. When the sequence exceeds the maximum,
positional encoding lookup or attention computation fails or produces wrong results.

### What happens

- If PE is precomputed for `max_seq_len` positions, accessing `PE[seq_len:]` when
  `seq_len > max_seq_len` causes an IndexError or silent wrong behavior.
- The model's context window is exceeded, leading to garbage output or crashes.

### How to detect

- Generate more tokens than `max_seq_len` — does it crash?
- Check if the input is truncated to the last `max_seq_len` tokens at each step.

### The Fix

At each generation step, truncate: `input_ids = generated[-max_seq_len:]`.
This is the sliding window approach — the model only sees the last `max_seq_len` tokens.
