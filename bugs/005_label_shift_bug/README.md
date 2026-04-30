# Bug 005: Label Shift Bug in Next-Token Prediction

## The Bug

In next-token prediction, the labels are misaligned with the input tokens.
Common mistakes:
- Labels not shifted by one position
- Using the same tokens as both input and target without shifting
- Off-by-one errors in the training loop

### What happens

The model learns to predict the CURRENT token instead of the NEXT token.
This makes the model appear to learn during training (loss decreases) but
it learns a trivial identity mapping, not useful next-token prediction.

### How to detect

- Input ["I", "love", "AI"] with labels ["I", "love", "AI"] is WRONG.
- Correct: input ["I", "love", "AI"] with labels ["love", "AI", <EOS>].
- The label for position i should be the token at position i+1.

### The Fix

Shift labels: `labels = tokens[1:]` and `input = tokens[:-1]`.
