"""
model.py — TinyGPT: A complete (tiny) GPT model in pure NumPy.

Implements a GPT-2 style model with:
  - Token + Positional Embeddings
  - N stacked Transformer blocks (Multi-Head Attention + FFN + Residual + LayerNorm)
  - Output projection with optional weight tying
  - Autoregressive text generation (greedy + temperature sampling)

Every operation can be traced with Tracer for full transparency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tracegpt.ops import (
    softmax,
    causal_mask,
    layer_norm,
    linear,
    relu,
    gelu,
    sinusoidal_position_encoding,
    multi_head_attention,
)
from tracegpt.tracer import Tracer


@dataclass
class GPTConfig:
    """Configuration for a TinyGPT model."""

    vocab_size: int = 16
    max_seq_len: int = 8
    d_model: int = 16
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 32
    weight_tying: bool = True
    activation: str = "gelu"  # "relu" or "gelu"


class TransformerBlock:
    """A single Transformer block: MHA → Add & Norm → FFN → Add & Norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, activation: str = "gelu"):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation_fn = gelu if activation == "gelu" else relu

        # Initialize weights with small random values (Xavier-like)
        scale_qkv = np.sqrt(2.0 / (d_model + d_model))
        scale_ff = np.sqrt(2.0 / (d_model + d_ff))

        rng = np.random.RandomState(hash((d_model, n_heads, d_ff)) % 2**31)

        # MHA projections
        self.W_Q = rng.randn(d_model, d_model) * scale_qkv
        self.W_K = rng.randn(d_model, d_model) * scale_qkv
        self.W_V = rng.randn(d_model, d_model) * scale_qkv
        self.W_O = rng.randn(d_model, d_model) * scale_qkv

        # FFN weights
        self.W_ff1 = rng.randn(d_model, d_ff) * scale_ff
        self.b_ff1 = np.zeros(d_ff)
        self.W_ff2 = rng.randn(d_ff, d_model) * scale_ff
        self.b_ff2 = np.zeros(d_model)

        # LayerNorm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def forward(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        tracer: Tracer | None = None,
        layer_id: int = 0,
    ) -> np.ndarray:
        """
        Forward pass through the Transformer block.

        Parameters
        ----------
        X : np.ndarray, shape (seq_len, d_model)
        mask : np.ndarray, shape (seq_len, seq_len)
        tracer : Tracer or None
        layer_id : int, for labeling trace steps

        Returns
        -------
        np.ndarray, shape (seq_len, d_model)
        """
        prefix = f"L{layer_id}"

        # 1. Multi-head causal self-attention
        attn_out = multi_head_attention(X, self.W_Q, self.W_K, self.W_V, self.W_O, self.n_heads, mask)

        if tracer is not None:
            tracer.trace(
                name=f"{prefix}_multi_head_attention",
                formula="attn = MHA(X)",
                inputs={"X": X},
                output=attn_out,
                explanation=f"Layer {layer_id}: Multi-head causal self-attention with {self.n_heads} heads.",
            )

        # 2. Residual + LayerNorm
        residual1 = X + attn_out
        norm1 = layer_norm(residual1, self.gamma1, self.beta1)

        if tracer is not None:
            tracer.trace(
                name=f"{prefix}_add_norm_1",
                formula="norm1 = LayerNorm(X + MHA(X))",
                inputs={"X": X, "attn": attn_out},
                output=norm1,
                explanation=f"Layer {layer_id}: Residual connection + LayerNorm after attention.",
            )

        # 3. Feed-forward network
        ff_hidden = linear(norm1, self.W_ff1, self.b_ff1)
        ff_activated = self.activation_fn(ff_hidden)
        ff_out = linear(ff_activated, self.W_ff2, self.b_ff2)

        if tracer is not None:
            tracer.trace(
                name=f"{prefix}_ffn",
                formula=f"FFN(x) = {self.activation_fn.__name__}(x@W1+b1)@W2+b2",
                inputs={"input": norm1},
                output=ff_out,
                explanation=f"Layer {layer_id}: Position-wise feed-forward ({self.d_model}→{self.d_ff}→{self.d_model}).",
            )

        # 4. Residual + LayerNorm
        residual2 = norm1 + ff_out
        norm2 = layer_norm(residual2, self.gamma2, self.beta2)

        if tracer is not None:
            tracer.trace(
                name=f"{prefix}_add_norm_2",
                formula="output = LayerNorm(norm1 + FFN(norm1))",
                inputs={"norm1": norm1, "ff_out": ff_out},
                output=norm2,
                explanation=f"Layer {layer_id}: Residual connection + LayerNorm after FFN.",
            )

        return norm2


class TinyGPT:
    """
    A complete GPT model in pure NumPy.

    Usage:
        config = GPTConfig(vocab_size=16, d_model=16, n_heads=2, n_layers=2)
        model = TinyGPT(config)
        logits = model.forward(token_ids)
        next_token = model.generate(token_ids, max_new_tokens=5)
    """

    def __init__(self, config: GPTConfig):
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model

        # Token embedding table
        rng = np.random.RandomState(42)
        self.token_embedding = rng.randn(config.vocab_size, config.d_model) * 0.02

        # Transformer blocks
        self.layers = [
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.activation)
            for _ in range(config.n_layers)
        ]

        # Final LayerNorm
        self.gamma_final = np.ones(config.d_model)
        self.beta_final = np.zeros(config.d_model)

        # Output projection (optionally tied with embedding)
        if config.weight_tying:
            self.W_out = None  # will use token_embedding.T
        else:
            self.W_out = rng.randn(config.d_model, config.vocab_size) * 0.02
        self.b_out = np.zeros(config.vocab_size)

        # Positional encoding (precomputed)
        self.PE = sinusoidal_position_encoding(config.max_seq_len, config.d_model)

    def forward(
        self,
        token_ids: np.ndarray,
        tracer: Tracer | None = None,
    ) -> np.ndarray:
        """
        Full forward pass: token_ids → logits.

        Parameters
        ----------
        token_ids : np.ndarray, shape (seq_len,)
            Integer token IDs.
        tracer : Tracer or None
            If provided, records every operation.

        Returns
        -------
        np.ndarray, shape (seq_len, vocab_size)
            Logits for each position.
        """
        seq_len = len(token_ids)
        assert seq_len <= self.max_seq_len, f"seq_len={seq_len} > max_seq_len={self.max_seq_len}"

        # 1. Token embedding
        X = self.token_embedding[token_ids]  # (seq_len, d_model)

        if tracer is not None:
            tracer.trace(
                name="embedding",
                formula="X = Embedding[token_ids]",
                inputs={"token_ids": token_ids},
                output=X,
                explanation=f"Look up embeddings for {seq_len} tokens.",
            )

        # 2. Add positional encoding
        X = X + self.PE[:seq_len]

        if tracer is not None:
            tracer.trace(
                name="add_pe",
                formula="X = X + PE[:seq_len]",
                inputs={"PE": self.PE[:seq_len]},
                output=X,
                explanation="Add sinusoidal positional encodings.",
            )

        # 3. Causal mask (shared across all layers)
        mask = causal_mask(seq_len)

        # 4. Transformer blocks
        for i, layer in enumerate(self.layers):
            X = layer.forward(X, mask, tracer=tracer, layer_id=i)

        # 5. Final LayerNorm
        X = layer_norm(X, self.gamma_final, self.beta_final)

        if tracer is not None:
            tracer.trace(
                name="final_layernorm",
                formula="X = LayerNorm(X)",
                inputs={"input": X},
                output=X,
                explanation="Final layer normalization before output projection.",
            )

        # 6. Output projection
        W = self.token_embedding.T if self.config.weight_tying else self.W_out
        logits = linear(X, W, self.b_out)  # (seq_len, vocab_size)

        if tracer is not None:
            tracer.trace(
                name="output_projection",
                formula="logits = X @ W_out + b_out" + (" (weight-tied)" if self.config.weight_tying else ""),
                inputs={"X": X},
                output=logits,
                explanation="Project hidden states to vocabulary logits.",
            )

        return logits

    def predict_next(self, token_ids: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Predict the next token given a sequence.

        Returns
        -------
        (predicted_id, probabilities)
        """
        logits = self.forward(token_ids)
        last_logits = logits[-1]  # logits for last position
        probs = softmax(last_logits)
        predicted_id = int(np.argmax(probs))
        return predicted_id, probs

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: int | None = None,
        tracer: Tracer | None = None,
    ) -> np.ndarray:
        """
        Autoregressive generation.

        Parameters
        ----------
        prompt_ids : np.ndarray
            Starting token IDs.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        temperature : float
            Sampling temperature. 1.0 = normal, <1 = more deterministic, >1 = more random.
            Set to 0.0 for greedy (argmax) decoding.
        top_k : int or None
            If set, only sample from the top-k most likely tokens.
        tracer : Tracer or None
            If provided, records each generation step.

        Returns
        -------
        np.ndarray
            Complete sequence (prompt + generated tokens).
        """
        generated = list(prompt_ids)

        for step in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            input_ids = np.array(generated[-self.max_seq_len:])

            # Forward pass
            logits = self.forward(input_ids)
            next_logits = logits[-1]  # last position

            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_indices = np.argsort(next_logits)[-top_k:]
                mask = np.full_like(next_logits, -1e9)
                mask[top_k_indices] = next_logits[top_k_indices]
                next_logits = mask

            # Sample or argmax
            if temperature == 0:
                next_token = int(np.argmax(next_logits))
            else:
                probs = softmax(next_logits)
                next_token = int(np.random.choice(len(probs), p=probs))

            generated.append(next_token)

            if tracer is not None:
                tracer.trace(
                    name=f"generate_step_{step}",
                    formula=f"next_token = {'argmax' if temperature == 0 else 'sample'}(logits / T)",
                    inputs={"input_ids": input_ids, "temperature": np.array(temperature)},
                    output=np.array(next_token),
                    explanation=f"Generation step {step}: predicted token {next_token}.",
                )

        return np.array(generated)

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        total = self.token_embedding.size  # embedding
        for layer in self.layers:
            total += layer.W_Q.size + layer.W_K.size + layer.W_V.size + layer.W_O.size
            total += layer.W_ff1.size + layer.b_ff1.size
            total += layer.W_ff2.size + layer.b_ff2.size
            total += layer.gamma1.size + layer.beta1.size
            total += layer.gamma2.size + layer.beta2.size
        total += self.gamma_final.size + self.beta_final.size
        if not self.config.weight_tying:
            total += self.W_out.size
        total += self.b_out.size
        return total

    def __repr__(self) -> str:
        return (
            f"TinyGPT(vocab={self.config.vocab_size}, d={self.config.d_model}, "
            f"heads={self.config.n_heads}, layers={self.config.n_layers}, "
            f"ff={self.config.d_ff}, params={self.count_parameters():,})"
        )
