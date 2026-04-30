"""
Tests for v0.3: TinyGPT model, generation, TransformerBlock.
"""

import numpy as np
import pytest

from tracegpt.model import TinyGPT, GPTConfig, TransformerBlock
from tracegpt.ops import causal_mask, softmax
from tracegpt.tracer import Tracer


class TestGPTConfig:
    def test_default_config(self):
        config = GPTConfig()
        assert config.vocab_size == 16
        assert config.n_layers == 2
        assert config.weight_tying is True

    def test_custom_config(self):
        config = GPTConfig(vocab_size=32, d_model=64, n_heads=4, n_layers=3)
        assert config.d_model == 64
        assert config.d_model // config.n_heads == 16


class TestTransformerBlock:
    def test_output_shape(self):
        seq_len, d_model, n_heads, d_ff = 4, 8, 2, 16
        block = TransformerBlock(d_model, n_heads, d_ff)
        X = np.random.randn(seq_len, d_model)
        mask = causal_mask(seq_len)
        output = block.forward(X, mask)
        assert output.shape == (seq_len, d_model)

    def test_no_nan(self):
        seq_len, d_model, n_heads, d_ff = 4, 8, 2, 16
        block = TransformerBlock(d_model, n_heads, d_ff)
        X = np.random.randn(seq_len, d_model)
        mask = causal_mask(seq_len)
        output = block.forward(X, mask)
        assert not np.any(np.isnan(output))

    def test_with_tracer(self):
        seq_len, d_model, n_heads, d_ff = 3, 8, 2, 16
        block = TransformerBlock(d_model, n_heads, d_ff)
        X = np.random.randn(seq_len, d_model)
        mask = causal_mask(seq_len)
        tracer = Tracer()
        block.forward(X, mask, tracer=tracer, layer_id=0)
        # Should have 4 trace steps: mha, add_norm1, ffn, add_norm2
        assert len(tracer) == 4


class TestTinyGPT:
    def test_forward_shape(self):
        config = GPTConfig(vocab_size=16, max_seq_len=8, d_model=16, n_heads=2, n_layers=2, d_ff=32)
        model = TinyGPT(config)
        token_ids = np.array([1, 2, 3])
        logits = model.forward(token_ids)
        assert logits.shape == (3, 16)

    def test_predict_next(self):
        config = GPTConfig(vocab_size=16, max_seq_len=8, d_model=16, n_heads=2, n_layers=1, d_ff=32)
        model = TinyGPT(config)
        token_ids = np.array([1, 2])
        pred_id, probs = model.predict_next(token_ids)
        assert isinstance(pred_id, int)
        assert 0 <= pred_id < 16
        assert probs.shape == (16,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    def test_generate_greedy(self):
        config = GPTConfig(vocab_size=16, max_seq_len=8, d_model=16, n_heads=2, n_layers=1, d_ff=32)
        model = TinyGPT(config)
        prompt = np.array([1, 2])
        result = model.generate(prompt, max_new_tokens=3, temperature=0.0)
        assert len(result) == 5
        np.testing.assert_array_equal(result[:2], prompt)

    def test_generate_with_temperature(self):
        config = GPTConfig(vocab_size=16, max_seq_len=8, d_model=16, n_heads=2, n_layers=1, d_ff=32)
        model = TinyGPT(config)
        prompt = np.array([1, 2])
        result = model.generate(prompt, max_new_tokens=3, temperature=1.0)
        assert len(result) == 5
        assert all(0 <= t < 16 for t in result)

    def test_generate_top_k(self):
        config = GPTConfig(vocab_size=16, max_seq_len=8, d_model=16, n_heads=2, n_layers=1, d_ff=32)
        model = TinyGPT(config)
        prompt = np.array([1, 2])
        result = model.generate(prompt, max_new_tokens=3, temperature=1.0, top_k=5)
        assert len(result) == 5

    def test_weight_tying(self):
        config = GPTConfig(vocab_size=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, weight_tying=True)
        model = TinyGPT(config)
        assert model.W_out is None

    def test_no_weight_tying(self):
        config = GPTConfig(vocab_size=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, weight_tying=False)
        model = TinyGPT(config)
        assert model.W_out is not None
        assert model.W_out.shape == (8, 8)

    def test_count_parameters(self):
        config = GPTConfig(vocab_size=8, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        model = TinyGPT(config)
        params = model.count_parameters()
        assert params > 0

    def test_seq_len_exceeds_max(self):
        config = GPTConfig(vocab_size=8, max_seq_len=4, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        model = TinyGPT(config)
        with pytest.raises(AssertionError):
            model.forward(np.array([1, 2, 3, 4, 5]))  # 5 > max_seq_len=4

    def test_with_tracer(self):
        config = GPTConfig(vocab_size=8, max_seq_len=4, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        model = TinyGPT(config)
        tracer = Tracer()
        model.forward(np.array([1, 2]), tracer=tracer)
        # Should have: embedding, add_pe, 4 block steps, final_layernorm, output = 8
        assert len(tracer) == 8

    def test_deterministic_forward(self):
        config = GPTConfig(vocab_size=8, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        m1 = TinyGPT(config)
        m2 = TinyGPT(config)  # same seed=42
        ids = np.array([1, 2, 3])
        np.testing.assert_allclose(m1.forward(ids), m2.forward(ids))

    def test_repr(self):
        config = GPTConfig(vocab_size=8, d_model=8, n_heads=2, n_layers=2, d_ff=16)
        model = TinyGPT(config)
        s = repr(model)
        assert "TinyGPT" in s
        assert "vocab=8" in s
