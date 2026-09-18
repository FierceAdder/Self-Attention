"""
Unit Tests for Transformer Encoder
==================================
Tests FeedForward, EncoderLayer (Pre-LN & Post-LN), and full TransformerEncoder stack.
"""

import pytest
import torch
import torch.nn as nn

from src.config import TransformerConfig
from src.encoder import (
    PositionwiseFeedForward,
    TransformerEncoderLayer,
    TransformerEncoder,
)


class TestPositionwiseFeedForward:
    """Test suite for PositionwiseFeedForward."""

    def test_output_shape(self):
        batch_size, seq_len, d_model, d_ff = 2, 8, 64, 256
        ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)

        out = ffn(x)
        assert out.shape == (batch_size, seq_len, d_model)

    def test_gradient_flow(self):
        ffn = PositionwiseFeedForward(d_model=32, d_ff=128, dropout=0.0)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = ffn(x)
        out.sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestTransformerEncoderLayer:
    """Test suite for TransformerEncoderLayer."""

    @pytest.mark.parametrize("norm_first", [False, True])
    def test_encoder_layer_shapes(self, norm_first):
        batch_size, seq_len, d_model = 2, 10, 64
        layer = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=4,
            d_ff=256,
            dropout=0.0,
            norm_first=norm_first,
        )
        x = torch.randn(batch_size, seq_len, d_model)
        out = layer(x)

        assert out.shape == (batch_size, seq_len, d_model)

    def test_encoder_layer_with_mask(self):
        batch_size, seq_len, d_model = 2, 6, 32
        layer = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            dropout=0.0,
            norm_first=False,
        )
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
        mask[:, :, :, 4:] = False

        out = layer(x, mask=mask)
        assert out.shape == (batch_size, seq_len, d_model)


class TestTransformerEncoder:
    """Test suite for full TransformerEncoder stack."""

    @pytest.fixture
    def config(self):
        return TransformerConfig(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            d_ff=256,
            n_layers=3,
            max_seq_len=128,
            dropout=0.0,
            norm_first=False,
        )

    def test_encoder_forward_shape(self, config):
        encoder = TransformerEncoder(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        out = encoder(tokens)

        assert out.shape == (2, 16, config.d_model)

    def test_full_backward_pass(self, config):
        encoder = TransformerEncoder(config)
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        out = encoder(tokens)
        loss = out.sum()
        loss.backward()

        # Check embedding gradients
        assert encoder.token_embedding.weight.grad is not None
        # Check all encoder layer parameters have finite gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Param {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"Param {name} has NaN/Inf gradient"
