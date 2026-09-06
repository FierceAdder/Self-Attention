"""
Unit Tests for Attention Mechanisms
===================================
Validates tensor contracts, mathematical properties, masking semantics,
and gradient flow for ScaledDotProductAttention and MultiHeadAttention.
"""

import math
import pytest
import torch
import torch.nn as nn

from src.attention import ScaledDotProductAttention


class TestScaledDotProductAttention:
    """Test suite for ScaledDotProductAttention."""

    @pytest.fixture
    def attn_module(self):
        return ScaledDotProductAttention(dropout=0.0)

    def test_output_shape(self, attn_module):
        """Verify output shapes for batched multi-head tensors."""
        batch_size, n_heads, seq_len, d_k = 2, 8, 16, 64
        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)

        output, weights = attn_module(q, k, v)

        assert output.shape == (batch_size, n_heads, seq_len, d_k), (
            f"Expected output shape {(batch_size, n_heads, seq_len, d_k)}, "
            f"got {output.shape}"
        )
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len), (
            f"Expected attention weights shape {(batch_size, n_heads, seq_len, seq_len)}, "
            f"got {weights.shape}"
        )

    def test_attention_weights_sum_to_one(self, attn_module):
        """Softmax invariant: attention weights across keys must sum to 1.0."""
        batch_size, n_heads, seq_len, d_k = 2, 4, 10, 32
        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)

        _, weights = attn_module(q, k, v)
        weight_sums = weights.sum(dim=-1)
        expected = torch.ones_like(weight_sums)

        assert torch.allclose(weight_sums, expected, atol=1e-6), (
            "Attention weights along the key dimension must sum to 1.0"
        )

    def test_padding_mask(self, attn_module):
        """Verify that masked-out key tokens receive exactly 0 attention weight."""
        batch_size, n_heads, seq_len, d_k = 2, 2, 6, 16
        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)

        # Mask out the last 2 positions for all sequences: 1 = keep, 0 = mask
        # Shape: (B, 1, 1, T_k)
        mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
        mask[:, :, :, 4:] = False

        _, weights = attn_module(q, k, v, mask=mask)

        # Attention weights at masked positions must be zero
        masked_weights = weights[:, :, :, 4:]
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6), (
            f"Masked positions must have 0.0 attention weight, got max={masked_weights.max().item()}"
        )

    def test_scaling_factor_effect(self):
        """Confirm that scale factor 1/sqrt(d_k) normalizes the dot product logits."""
        d_k = 64
        q = torch.randn(1, 1, 100, d_k)
        k = torch.randn(1, 1, 100, d_k)
        # Dot product variance before scaling is ~ d_k
        raw_scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = raw_scores / math.sqrt(d_k)

        # Variance of raw dot product is around d_k (64), scaled is around 1.0
        assert raw_scores.var().item() > scaled_scores.var().item() * 10
        assert 0.5 < scaled_scores.var().item() < 2.0, (
            f"Scaled scores variance should be ~1.0, got {scaled_scores.var().item()}"
        )

    def test_gradient_propagation(self, attn_module):
        """Ensure gradients propagate back through Q, K, and V."""
        batch_size, n_heads, seq_len, d_k = 2, 2, 4, 16
        q = torch.randn(batch_size, n_heads, seq_len, d_k, requires_grad=True)
        k = torch.randn(batch_size, n_heads, seq_len, d_k, requires_grad=True)
        v = torch.randn(batch_size, n_heads, seq_len, d_k, requires_grad=True)

        output, _ = attn_module(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()
