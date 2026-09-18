"""
Unit Tests for Positional Encoding
==================================
Tests mathematical properties, non-trainable buffer registration,
tensor shape handling, and device placement for SinusoidalPositionalEncoding.
"""

import math
import pytest
import torch

from src.positional_encoding import SinusoidalPositionalEncoding


class TestSinusoidalPositionalEncoding:
    """Test suite for SinusoidalPositionalEncoding."""

    @pytest.fixture
    def pe_module(self):
        return SinusoidalPositionalEncoding(d_model=64, max_seq_len=100, dropout=0.0)

    def test_output_shape(self, pe_module):
        """Verify output tensor preserves input dimensions (B, T, d_model)."""
        batch_size, seq_len, d_model = 3, 25, 64
        x = torch.randn(batch_size, seq_len, d_model)

        out = pe_module(x)
        assert out.shape == (batch_size, seq_len, d_model)

    def test_buffer_registered_and_not_parameter(self, pe_module):
        """Confirm 'pe' is registered as a non-trainable buffer, not a parameter."""
        # Must be in buffers
        assert "pe" in dict(pe_module.named_buffers()), "'pe' must be registered as a persistent buffer"
        # Must NOT be in trainable parameters
        assert len(list(pe_module.parameters())) == 0, "Sinusoidal PE should contain no trainable parameters"

    def test_exact_mathematical_values(self):
        """Verify computed sinusoidal values strictly match closed-form formulas."""
        d_model = 8
        pe_module = SinusoidalPositionalEncoding(d_model=d_model, max_seq_len=10, dropout=0.0)
        pe = pe_module.pe.squeeze(0)  # Shape: (10, 8)

        for pos in range(5):
            for i in range(d_model // 2):
                denominator = 10000.0 ** (2 * i / d_model)
                expected_sin = math.sin(pos / denominator)
                expected_cos = math.cos(pos / denominator)

                actual_sin = pe[pos, 2 * i].item()
                actual_cos = pe[pos, 2 * i + 1].item()

                assert math.isclose(actual_sin, expected_sin, rel_tol=1e-5, abs_tol=1e-5), (
                    f"Sin mismatch at pos={pos}, i={i}: expected {expected_sin}, got {actual_sin}"
                )
                assert math.isclose(actual_cos, expected_cos, rel_tol=1e-5, abs_tol=1e-5), (
                    f"Cos mismatch at pos={pos}, i={i}: expected {expected_cos}, got {actual_cos}"
                )

    def test_position_distinctness(self, pe_module):
        """Different positions must have distinct positional encoding vectors."""
        pe = pe_module.pe.squeeze(0)
        # Check cosine similarity between pos 0 and pos 1 is strictly less than 1.0
        dot_product = torch.dot(pe[0], pe[1])
        norm_product = torch.norm(pe[0]) * torch.norm(pe[1])
        cos_sim = (dot_product / norm_product).item()
        assert cos_sim < 0.99, f"Adjacent positions should be distinct, got cosine similarity {cos_sim}"

    def test_gradient_flow_to_inputs(self, pe_module):
        """Gradients should flow straight back to input embeddings without obstruction."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = pe_module(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x)), (
            "Because PE is added elementwise, d(loss)/dx should be 1.0 when dropout=0"
        )
