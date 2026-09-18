"""
Unit Tests for Utilities
========================
Tests padding mask and causal mask generation.
"""

import pytest
import torch

from src.utils import create_padding_mask, create_causal_mask


def test_create_padding_mask():
    lengths = torch.tensor([2, 4, 3])
    mask = create_padding_mask(lengths, max_len=5)

    assert mask.shape == (3, 1, 1, 5)
    # Batch 0 has length 2
    assert mask[0, 0, 0].tolist() == [True, True, False, False, False]
    # Batch 1 has length 4
    assert mask[1, 0, 0].tolist() == [True, True, True, True, False]
    # Batch 2 has length 3
    assert mask[2, 0, 0].tolist() == [True, True, True, False, False]


def test_create_causal_mask():
    mask = create_causal_mask(seq_len=4)
    assert mask.shape == (1, 1, 4, 4)

    expected = [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]
    assert mask[0, 0].tolist() == expected
