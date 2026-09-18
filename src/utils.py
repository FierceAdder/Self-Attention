"""
Attention and Masking Utilities
===============================
Helper functions for generating padding masks, causal masks, and shape validation.
"""

from typing import Optional
import torch


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    r"""
    Generate boolean padding mask for variable-length batch sequences.

    Tokens at positions < length are valid (`True`), while padding tokens are
    masked out (`False`).

    Args:
        lengths (torch.Tensor): 1D integer tensor of shape (batch_size,) containing
            valid sequence lengths for each item in the batch.
        max_len (int): Maximum sequence length.

    Returns:
        torch.Tensor: Boolean mask of shape (batch_size, 1, 1, max_len)
            broadcastable across all attention heads and query positions.

    Examples:
        >>> lengths = torch.tensor([3, 5])
        >>> mask = create_padding_mask(lengths, max_len=5)
        >>> mask.shape
        torch.Size([2, 1, 1, 5])
        >>> mask[0, 0, 0].tolist()
        [True, True, True, False, False]
    """
    batch_size = lengths.size(0)
    # Range matrix: (1, max_len)
    positions = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)
    # Broadcast comparison: (batch_size, max_len)
    mask = positions < lengths.unsqueeze(1)
    # Reshape to (batch_size, 1, 1, max_len) for attention broadcasting
    return mask.unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    r"""
    Generate autoregressive lower-triangular causal mask.

    Positions where key index <= query index are valid (`True`), preventing tokens
    from attending to future tokens.

    Args:
        seq_len (int): Sequence length of queries and keys.
        device (torch.device, optional): Device on which to create the tensor.

    Returns:
        torch.Tensor: Boolean lower-triangular mask of shape (1, 1, seq_len, seq_len).

    Examples:
        >>> mask = create_causal_mask(3)
        >>> mask[0, 0].tolist()
        [[True, False, False], [True, True, False], [True, True, True]]
    """
    # Lower triangular matrix with ones on and below diagonal
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(1)
