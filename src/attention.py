"""
Attention Mechanisms
====================
Core attention building blocks implementing:
1. ScaledDotProductAttention (Vaswani et al. 2017, Section 3.2.1)
2. MultiHeadAttention (Vaswani et al. 2017, Section 3.2.2)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    r"""
    Scaled Dot-Product Attention mechanism.

    Computes attention weights and weighted context vectors according to:

    .. math::
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V

    where :math:`Q \in \mathbb{R}^{B \times h \times T_q \times d_k}`,
    :math:`K \in \mathbb{R}^{B \times h \times T_k \times d_k}`, and
    :math:`V \in \mathbb{R}^{B \times h \times T_k \times d_v}`.

    The scaling factor :math:`\frac{1}{\sqrt{d_k}}` counters the growth of dot product
    variance with respect to dimension :math:`d_k`, ensuring stable gradients during
    backpropagation by preventing the softmax function from saturating into regions
    with near-zero gradients.

    Args:
        dropout (float, optional): Dropout probability applied to the attention
            weight matrix. Defaults to 0.0.

    Shape:
        - **Query (Q)**: :math:`(B, h, T_q, d_k)`
        - **Key (K)**: :math:`(B, h, T_k, d_k)`
        - **Value (V)**: :math:`(B, h, T_k, d_v)`
        - **Mask (optional)**: :math:`(B, 1, 1, T_k)` or :math:`(B, 1, T_q, T_k)`
          where values are boolean (`True` to keep, `False` to mask out) or
          binary float (`1.0` to keep, `0.0` to mask out).
        - **Output**: :math:`(B, h, T_q, d_v)`
        - **Attention Weights**: :math:`(B, h, T_q, T_k)`

    Examples:
        >>> attn = ScaledDotProductAttention(dropout=0.1)
        >>> q = torch.randn(2, 8, 16, 64)
        >>> k = torch.randn(2, 8, 16, 64)
        >>> v = torch.randn(2, 8, 16, 64)
        >>> output, weights = attn(q, k, v)
        >>> output.shape
        torch.Size([2, 8, 16, 64])
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Execute scaled dot-product attention.

        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len_q, d_k).
            k: Key tensor of shape (batch_size, n_heads, seq_len_k, d_k).
            v: Value tensor of shape (batch_size, n_heads, seq_len_k, d_v).
            mask: Optional attention mask tensor broadcastable to
                (batch_size, n_heads, seq_len_q, seq_len_k). Positions where
                mask == 0 or mask == False will be replaced with -1e9.

        Returns:
            Tuple of:
                - Context output tensor of shape (batch_size, n_heads, seq_len_q, d_v).
                - Normalized attention weights of shape (batch_size, n_heads, seq_len_q, seq_len_k).
        """
        # TODO: Implement Scaled Dot-Product Attention:
        # 1. Extract d_k from q.size(-1).
        # 2. Compute raw attention scores: Q @ K.transpose(-2, -1) / sqrt(d_k).
        # 3. If mask is provided, apply masked_fill (set positions where mask == 0 to -1e9).
        # 4. Compute attention probabilities via softmax over the key dimension (dim=-1).
        # 5. Apply dropout (self.dropout) to attention weights.
        # 6. Compute output: attention_weights @ V.
        # 7. Return (output, attention_weights).
        dk=q.size(-1)
        raw_attn_scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None:
            raw_attn_scores=raw_attn_scores.masked_fill(mask==0,-1e9)
        attn_probability=torch.softmax(raw_attn_scores,dim=-1)
        attn_probability=self.dropout(attn_probability)
        output=torch.matmul(attn_probability, v)
        return output,attn_probability

