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
        dk=q.size(-1)
        raw_attn_scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None:
            raw_attn_scores=raw_attn_scores.masked_fill(mask==0,-1e9)
        attn_probability=torch.softmax(raw_attn_scores,dim=-1)
        attn_probability=self.dropout(attn_probability)
        output=torch.matmul(attn_probability, v)
        return output, attn_probability


class MultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention (MHA) mechanism.

    Allows the model to jointly attend to information from different representation
    subspaces at different positions, as formulated in Vaswani et al. (2017):

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O

    where:
    .. math::
        \text{head}_i = \text{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)

    Projections:
        - :math:`W^Q \in \mathbb{R}^{d_{model} \times d_{model}}`
        - :math:`W^K \in \mathbb{R}^{d_{model} \times d_{model}}`
        - :math:`W^V \in \mathbb{R}^{d_{model} \times d_{model}}`
        - :math:`W^O \in \mathbb{R}^{d_{model} \times d_{model}}`

    Instead of computing each head sequentially, all :math:`h` heads are parallelized
    into a single batched tensor operation by reshaping projections to
    :math:`(B, h, T, d_k)` and executing batched matrix multiplication.

    Args:
        d_model (int): Total dimensionality of the model (input and output).
        n_heads (int): Number of parallel attention heads. Must divide `d_model`.
        dropout (float, optional): Dropout probability applied to attention weights
            and output projection. Defaults to 0.1.

    Raises:
        ValueError: If `d_model` is not divisible by `n_heads`.

    Shape:
        - **Query (Q)**: :math:`(B, T_q, d_{model})`
        - **Key (K)**: :math:`(B, T_k, d_{model})`
        - **Value (V)**: :math:`(B, T_k, d_{model})`
        - **Mask (optional)**: :math:`(B, 1, 1, T_k)` or :math:`(B, 1, T_q, T_k)`
        - **Output**: :math:`(B, T_q, d_{model})`
        - **Attention Weights**: :math:`(B, h, T_q, T_k)`

    Examples:
        >>> mha = MultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 16, 512)
        >>> out, weights = mha(x, x, x)
        >>> out.shape
        torch.Size([2, 16, 512])
        >>> weights.shape
        torch.Size([2, 8, 16, 16])
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Scaled Dot-Product Attention core
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Final linear projection and dropout
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Perform multi-head attention forward pass.

        Args:
            q: Query sequence tensor of shape (batch_size, seq_len_q, d_model).
            k: Key sequence tensor of shape (batch_size, seq_len_k, d_model).
            v: Value sequence tensor of shape (batch_size, seq_len_k, d_model).
            mask: Optional attention mask tensor broadcastable to
                (batch_size, n_heads, seq_len_q, seq_len_k).

        Returns:
            Tuple of:
                - Projected multi-head context tensor of shape (batch_size, seq_len_q, d_model).
                - Attention weights tensor across all heads of shape
                  (batch_size, n_heads, seq_len_q, seq_len_k).
        """
        batch_size,seq_len_q=q.size()[:2]
        q_proj=self.w_q(q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        k_proj=self.w_k(k).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        v_proj=self.w_v(v).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        context,attn_weights=self.attention(q_proj,k_proj,v_proj,mask=mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output=self.dropout(self.w_o(context))
        return output, attn_weights
        
        


