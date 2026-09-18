"""
Transformer Encoder
===================
Hierarchical encoder building blocks:
1. PositionwiseFeedForward (Vaswani et al. 2017, Section 3.3)
2. TransformerEncoderLayer (Vaswani et al. 2017, Section 3.1)
3. TransformerEncoder (Full encoder stack)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.config import TransformerConfig
from src.positional_encoding import SinusoidalPositionalEncoding


class PositionwiseFeedForward(nn.Module):
    r"""
    Position-wise Feed-Forward Network (FFN).

    Applies two linear transformations with an activation function in between,
    applied to each position separately and identically:

    .. math::
        \text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2

    Args:
        d_model (int): Dimensionality of input and output representations.
        d_ff (int): Dimensionality of the inner hidden layer. Typically 4 * d_model.
        dropout (float, optional): Dropout probability applied after activation.
            Defaults to 0.1.

    Shape:
        - Input: :math:`(B, T, d_{model})`
        - Output: :math:`(B, T, d_{model})`
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Pass input through feed-forward layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        x=self.w_1(x) #x=w1*x+b1
        x=self.activation(x) #x=relu(w1*x+b1)
        x=self.dropout(x) #x=dropout(relu(w1*x+b1))
        x=self.w_2(x) #x=w2*x+b2
        x=self.dropout(x) #x=dropout(w2*x+b2)
        return x
        


class TransformerEncoderLayer(nn.Module):
    r"""
    Individual Transformer Encoder Layer.

    Composed of two primary sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network

    Each sublayer is wrapped with a residual connection and Layer Normalization.

    Supports both:
    - **Post-LN** (Vaswani et al. 2017):
      :math:`x = \text{LayerNorm}(x + \text{Dropout}(\text{SubLayer}(x)))`
    - **Pre-LN** (Modern standard for deep Transformers):
      :math:`x = x + \text{Dropout}(\text{SubLayer}(\text{LayerNorm}(x)))`

    Args:
        d_model (int): Model dimensionality.
        n_heads (int): Number of attention heads.
        d_ff (int): Inner dimension of feed-forward network.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        norm_first (bool, optional): If True, applies Pre-LN. If False,
            applies Post-LN strictly following the paper. Defaults to False.

    Shape:
        - Input: :math:`(B, T, d_{model})`
        - Output: :math:`(B, T, d_{model})`
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first

        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        # Sub-layer 2: Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Pass representation through encoder sublayers with residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask broadcastable to
                (batch_size, n_heads, seq_len, seq_len).

        Returns:
            Processed tensor of shape (batch_size, seq_len, d_model).
        """
        if self.norm_first:
            # Pre-LN: x + Sublayer(LayerNorm(x))
            norm_x = self.norm1(x)
            attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, mask=mask)
            x = x + self.dropout1(attn_out)

            norm_x = self.norm2(x)
            ff_out = self.feed_forward(norm_x)
            x = x + self.dropout2(ff_out)
        else:
            # Post-LN (Vaswani et al. 2017): LayerNorm(x + Dropout(Sublayer(x)))
            attn_out, _ = self.self_attn(x, x, x, mask=mask)
            x = self.norm1(x + self.dropout1(attn_out))

            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_out))

        return x



class TransformerEncoder(nn.Module):
    r"""
    Full Transformer Encoder Stack.

    Stacks :math:`N` identical `TransformerEncoderLayer` modules preceded by
    token embedding lookup and sinusoidal positional encoding injection:

    1. Token Embedding: :math:`E \in \mathbb{R}^{\text{vocab\_size} \times d_{model}}`
    2. Embedding Scaling: Multiply by :math:`\sqrt{d_{model}}` (Vaswani et al. Section 3.4)
    3. Positional Encoding: Add sinusoidal positional table + dropout
    4. Encoder Stack: Pass through :math:`N` sequential encoder layers
    5. Final Layer Normalization: Applied to stabilize the output representations

    Args:
        config (TransformerConfig): Configuration dataclass containing all hyperparameters.

    Shape:
        - Tokens Input: :math:`(B, T)` with integer indices in range :math:`[0, \text{vocab\_size}-1]`
        - Mask (optional): :math:`(B, 1, 1, T)` or :math:`(B, 1, T, T)`
        - Output: :math:`(B, T, d_{model})`
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_scale = math.sqrt(config.d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Stack of N identical encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm_first=config.norm_first,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Execute full encoding pipeline from token indices to contextual representations.

        Args:
            tokens: Integer tensor of token IDs, shape (batch_size, seq_len).
            mask: Optional attention mask broadcastable to
                (batch_size, n_heads, seq_len, seq_len).

        Returns:
            Encoded representation tensor of shape (batch_size, seq_len, d_model).
        """
        # 1. Token Embedding (Lookup)
        x = self.token_embedding(tokens)
        # 2. Embedding Scaling
        x = x * self.embedding_scale
        # 3. Positional Encoding
        x = self.pos_encoder(x)
        # 4. Stack of N encoder layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        # 5. Final layer normalization
        x = self.final_norm(x)
        return x
