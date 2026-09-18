"""
Positional Encodings
====================
Injects information about token order into representations using
fixed sinusoidal functions as formulated in Vaswani et al. (2017, Section 3.5).
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    r"""
    Sinusoidal Positional Encoding module.

    Because self-attention is permutation-equivariant, positional encodings are
    added directly to the token embeddings to supply order information:

    .. math::
        PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)

    .. math::
        PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)

    where:
        - :math:`pos \in [0, \dots, \text{max\_seq\_len}-1]` is the position index.
        - :math:`i \in [0, \dots, d_{model}/2 - 1]` is the embedding dimension index.

    Numerical Computation in Log-Space:
    To prevent numerical instability and division by large powers, the denominator
    is computed in log-space:

    .. math::
        \frac{1}{10000^{2i/d_{model}}} = \exp\left(-\frac{2i}{d_{model}} \cdot \ln(10000)\right)

    Registered Buffer:
    Positional encodings are fixed (non-learnable), but need to be saved/loaded
    with `state_dict` and moved automatically between devices (CPU/GPU) with the module.
    Therefore, they are registered using `register_buffer('pe', pe)`.

    Args:
        d_model (int): Embedding dimension. Must be an even integer.
        max_seq_len (int, optional): Maximum supported sequence length. Defaults to 5000.
        dropout (float, optional): Dropout probability applied after adding PE. Defaults to 0.1.

    Raises:
        ValueError: If `d_model` is not an even integer.

    Shape:
        - Input: :math:`(B, T, d_{model})`
        - Output: :math:`(B, T, d_{model})`

    Examples:
        >>> pe = SinusoidalPositionalEncoding(d_model=512, max_seq_len=100)
        >>> x = torch.zeros(2, 10, 512)
        >>> out = pe(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal encoding, got {d_model}")

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Add positional encodings to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor with positional encodings added and dropout applied,
            of shape (batch_size, seq_len, d_model).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
