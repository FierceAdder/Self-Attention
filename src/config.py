"""
Model Configuration
===================
Configuration dataclass defining hyperparameter specifications for the
Transformer architecture as presented in Vaswani et al. (2017).
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """
    Hyperparameter configuration for the Transformer Encoder.

    Attributes:
        vocab_size (int): Size of the input vocabulary.
            Default: 30000.
        d_model (int): Dimensionality of the input and hidden representations.
            Must be divisible by `n_heads`.
            Default: 512 (Vaswani et al. base model).
        n_heads (int): Number of parallel attention heads.
            Default: 8 (Vaswani et al. base model).
        d_ff (int): Dimensionality of the inner position-wise feed-forward layer.
            Typically set to 4 * d_model.
            Default: 2048 (Vaswani et al. base model).
        n_layers (int): Number of stacked Transformer encoder layers.
            Default: 6 (Vaswani et al. base model).
        max_seq_len (int): Maximum sequence length for positional encodings.
            Default: 5000.
        dropout (float): Dropout probability applied to attention weights,
            sublayer outputs, and embeddings.
            Default: 0.1.
        norm_first (bool): If True, applies Pre-Layer Normalization
            (x + SubLayer(LN(x))), which stabilizes gradient dynamics.
            If False, applies Post-Layer Normalization (LN(x + SubLayer(x)))
            strictly following the original Vaswani et al. (2017) specification.
            Default: False (strict paper adherence).
    """
    vocab_size: int = 30000
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    max_seq_len: int = 5000
    dropout: float = 0.1
    norm_first: bool = False

    def __post_init__(self):
        """Validate architectural invariants."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}) "
                f"to split channels evenly across heads (d_k = d_model // n_heads)."
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in range [0.0, 1.0), got {self.dropout}")

    @property
    def d_k(self) -> int:
        """Dimensionality of query and key projections per attention head."""
        return self.d_model // self.n_heads

    @property
    def d_v(self) -> int:
        """Dimensionality of value projections per attention head."""
        return self.d_model // self.n_heads
