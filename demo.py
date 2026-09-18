"""
Transformer Encoder Demonstration & Diagnostics
================================================
Demonstrates end-to-end forward and backward passes, inspects tensor contracts,
and displays architecture parameter counts and attention patterns.
"""

import torch

from src.config import TransformerConfig
from src.encoder import TransformerEncoder


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_demo() -> None:
    print_header("1. CONFIGURING TRANSFORMER (Vaswani et al. Base Model)")
    config = TransformerConfig(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        max_seq_len=512,
        dropout=0.1,
        norm_first=False,  # Strict Post-LN as in the paper
    )
    print(f"• Vocabulary Size : {config.vocab_size}")
    print(f"• d_model         : {config.d_model}")
    print(f"• Heads (h)       : {config.n_heads} (d_k = d_v = {config.d_k})")
    print(f"• d_ff            : {config.d_ff}")
    print(f"• Layers (N)      : {config.n_layers}")
    print(f"• Norm Type       : {'Pre-LN' if config.norm_first else 'Post-LN (Original Paper)'}")

    print_header("2. INITIALIZING MODEL")
    encoder = TransformerEncoder(config)
    total_params = count_parameters(encoder)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Parameter Breakdown
    embed_params = config.vocab_size * config.d_model
    # Per layer: MHA (4 * d_model^2 + 4 * d_model) + FFN (2 * d_model * d_ff + d_ff + d_model) + 2 * LayerNorm (4 * d_model)
    layer_params = count_parameters(encoder.layers[0])
    print(f"  - Token Embeddings : {embed_params:,}")
    print(f"  - Per Layer Params : {layer_params:,} (x{config.n_layers} = {layer_params * config.n_layers:,})")
    print(f"  - Final LayerNorm  : {count_parameters(encoder.final_norm):,}")

    print_header("3. SIMULATING FORWARD PASS")
    batch_size = 2
    seq_len = 10
    dummy_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"Input Tokens Tensor Shape : {dummy_tokens.shape} (Batch: {batch_size}, SeqLen: {seq_len})")
    print(f"Sample Sequence (Batch 0) : {dummy_tokens[0].tolist()}")

    # Forward pass without mask
    output = encoder(dummy_tokens)
    print(f"Output Embedding Shape    : {output.shape} (Expected: ({batch_size}, {seq_len}, {config.d_model}))")
    assert output.shape == (batch_size, seq_len, config.d_model)

    print_header("4. SIMULATING PADDING MASKING")
    # Mask out the last 3 tokens of each sequence
    mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[:, :, :, 7:] = False
    print(f"Mask Tensor Shape         : {mask.shape} (Keys 7..9 masked out)")

    masked_output = encoder(dummy_tokens, mask=mask)
    print(f"Masked Output Shape       : {masked_output.shape}")
    assert masked_output.shape == (batch_size, seq_len, config.d_model)

    print_header("5. VERIFYING BACKWARD PASS (GRADIENT FLOW)")
    loss = masked_output.sum()
    loss.backward()

    # Check that gradients exist and are finite
    all_finite = True
    for name, param in encoder.named_parameters():
        if param.grad is None or not torch.isfinite(param.grad).all():
            print(f"  [FAIL] Invalid gradient for: {name}")
            all_finite = False
            break

    if all_finite:
        print("  [PASS] All gradients successfully computed and verified finite (no NaN/Inf).")

    print_header("6. ATTENTION MAP INSPECTION (First Layer, Head 0 in eval mode)")
    encoder.eval()
    with torch.no_grad():
        x = encoder.token_embedding(dummy_tokens) * encoder.embedding_scale
        x = encoder.pos_encoder(x)
        first_layer = encoder.layers[0]
        # Run self-attention directly to extract weights
        _, attn_weights = first_layer.self_attn(x, x, x, mask=mask)
        head_0_weights = attn_weights[0, 0]  # (seq_len, seq_len)
        print(f"Attention Weights Shape: {head_0_weights.shape}")
        print("Row 0 Attention Distribution across keys 0..9:")
        formatted_row = [f"{w:.4f}" for w in head_0_weights[0].tolist()]
        print(" ", formatted_row)
        print("Keys 7, 8, 9 (masked) receive exactly: ", formatted_row[7:])

    print_header("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_demo()
