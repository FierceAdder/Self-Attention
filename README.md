# Transformer Encoder & Multi-Head Attention from Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous, modular implementation of **Scaled Dot-Product Attention**, **Multi-Head Self-Attention (MHA)**, and a full **Transformer Encoder** built from fundamental PyTorch tensor operations, strictly adhering to the seminal paper:

> **Attention Is All You Need**  
> *Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin*  
> NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## Table of Contents
1. [Theoretical Architecture & Mathematics](#theoretical-architecture--mathematics)
   - [Scaled Dot-Product Attention](#1-scaled-dot-product-attention)
   - [Multi-Head Attention (MHA)](#2-multi-head-attention-mha)
   - [Sinusoidal Positional Encoding](#3-sinusoidal-positional-encoding)
   - [Position-wise Feed-Forward Network (FFN)](#4-position-wise-feed-forward-network-ffn)
   - [Transformer Encoder Layer & Stack](#5-transformer-encoder-layer--stack)
2. [Tensor Shape Contracts](#tensor-shape-contracts)
3. [Project Directory Structure](#project-directory-structure)
4. [Commit Message Standards](#commit-message-standards)
5. [Testing & Verification](#testing--verification)

---

## Theoretical Architecture & Mathematics

### 1. Scaled Dot-Product Attention

Given query matrix $Q \in \mathbb{R}^{T_q \times d_k}$, key matrix $K \in \mathbb{R}^{T_k \times d_k}$, and value matrix $V \in \mathbb{R}^{T_k \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

Where:
- $T_q, T_k$: Sequence lengths of queries and keys (in self-attention, $T_q = T_k = T$).
- $d_k, d_v$: Projection dimensions of keys/queries and values (typically $d_k = d_v = d_{model} / h$).
- $M$: Optional mask tensor where masked elements are set to $-\infty$ (or $-1e9$) to yield zero probability after softmax.

#### Why Scale by $1 / \sqrt{d_k}$?
Assume components of $q \in \mathbb{R}^{d_k}$ and $k \in \mathbb{R}^{d_k}$ are independent random variables with mean $0$ and variance $1$.  
The dot product is:
$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$
- $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$
- $\text{Var}(q_i k_i) = \text{Var}(q_i)\text{Var}(k_i) = 1 \times 1 = 1$
- By linearity of variance for independent variables:
  $$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

Without scaling, as $d_k$ grows large, the variance of the logits grows proportionally to $d_k$, leading to extreme values where $\text{softmax}$ saturates. This drives the gradients $\frac{\partial \text{softmax}}{\partial z}$ toward zero (vanishing gradient problem). Dividing by $\sqrt{d_k}$ normalizes the variance back to $1$:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(q \cdot k) = 1$$

---

### 2. Multi-Head Attention (MHA)

Instead of performing a single attention function with $d_{model}$-dimensional queries, keys, and values, Multi-Head Attention projects $Q, K, V$ with $h$ different learned linear projections to $d_k, d_k, d_v$ dimensions respectively:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)$$

Where parameter matrices:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{model}}$

In practice, all heads are computed in parallel using batched matrix multiplications by packing projections into single weight matrices $W^Q, W^K, W^V \in \mathbb{R}^{d_{model} \times (h \cdot d_k)}$ and reshaping tensors:

$$\underbrace{(B, T, d_{model})}_{\text{Linear projection}} \longrightarrow (B, T, h, d_k) \xrightarrow{\text{transpose(1, 2)}} \underbrace{(B, h, T, d_k)}_{\text{Ready for batched } QK^T}$$

---

### 3. Sinusoidal Positional Encoding

Because self-attention is permutation-equivariant (it contains no inherent notion of sequence order), positional encodings are added to the input embeddings:

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where $pos \in [0, \dots, T - 1]$ is the token position ($T$ is the sequence length) and $i \in [0, \dots, d_{\text{model}}/2 - 1]$ is the channel index.

**Key Property**: For any fixed offset $k$, $\text{PE}_{pos+k}$ can be represented as a linear function of $\text{PE}_{pos}$ via trigonometric angle addition formulas, facilitating the model in learning relative positional relationships.

---

### 4. Position-wise Feed-Forward Network (FFN)

Applied to each position separately and identically:

$$\text{FFN}(x) = \max\left(0, x W_1 + b_1\right) W_2 + b_2$$

Where $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ and $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ (typically $d_{ff} = 4 \times d_{model}$).

---

### 5. Transformer Encoder Layer & Stack

Each Encoder Layer consists of two sub-layers:
1. **Multi-Head Self-Attention**
2. **Position-wise Feed-Forward Network**

Surrounding each sublayer is a **residual connection** followed by **Layer Normalization**:

- **Post-LN (Original Vaswani et al., 2017)**:
  $$x^{(1)} = \text{LayerNorm}(x + \text{Dropout}(\text{MHA}(x, x, x)))$$
  $$x^{(2)} = \text{LayerNorm}(x^{(1)} + \text{Dropout}(\text{FFN}(x^{(1)})))$$

- **Pre-LN (Modern Standard)**:
  $$x^{(1)} = x + \text{Dropout}(\text{MHA}(\text{LayerNorm}(x)))$$
  $$x^{(2)} = x^{(1)} + \text{Dropout}(\text{FFN}(\text{LayerNorm}(x^{(1)})))$$

The full Encoder stacks $N$ identical layers, preceded by input embedding addition with positional encoding (scaled by $\sqrt{d_{model}}$).

---

## Tensor Shape Contracts

| Module | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `ScaledDotProductAttention` | $Q, K, V: (B, h, T, d_k)$ | $(B, h, T, d_k)$ | Attention weights: $(B, h, T, T)$ |
| `MultiHeadAttention` | $x: (B, T, d_{model})$ | $(B, T, d_{model})$ | Supports optional mask: $(B, 1, 1, T)$ |
| `PositionalEncoding` | $x: (B, T, d_{model})$ | $(B, T, d_{model})$ | Added elementwise to input |
| `PositionwiseFeedForward` | $x: (B, T, d_{model})$ | $(B, T, d_{model})$ | Expands to $d_{ff}$, projects back |
| `TransformerEncoderLayer` | $x: (B, T, d_{model})$ | $(B, T, d_{model})$ | Preserves shape with residual |
| `TransformerEncoder` | `tokens: (B, T)` | $(B, T, d_{model})$ | End-to-end token sequence encoding |

---

## Project Directory Structure

```
Attention/
├── src/
│   ├── __init__.py
│   ├── config.py              # Architecture hyperparameters (dataclass)
│   ├── attention.py           # ScaledDotProductAttention & MultiHeadAttention
│   ├── positional_encoding.py # Sinusoidal Positional Encoding
│   ├── encoder.py             # FeedForward, EncoderLayer, TransformerEncoder
│   └── utils.py               # Masking generators & shape validation
├── tests/
│   ├── __init__.py
│   ├── test_attention.py      # Numerical, shape, and masking tests
│   ├── test_positional.py     # Deterministic & frequency tests
│   └── test_encoder.py        # Gradient flow & end-to-end tests
├── demo.py                    # Quickstart verification demo
├── requirements.txt           # Minimal dependencies
├── .gitignore                 # Python clean ignore
└── README.md                  # Complete mathematical & design documentation
```


---

## Commit Message Standards

Every commit in this repository follows the structured format below so evaluators can inspect the exact architectural rationale and contract changes without reading diffs:

```text
<type>(<scope>): <high-level imperative summary>

Rationale & Architectural Overview:
- Why this component was designed this way and what formulas govern it.

Tensor Contracts & Shapes:
- Exact tensor inputs, shapes, transformations, and outputs.

Verification:
- Automated tests run, shapes verified, and pass confirmation.
```

---

## Testing & Verification

Run the comprehensive unit test suite:
```bash
pytest -v tests/
```

Run the architecture demonstration and diagnostics:
```bash
python demo.py
```
