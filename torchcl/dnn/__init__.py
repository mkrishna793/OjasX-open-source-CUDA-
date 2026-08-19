"""
ojasDNN — High-Performance Deep Neural Network & Attention Primitives for OjasX.
Replaces cuDNN & FlashAttention with Categorical Monoidal Attention (CMA) and Workgroup-Parallel Reductions.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor, _get_buf, _wrap_output


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused FlashAttention-2: Tiled Online-Softmax Attention across Q, K, V [B, H, S, d]."""
    if not is_opencl_tensor(q): q = to_opencl(q)
    if not is_opencl_tensor(k): k = to_opencl(k)
    if not is_opencl_tensor(v): v = to_opencl(v)
    return torchcl.fused_attention(q, k, v, scale=scale, is_causal=is_causal)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Workgroup-Parallel RMSNorm for LLaMA/Mistral architectures."""
    if not is_opencl_tensor(x): x = to_opencl(x)
    if not is_opencl_tensor(weight): weight = to_opencl(weight)
    dim = x.shape[-1]
    return torchcl.rms_norm(x, weight, dim, eps=eps)


def layer_norm(
    x: torch.Tensor,
    normalized_shape: int | tuple[int, ...],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Workgroup-Parallel LayerNorm with local SRAM tree reductions."""
    if not is_opencl_tensor(x): x = to_opencl(x)
    if weight is not None and not is_opencl_tensor(weight): weight = to_opencl(weight)
    if bias is not None and not is_opencl_tensor(bias): bias = to_opencl(bias)
    return torchcl.layer_norm(x, normalized_shape, weight, bias, eps)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU Activation: out = SiLU(gate) * up."""
    if not is_opencl_tensor(gate): gate = to_opencl(gate)
    if not is_opencl_tensor(up): up = to_opencl(up)
    return torchcl.swiglu(gate, up)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary Position Embeddings (RoPE)."""
    if not is_opencl_tensor(x): x = to_opencl(x)
    if not is_opencl_tensor(cos): cos = to_opencl(cos)
    if not is_opencl_tensor(sin): sin = to_opencl(sin)
    return torchcl.rope(x, cos, sin)


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    """High-Performance 2D Convolution."""
    if not is_opencl_tensor(input): input = to_opencl(input)
    if not is_opencl_tensor(weight): weight = to_opencl(weight)
    if bias is not None and not is_opencl_tensor(bias): bias = to_opencl(bias)
    return torchcl.conv2d(input, weight, bias, stride=stride, padding=padding)


class TransformerBlock(nn.Module):
    """Modern LLaMA-style Transformer Block executing purely in OpenCL GPU VRAM."""
    def __init__(self, dim: int = 256, num_heads: int = 8, hidden_dim: int = 512, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.gamma2 = nn.Parameter(torch.ones(dim))
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # 1. Norm & Attention
        h = rms_norm(x, self.gamma1, eps=self.eps)
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = flash_attention(q, k, v, is_causal=True)
        attn_flat = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.out_proj(attn_flat)

        # 2. Norm & SwiGLU MLP
        h2 = rms_norm(x, self.gamma2, eps=self.eps)
        gate = self.gate_proj(h2)
        up = self.up_proj(h2)
        mlp_out = self.down_proj(swiglu(gate, up))
        x = x + mlp_out

        return x


__all__ = [
    "flash_attention",
    "rms_norm",
    "layer_norm",
    "swiglu",
    "rope",
    "conv2d",
    "TransformerBlock",
]
