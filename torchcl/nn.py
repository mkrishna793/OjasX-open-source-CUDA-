"""
OjasX Neural Network Modules — PyTorch nn.Module subclasses that
run forward/backward passes on OpenCL.

These modules manage weight tensors on the OpenCL device and use
the autograd functions from torchcl.autograd for gradient computation.

Usage:
    import torchcl
    from torchcl.nn import OpenCLLinear, OpenCLLayerNorm

    linear = OpenCLLinear(768, 512)
    ln = OpenCLLayerNorm(512)

    x = torchcl.to_opencl(torch.randn(32, 768))
    y = linear(x)
    y = ln(y)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torchcl.ops.engine import get_engine
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.api import (
    _get_buf,
    _get_shape,
    _wrap_output,
    to_opencl,
    to_cpu,
    is_opencl_tensor,
)
from torchcl.autograd import (
    ReluFunction,
    SigmoidFunction,
    TanhFunction,
    GeluFunction,
    SiluFunction,
    LeakyReluFunction,
    LinearFunction,
    MatmulFunction,
    AddFunction,
    LayerNormFunction,
    CrossEntropyFunction,
    Conv2dFunction,
)


# ═══════════════════════════════════════════════════════════════════════
# Linear Layer
# ═══════════════════════════════════════════════════════════════════════

class OpenCLLinear(nn.Module):
    """Fully-connected layer running on OpenCL.

    Computes: output = input @ weight.T + bias
    Weight and bias are uploaded to GPU on first forward pass.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights (Kaiming uniform like PyTorch default)
        k = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-k, k)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features).uniform_(-k, k))
        else:
            self.bias = None

        self._weight_gpu = None
        self._bias_gpu = None

    def _ensure_on_gpu(self) -> None:
        """Upload weight and bias to GPU if not already there."""
        if self._weight_gpu is None or not is_opencl_tensor(self._weight_gpu):
            self._weight_gpu = to_opencl(self.weight.data)
        if self.bias is not None:
            if self._bias_gpu is None or not is_opencl_tensor(self._bias_gpu):
                self._bias_gpu = to_opencl(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_on_gpu()
        return LinearFunction.apply(x, self._weight_gpu, self._bias_gpu)


# ═══════════════════════════════════════════════════════════════════════
# Layer Normalization
# ═══════════════════════════════════════════════════════════════════════

class OpenCLLayerNorm(nn.Module):
    """Layer normalization on OpenCL.

    Normalizes across the last dimension (normalized_shape).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self._weight_gpu = None
        self._bias_gpu = None

    def _ensure_on_gpu(self) -> None:
        if self._weight_gpu is None or not is_opencl_tensor(self._weight_gpu):
            self._weight_gpu = to_opencl(self.weight.data)
        if self._bias_gpu is None or not is_opencl_tensor(self._bias_gpu):
            self._bias_gpu = to_opencl(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_on_gpu()
        return LayerNormFunction.apply(
            x, self._weight_gpu, self._bias_gpu,
            self.normalized_shape, self.eps,
        )


# ═══════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════

class OpenCLEmbedding(nn.Module):
    """Embedding lookup table on OpenCL.

    Maps integer token indices to dense vectors.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self._weight_gpu = None

    def _ensure_on_gpu(self) -> None:
        if self._weight_gpu is None or not is_opencl_tensor(self._weight_gpu):
            self._weight_gpu = to_opencl(self.weight.data)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for given indices.

        Args:
            indices: 1D tensor of integer indices (on CPU).
        Returns:
            OpenCL tensor of shape [len(indices), embedding_dim].
        """
        self._ensure_on_gpu()
        engine = get_engine()

        if indices.is_floating_point():
            indices_float = indices.float()
        else:
            indices_float = indices.float()

        indices_cl = to_opencl(indices_float)

        N = indices.numel()
        D = self.embedding_dim
        out_shape = (N, D)
        out_handle, out_buf = _alloc_helper(out_shape)

        engine.run_embedding(
            _get_buf(self._weight_gpu),
            _get_buf(indices_cl),
            out_buf, N, D,
        )
        return _wrap_output(out_buf, out_shape)


# ═══════════════════════════════════════════════════════════════════════
# Convenience Activation Modules
# ═══════════════════════════════════════════════════════════════════════

class OpenCLReLU(nn.Module):
    """ReLU activation on OpenCL with autograd support."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ReluFunction.apply(x)


class OpenCLGELU(nn.Module):
    """GELU activation on OpenCL with autograd support."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GeluFunction.apply(x)


class OpenCLSiLU(nn.Module):
    """SiLU/Swish activation on OpenCL with autograd support."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SiluFunction.apply(x)


# ═══════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════

def _alloc_helper(shape: tuple, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, CLBuffer]:
    """Allocate an OpenCL buffer and return (handle, cl_buf)."""
    engine = get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    handle = _wrap_output(cl_buf, shape, dtype)
    return handle, cl_buf


def _add_bias(
    result: torch.Tensor,
    bias_gpu: torch.Tensor,
    batch: int,
    out_features: int,
) -> torch.Tensor:
    """Add bias vector to each row of a [batch, out_features] matrix.

    Uses elementwise add by tiling the bias across the batch dimension.
    This is done row-by-row to avoid needing a broadcast kernel.
    """
    engine = get_engine()
    pool = get_buffer_pool()

    result_shape = _get_shape(result)
    n = int(np.prod(result_shape))

    # Create a buffer with bias repeated for each batch row
    bias_data = to_cpu(bias_gpu).numpy()
    tiled_bias = np.tile(bias_data, (batch, 1)).astype(np.float32)
    tiled_buf = pool.host_to_device(tiled_bias)
    tiled_handle = _wrap_output(tiled_buf, result_shape)

    return AddFunction.apply(result, tiled_handle)


class OpenCLConv2d(nn.Module):
    """2D Convolution layer running on OpenCL."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

        # Initialize weight parameter [C_out, C_in, kH, kW]
        k = 1.0 / math.sqrt(in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-k, k)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).uniform_(-k, k))
        else:
            self.bias = None

        self._weight_gpu = None
        self._bias_gpu = None

    def _ensure_on_gpu(self) -> None:
        if self._weight_gpu is None or not is_opencl_tensor(self._weight_gpu):
            self._weight_gpu = to_opencl(self.weight.data)
        if self.bias is not None:
            if self._bias_gpu is None or not is_opencl_tensor(self._bias_gpu):
                self._bias_gpu = to_opencl(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_on_gpu()
        return Conv2dFunction.apply(
            x, self._weight_gpu, self._bias_gpu,
            self.stride, self.padding
        )
