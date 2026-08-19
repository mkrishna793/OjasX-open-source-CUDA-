"""
ojasBLAS — High-Performance Sub-Cubic & Hardware-Aware Matrix Math Library for OjasX.
Replaces cuBLAS / CUTLASS with 2D Register Tiling, Winograd Transforms, and Zero-Skipping.
"""

from __future__ import annotations
import numpy as np
import torch
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor, _get_buf, _wrap_output, _make_handle
from torchcl.ops.engine import get_engine
from torchcl.runtime.context import get_queue


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Universal 2D Register-Blocked SGEMM with fused bias and activation."""
    if not is_opencl_tensor(a): a = to_opencl(a)
    if not is_opencl_tensor(b): b = to_opencl(b)
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"

    engine = get_engine()
    out_buf = engine._pool.allocate(M * N * 4)

    if bias is not None:
        if not is_opencl_tensor(bias): bias = to_opencl(bias)
        engine.run_matmul_bias(_get_buf(a), _get_buf(b), _get_buf(bias), out_buf, M, N, K)
    else:
        engine.run_matmul(_get_buf(a), _get_buf(b), out_buf, M, N, K)

    out = _wrap_output(out_buf, (M, N))

    if activation == "relu":
        out = torchcl.relu(out)
    elif activation == "gelu":
        out = torchcl.gelu(out)
    elif activation == "silu":
        out = torchcl.silu(out)
    return out


def winograd_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sub-Cubic Winograd Matrix Multiplication O(N^2.81) in registers."""
    if not is_opencl_tensor(a): a = to_opencl(a)
    if not is_opencl_tensor(b): b = to_opencl(b)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"

    engine = get_engine()
    out_buf = engine._pool.allocate(M * N * 4)
    kernel = engine._registry.get_kernel("winograd_gemm.cl", "matmul_winograd_2x2_f32")
    
    queue = get_queue()
    global_size = (((N + 1) // 2), ((M + 1) // 2))
    local_size = None

    kernel(queue, global_size, local_size,
           _get_buf(a).buffer, _get_buf(b).buffer, out_buf.buffer,
           np.int32(M), np.int32(N), np.int32(K))

    return _wrap_output(out_buf, (M, N))


def sparse_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitmask Zero-Skipping GEMM (Skips zero activations dynamically)."""
    if not is_opencl_tensor(a): a = to_opencl(a)
    if not is_opencl_tensor(b): b = to_opencl(b)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"

    engine = get_engine()
    out_buf = engine._pool.allocate(M * N * 4)
    kernel = engine._registry.get_kernel("winograd_gemm.cl", "matmul_zero_skip_f32")
    
    queue = get_queue()
    global_size = (N, M)
    local_size = None

    kernel(queue, global_size, local_size,
           _get_buf(a).buffer, _get_buf(b).buffer, out_buf.buffer,
           np.int32(M), np.int32(N), np.int32(K))

    return _wrap_output(out_buf, (M, N))


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched Matrix Multiplication: [B, M, K] @ [B, K, N] -> [B, M, N]."""
    if not is_opencl_tensor(a): a = to_opencl(a)
    if not is_opencl_tensor(b): b = to_opencl(b)

    assert a.dim() == 3 and b.dim() == 3, "bmm requires 3D tensors"
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2, f"Shape mismatch in bmm: {a.shape} vs {b.shape}"

    engine = get_engine()
    out_buf = engine._pool.allocate(B * M * N * 4)
    kernel = engine._registry.get_kernel("winograd_gemm.cl", "matmul_batched_strided_f32")

    queue = get_queue()
    global_size = (N, M, B)
    local_size = None

    stride_a = M * K
    stride_b = K * N
    stride_c = M * N

    kernel(queue, global_size, local_size,
           _get_buf(a).buffer, _get_buf(b).buffer, out_buf.buffer,
           np.int32(B), np.int32(M), np.int32(N), np.int32(K),
           np.int32(stride_a), np.int32(stride_b), np.int32(stride_c))

    return _wrap_output(out_buf, (B, M, N))


__all__ = ["gemm", "winograd_gemm", "sparse_gemm", "bmm"]
