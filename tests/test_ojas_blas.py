"""
Test Suite for ojasBLAS: High-Performance Sub-Cubic & Hardware-Aware Matrix Engine.
"""

import pytest
import torch
import torchcl
import torchcl.blas as blas


def test_ojas_blas_gemm_fused():
    torch.manual_seed(42)
    A = torch.randn(64, 128)
    B = torch.randn(128, 64)
    bias = torch.randn(64)

    expected = torch.relu(A @ B + bias)

    out_cl = blas.gemm(A, B, bias=bias, activation="relu")
    out_cpu = torchcl.to_cpu(out_cl)

    diff = (out_cpu - expected).abs().max().item()
    assert diff < 1e-3, f"Difference too high in fused GEMM: {diff}"


def test_ojas_blas_winograd_gemm():
    torch.manual_seed(42)
    A = torch.randn(64, 64)
    B = torch.randn(64, 64)

    expected = A @ B

    out_cl = blas.winograd_gemm(A, B)
    out_cpu = torchcl.to_cpu(out_cl)

    diff = (out_cpu - expected).abs().max().item()
    assert diff < 1e-3, f"Difference too high in Winograd GEMM: {diff}"


def test_ojas_blas_sparse_gemm():
    torch.manual_seed(42)
    # Create sparse activation matrix (60% zeros)
    A = torch.relu(torch.randn(64, 64) - 0.2)
    B = torch.randn(64, 64)

    expected = A @ B

    out_cl = blas.sparse_gemm(A, B)
    out_cpu = torchcl.to_cpu(out_cl)

    diff = (out_cpu - expected).abs().max().item()
    assert diff < 1e-3, f"Difference too high in Sparse GEMM: {diff}"


def test_ojas_blas_bmm():
    torch.manual_seed(42)
    A = torch.randn(4, 32, 64)
    B = torch.randn(4, 64, 32)

    expected = torch.bmm(A, B)

    out_cl = blas.bmm(A, B)
    out_cpu = torchcl.to_cpu(out_cl)

    diff = (out_cpu - expected).abs().max().item()
    assert diff < 1e-3, f"Difference too high in Batched GEMM: {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
