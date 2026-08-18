"""
TorchCL Smoke Test — Verifies all core operations work correctly
by comparing OpenCL results against CPU PyTorch results.
"""

import time
import torch
import numpy as np
import pytest
import torchcl


def test_data_movement():
    a_cpu = torch.randn(100, 100)
    a_cl = torchcl.to_opencl(a_cpu)
    a_back = torchcl.to_cpu(a_cl)
    assert torch.allclose(a_back, a_cpu, atol=1e-4)
    assert torchcl.is_opencl_tensor(a_cl) is True
    assert torchcl.is_opencl_tensor(a_cpu) is False


def test_tensor_creation():
    z = torchcl.zeros(3, 3)
    assert torch.allclose(torchcl.to_cpu(z), torch.zeros(3, 3), atol=1e-4)

    o = torchcl.ones(4, 4)
    assert torch.allclose(torchcl.to_cpu(o), torch.ones(4, 4), atol=1e-4)

    f = torchcl.full(2, 3, fill_value=7.0)
    assert torch.allclose(torchcl.to_cpu(f), torch.full((2, 3), 7.0), atol=1e-4)

    r = torchcl.randn(5, 5)
    r_cpu = torchcl.to_cpu(r)
    assert r_cpu.shape == (5, 5)


def test_arithmetic():
    a = torch.randn(256, 256)
    b = torch.randn(256, 256)
    a_cl = torchcl.to_opencl(a)
    b_cl = torchcl.to_opencl(b)

    assert torch.allclose(torchcl.to_cpu(torchcl.add(a_cl, b_cl)), a + b, atol=1e-4)
    assert torch.allclose(torchcl.to_cpu(torchcl.sub(a_cl, b_cl)), a - b, atol=1e-4)
    assert torch.allclose(torchcl.to_cpu(torchcl.mul(a_cl, b_cl)), a * b, atol=1e-4)
    assert torch.allclose(torchcl.to_cpu(torchcl.div(a_cl, b_cl)), a / b, atol=1e-3)
    assert torch.allclose(torchcl.to_cpu(torchcl.neg(a_cl)), -a, atol=1e-4)

    a_pos = torch.rand(100, 100) + 0.1
    a_pos_cl = torchcl.to_opencl(a_pos)
    assert torch.allclose(torchcl.to_cpu(torchcl.abs_(a_cl)), torch.abs(a), atol=1e-4)
    assert torch.allclose(torchcl.to_cpu(torchcl.exp(a_cl)), torch.exp(a), atol=1e-3)
    assert torch.allclose(torchcl.to_cpu(torchcl.log(a_pos_cl)), torch.log(a_pos), atol=1e-3)
    assert torch.allclose(torchcl.to_cpu(torchcl.sqrt(a_pos_cl)), torch.sqrt(a_pos), atol=1e-3)


def test_activations():
    x = torch.randn(128, 128)
    x_cl = torchcl.to_opencl(x)

    assert torch.allclose(torchcl.to_cpu(torchcl.relu(x_cl)), torch.relu(x), atol=1e-4)
    assert torch.allclose(torchcl.to_cpu(torchcl.sigmoid(x_cl)), torch.sigmoid(x), atol=1e-3)
    assert torch.allclose(torchcl.to_cpu(torchcl.tanh_(x_cl)), torch.tanh(x), atol=1e-3)
    assert torch.allclose(torchcl.to_cpu(torchcl.gelu(x_cl)), torch.nn.functional.gelu(x), atol=1e-2)
    assert torch.allclose(torchcl.to_cpu(torchcl.silu(x_cl)), torch.nn.functional.silu(x), atol=1e-3)
    assert torch.allclose(torchcl.to_cpu(torchcl.leaky_relu(x_cl, 0.01)), torch.nn.functional.leaky_relu(x, 0.01), atol=1e-3)

    s = torch.randn(8, 16)
    s_cl = torchcl.to_opencl(s)
    assert torch.allclose(torchcl.to_cpu(torchcl.softmax(s_cl)), torch.softmax(s, dim=-1), atol=1e-3)


def test_matrix_operations():
    m1 = torch.randn(64, 128)
    m2 = torch.randn(128, 32)
    m1_cl = torchcl.to_opencl(m1)
    m2_cl = torchcl.to_opencl(m2)
    assert torch.allclose(torchcl.to_cpu(torchcl.matmul(m1_cl, m2_cl)), m1 @ m2, atol=1e-2)

    m3 = torch.randn(16, 32)
    m3_cl = torchcl.to_opencl(m3)
    assert torch.allclose(torchcl.to_cpu(torchcl.transpose(m3_cl)), m3.T, atol=1e-4)


def test_reductions():
    r = torch.randn(1024)
    r_cl = torchcl.to_opencl(r)
    assert abs(torchcl.to_cpu(torchcl.sum_(r_cl)).item() - r.sum().item()) < 0.5
    assert abs(torchcl.to_cpu(torchcl.mean(r_cl)).item() - r.mean().item()) < 0.1
    assert abs(torchcl.to_cpu(torchcl.max_(r_cl)).item() - r.max().item()) < 1e-3
    assert abs(torchcl.to_cpu(torchcl.min_(r_cl)).item() - r.min().item()) < 1e-3


def test_performance_matmul():
    big_a = torch.randn(512, 512)
    big_b = torch.randn(512, 512)
    big_a_cl = torchcl.to_opencl(big_a)
    big_b_cl = torchcl.to_opencl(big_b)

    result_cl = torchcl.matmul(big_a_cl, big_b_cl)
    torchcl.synchronize()
    result_cpu = big_a @ big_b
    assert torch.allclose(torchcl.to_cpu(result_cl), result_cpu, atol=0.5)


if __name__ == "__main__":
    print("=" * 60)
    print("  TorchCL V1 - Comprehensive Smoke Test")
    print("=" * 60)
    test_data_movement()
    print("  [PASS] Data Movement")
    test_tensor_creation()
    print("  [PASS] Tensor Creation")
    test_arithmetic()
    print("  [PASS] Arithmetic")
    test_activations()
    print("  [PASS] Activations")
    test_matrix_operations()
    print("  [PASS] Matrix Operations")
    test_reductions()
    print("  [PASS] Reductions")
    test_performance_matmul()
    print("  [PASS] Performance Matmul (512x512)")
    print("\nALL SMOKE TESTS PASSED!")
