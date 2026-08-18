"""
TorchCL JIT Compiler Test — Tests fused kernel generation and execution.
"""

import numpy as np
import pytest
import torchcl
from torchcl.jit.compiler import get_jit_compiler
from torchcl.jit.tuner import get_auto_tuner
from torchcl.runtime.memory import get_buffer_pool
from torchcl.runtime.context import get_queue
from torchcl.jit.cache import get_kernel_cache


def test_fused_unary_chains():
    jit = get_jit_compiler()
    pool = get_buffer_pool()
    n = 1024
    x = np.random.randn(n).astype(np.float32)
    x_buf = pool.host_to_device(x)
    out_buf = pool.allocate(n * 4)

    # Test: relu(sigmoid(x))
    jit.fuse_elementwise_chain(["sigmoid", "relu"], n, [x_buf.buffer], out_buf.buffer)
    result = pool.device_to_host(out_buf, np.float32, (n,))
    expected = np.maximum(0, 1.0 / (1.0 + np.exp(-x)))
    assert np.allclose(result, expected, atol=1e-3)

    # Test: exp(neg(x))
    jit.fuse_elementwise_chain(["neg", "exp"], n, [x_buf.buffer], out_buf.buffer)
    result = pool.device_to_host(out_buf, np.float32, (n,))
    expected = np.exp(-x)
    assert np.allclose(result, expected, atol=1e-3)

    # Test: abs(tanh(x))
    jit.fuse_elementwise_chain(["tanh", "abs"], n, [x_buf.buffer], out_buf.buffer)
    result = pool.device_to_host(out_buf, np.float32, (n,))
    expected = np.abs(np.tanh(x))
    assert np.allclose(result, expected, atol=1e-3)

    pool.free(x_buf)
    pool.free(out_buf)


def test_fused_binary_unary():
    jit = get_jit_compiler()
    pool = get_buffer_pool()
    n = 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    a_buf = pool.host_to_device(a)
    b_buf = pool.host_to_device(b)
    out_buf = pool.allocate(n * 4)

    # Test: relu(a + b)
    jit.fuse_binary_then_unary("add", ["relu"], n, a_buf.buffer, b_buf.buffer, out_buf.buffer)
    result = pool.device_to_host(out_buf, np.float32, (n,))
    expected = np.maximum(0, a + b)
    assert np.allclose(result, expected, atol=1e-3)

    # Test: sigmoid(a * b)
    jit.fuse_binary_then_unary("mul", ["sigmoid"], n, a_buf.buffer, b_buf.buffer, out_buf.buffer)
    result = pool.device_to_host(out_buf, np.float32, (n,))
    expected = 1.0 / (1.0 + np.exp(-(a * b)))
    assert np.allclose(result, expected, atol=1e-3)

    pool.free(a_buf)
    pool.free(b_buf)
    pool.free(out_buf)


def test_autotuner():
    tuner = get_auto_tuner()
    wg = tuner.optimal_workgroup_1d(100000)
    assert wg > 0 and (wg & (wg - 1)) == 0

    tile = tuner.optimal_tile_size(512, 512, 512)
    assert tile in (4, 8, 16)


def test_kernel_cache():
    jit = get_jit_compiler()
    pool = get_buffer_pool()
    cache = get_kernel_cache()
    n = 512
    x = np.random.randn(n).astype(np.float32)
    x_buf = pool.host_to_device(x)
    out_buf = pool.allocate(n * 4)

    initial_hits = cache.stats()["hits"]
    jit.fuse_elementwise_chain(["sigmoid", "relu"], n, [x_buf.buffer], out_buf.buffer)
    jit.fuse_elementwise_chain(["sigmoid", "relu"], n, [x_buf.buffer], out_buf.buffer)
    final_hits = cache.stats()["hits"]

    assert final_hits >= initial_hits + 1

    pool.free(x_buf)
    pool.free(out_buf)


if __name__ == "__main__":
    print("=" * 60)
    print("  TorchCL JIT Compiler Test")
    print("=" * 60)
    test_fused_unary_chains()
    print("  [PASS] Fused Unary Chains")
    test_fused_binary_unary()
    print("  [PASS] Fused Binary + Unary Chains")
    test_autotuner()
    print("  [PASS] Auto-Tuner")
    test_kernel_cache()
    print("  [PASS] Kernel Cache")
    print("\nALL JIT TESTS PASSED!")
