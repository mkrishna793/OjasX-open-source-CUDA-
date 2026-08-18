"""
OjasX Phase 0: CUDA / CPU Correctness Verification Baseline Suite.
Strict numerical accuracy verification across all hardware compute paths.
"""

import pytest
import math
import torch
import torchcl


def test_baseline_arithmetic():
    torch.manual_seed(101)
    a_cpu = torch.randn(10000)
    b_cpu = torch.randn(10000) + 2.0  # Avoid division by zero

    a_cl = torchcl.to_opencl(a_cpu)
    b_cl = torchcl.to_opencl(b_cpu)

    ops = [
        ("add", lambda x, y: x + y, lambda x, y: torchcl.add(x, y), 1e-5),
        ("sub", lambda x, y: x - y, lambda x, y: torchcl.sub(x, y), 1e-5),
        ("mul", lambda x, y: x * y, lambda x, y: torchcl.mul(x, y), 1e-5),
        ("div", lambda x, y: x / y, lambda x, y: torchcl.div(x, y), 1e-4),
    ]

    for name, cpu_fn, cl_fn, tol in ops:
        expected = cpu_fn(a_cpu, b_cpu)
        actual = torchcl.to_cpu(cl_fn(a_cl, b_cl))
        max_diff = (expected - actual).abs().max().item()
        assert max_diff < tol, f"Op {name} failed: max diff {max_diff} > {tol}"


def test_baseline_activations():
    torch.manual_seed(202)
    x_cpu = torch.randn(10000)
    x_cl = torchcl.to_opencl(x_cpu)

    ops = [
        ("relu", torch.relu, torchcl.relu, 1e-6),
        ("sigmoid", torch.sigmoid, torchcl.sigmoid, 1e-5),
        ("tanh", torch.tanh, torchcl.tanh_, 1e-5),
        ("gelu", lambda x: torch.nn.functional.gelu(x, approximate="tanh"), torchcl.gelu, 1e-5),
        ("silu", torch.nn.functional.silu, torchcl.silu, 1e-5),
    ]

    for name, cpu_fn, cl_fn, tol in ops:
        expected = cpu_fn(x_cpu)
        actual = torchcl.to_cpu(cl_fn(x_cl))
        max_diff = (expected - actual).abs().max().item()
        assert max_diff < tol, f"Activation {name} failed: max diff {max_diff} > {tol}"


def test_baseline_matrix_ops():
    torch.manual_seed(303)
    # MatMul
    A_cpu = torch.randn(64, 128)
    B_cpu = torch.randn(128, 64)
    A_cl = torchcl.to_opencl(A_cpu)
    B_cl = torchcl.to_opencl(B_cpu)

    expected_mm = torch.matmul(A_cpu, B_cpu)
    actual_mm = torchcl.to_cpu(torchcl.matmul(A_cl, B_cl))
    assert (expected_mm - actual_mm).abs().max().item() < 1e-4

    # Transpose
    expected_t = A_cpu.t()
    actual_t = torchcl.to_cpu(torchcl.transpose(A_cl))
    assert (expected_t - actual_t).abs().max().item() < 1e-6


def test_baseline_reductions():
    torch.manual_seed(404)
    x_cpu = torch.randn(50000)
    x_cl = torchcl.to_opencl(x_cpu)

    # Sum
    diff_sum = (torchcl.to_cpu(torchcl.sum_(x_cl)) - x_cpu.sum()).abs().item()
    assert diff_sum < 1e-3

    # Max
    diff_max = (torchcl.to_cpu(torchcl.max_(x_cl)) - x_cpu.max()).abs().item()
    assert diff_max < 1e-6

    # Min
    diff_min = (torchcl.to_cpu(torchcl.min_(x_cl)) - x_cpu.min()).abs().item()
    assert diff_min < 1e-6

    # Mean
    diff_mean = (torchcl.to_cpu(torchcl.mean(x_cl)) - x_cpu.mean()).abs().item()
    assert diff_mean < 1e-5


def test_baseline_norms_and_attention():
    torch.manual_seed(505)
    # LayerNorm
    x_cpu = torch.randn(16, 64)
    ln = torch.nn.LayerNorm(64)
    expected_ln = ln(x_cpu)
    x_cl = torchcl.to_opencl(x_cpu)
    w_cl = torchcl.to_opencl(ln.weight)
    b_cl = torchcl.to_opencl(ln.bias)
    actual_ln = torchcl.to_cpu(torchcl.layer_norm(x_cl, (64,), w_cl, b_cl))
    assert (expected_ln - actual_ln).abs().max().item() < 1e-4

    # Fused Attention
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    expected_attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    q_cl = torchcl.to_opencl(q)
    k_cl = torchcl.to_opencl(k)
    v_cl = torchcl.to_opencl(v)
    actual_attn = torchcl.to_cpu(torchcl.fused_attention(q_cl, k_cl, v_cl))
    assert (expected_attn - actual_attn).abs().max().item() < 1e-4


def test_baseline_streams_and_events():
    s1 = torchcl.Stream()
    s2 = torchcl.Stream()
    e1 = torchcl.Event()
    e2 = torchcl.Event()

    e1.record(s1)
    with torchcl.stream(s1):
        a = torchcl.to_opencl(torch.randn(256, 256))
        b = torchcl.matmul(a, a)

    with torchcl.stream(s2):
        s2.wait_stream(s1)
        c = torchcl.relu(b)

    e2.record(s2)
    s2.synchronize()
    elapsed = e1.elapsed_time(e2)
    assert elapsed >= 0.0


if __name__ == "__main__":
    print("=" * 65)
    print("  Running Phase 0: CUDA / CPU Correctness Verification Suite")
    print("=" * 65)
    test_baseline_arithmetic()
    print("  [PASS] Baseline Arithmetic")
    test_baseline_activations()
    print("  [PASS] Baseline Activations")
    test_baseline_matrix_ops()
    print("  [PASS] Baseline Matrix Ops (GEMM & Transpose)")
    test_baseline_reductions()
    print("  [PASS] Baseline Reductions (Sum, Max, Min, Mean)")
    test_baseline_norms_and_attention()
    print("  [PASS] Baseline Norms & Fused Attention")
    test_baseline_streams_and_events()
    print("  [PASS] Baseline Streams & Event Synchronization")
    print("\nALL PHASE 0 CORRECTNESS VERIFICATION TESTS PASSED!")
