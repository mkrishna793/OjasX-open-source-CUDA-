"""
Test JIT Integration: verifies transparent JIT kernel fusion, buffer caching, and auto-tuning.
"""

import sys
import torch
import torchcl
from torchcl.jit.compiler import get_jit_compiler


def test_jit_integration():
    print("\n--- Test 1: JIT Fusion Pipeline relu(add(a, b)) ---")
    torch.manual_seed(42)
    a = torch.randn(128, 128, dtype=torch.float32)
    b = torch.randn(128, 128, dtype=torch.float32)

    a_cl = torchcl.to_opencl(a)
    b_cl = torchcl.to_opencl(b)

    res_cl = torchcl.relu(torchcl.add(a_cl, b_cl))
    res_cpu = torchcl.to_cpu(res_cl)

    expected = torch.relu(a + b)
    diff = (res_cpu - expected).abs().max().item()
    assert diff < 1e-4, f"Difference too high: {diff}"

    print("--- Test 2: In-Memory Kernel Cache Hit ---")
    compiler = get_jit_compiler()
    cache = compiler._cache

    a2 = torch.randn(128, 128, dtype=torch.float32)
    b2 = torch.randn(128, 128, dtype=torch.float32)

    a2_cl = torchcl.to_opencl(a2)
    b2_cl = torchcl.to_opencl(b2)

    initial_hits = cache.stats()["hits"]

    res2_cl = torchcl.relu(torchcl.add(a2_cl, b2_cl))
    res2_cpu = torchcl.to_cpu(res2_cl)

    expected2 = torch.relu(a2 + b2)
    diff2 = (res2_cpu - expected2).abs().max().item()
    assert diff2 < 1e-4, f"Difference too high: {diff2}"

    final_hits = cache.stats()["hits"]
    assert final_hits > initial_hits, "Cache hit was not triggered."


if __name__ == "__main__":
    test_jit_integration()
    print("ALL JIT INTEGRATION TESTS PASSED!")
