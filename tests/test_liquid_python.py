"""
OjasX Liquid Compute & Monoidal Ring AllReduce Test.
"""

import pytest
import torch
import torchcl
from torchcl.liquid import get_liquid_engine
from torchcl.api import to_cpu


def test_liquid_telemetry_and_partition():
    liquid = get_liquid_engine()
    telemetry = liquid.get_telemetry()
    assert len(telemetry) >= 1
    for dev in telemetry:
        assert "id" in dev and "name" in dev and "vram_mb" in dev

    x = torch.randn(12, 8)
    gpu_chunks = liquid.fluid_partition(x)
    assert len(gpu_chunks) == len(telemetry)


def test_monoidal_ring_allreduce():
    liquid = get_liquid_engine()
    x = torch.randn(12, 8)
    gpu_chunks = liquid.fluid_partition(x)

    allreduced_gpu = liquid.ring_allreduce(gpu_chunks, op="SUM")
    allreduced_cpu = to_cpu(allreduced_gpu)

    expected = gpu_chunks[0]
    for c in gpu_chunks[1:]:
        expected = expected + c
    expected_cpu = to_cpu(expected)

    assert torch.allclose(allreduced_cpu, expected_cpu, atol=1e-4)


if __name__ == "__main__":
    print("=" * 60)
    print("  OjasX Liquid Compute & Monoidal Ring AllReduce Test")
    print("=" * 60)
    test_liquid_telemetry_and_partition()
    print("  [PASS] Liquid telemetry & fluid partitioning")
    test_monoidal_ring_allreduce()
    print("  [PASS] Monoidal Ring AllReduce")
    print("\nLIQUID COMPUTE & RING ALLREDUCE PASSED 100%!")
