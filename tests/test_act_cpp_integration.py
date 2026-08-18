"""
OjasX Applied Category Theory (ACT) Python & C++ Integration Test Suite.
Verifies Cost Monoid algebra, Hardware Matrix Kernel Synthesizers, and Fluid Routing.
"""

import pytest
import math
import numpy as np
import torch
import torchcl
from torchcl.liquid.cost_model import CostModel, MorphismCost, DataProfile, HardwareProfile, KernelConfig
from torchcl.liquid.dispatch import DifferentialDispatcher


def test_act_cost_monoid():
    # 1. Cost Monoid Algebra
    cost_gemm = CostModel.morphism_cost("gemm", 1024, 1024, 1024)
    cost_relu = CostModel.morphism_cost("relu", 1024 * 1024)

    # Monoidal Composition: Cost(g ∘ f) = Cost(f) ⊕ Cost(g)
    composed_cost = cost_gemm + cost_relu

    assert composed_cost.flops == cost_gemm.flops + cost_relu.flops
    assert composed_cost.memory_bytes == cost_gemm.memory_bytes + cost_relu.memory_bytes
    assert composed_cost.latency_us > 0.0
    assert composed_cost.energy_joules > 0.0
    assert composed_cost.power_watts() > 0.0
    assert composed_cost.delta_temperature_c() > 0.0


def test_act_micro_kernel_synthesis():
    # Verify hardware-aware micro-kernel selection
    dispatcher = DifferentialDispatcher()
    data = DataProfile(shape=(512, 512), sparsity=0.0, mean=0.0, std=1.0, dtype=np.dtype(np.float32))
    candidates = dispatcher.get_candidate_configs("matmul", data)
    assert len(candidates) >= 1

    best_cfg = dispatcher.cost_model.select_best("matmul", data, dispatcher.hardware_profile, candidates)
    assert best_cfg.workgroup_size > 0
    assert best_cfg.strategy in ("tiled", "naive", "dpas", "wmma")


def test_act_liquid_telemetry_dispatch():
    liquid = torchcl.liquid.get_liquid_engine()
    telemetry = liquid.get_telemetry()
    assert len(telemetry) >= 1

    x = torch.randn(24, 16)
    chunks = liquid.fluid_partition(x)
    assert len(chunks) == len(telemetry)

    # Monoidal Ring AllReduce
    allreduced = liquid.ring_allreduce(chunks)
    assert allreduced.shape == chunks[0].shape


if __name__ == "__main__":
    print("=" * 65)
    print("  OjasX Applied Category Theory (ACT) Integration Verification  ")
    print("=" * 65)
    test_act_cost_monoid()
    print("  [PASS] Cost Monoid Algebra & Composition")
    test_act_micro_kernel_synthesis()
    print("  [PASS] Micro-Kernel Hardware Synthesis (DPAS/WMMA/SIMD)")
    test_act_liquid_telemetry_dispatch()
    print("  [PASS] Liquid Fluid Telemetry & Monoidal Ring AllReduce")
    print("\nALL ACT INTEGRATION TESTS PASSED (100%)!")
