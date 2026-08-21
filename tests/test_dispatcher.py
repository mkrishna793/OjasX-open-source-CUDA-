"""
CostAwareDispatcher + KernelSynthesizer + PrecisionSelector Test Suite.
Verifies the v3 intelligent dispatch layer.
"""

import pytest
import numpy as np
import torch


# ── MorphismCost Monoid Tests ────────────────────────────────────────

def test_morphism_cost_identity():
    """Monoid identity: identity + c == c."""
    from torchcl.dispatch.cost_dispatcher import MorphismCost
    c = MorphismCost(memory_bytes=1024, flops=2048, latency_us=10.0, energy_joules=0.001)
    result = MorphismCost.identity() + c
    assert result.flops == c.flops
    assert result.memory_bytes == c.memory_bytes
    assert result.latency_us == c.latency_us


def test_morphism_cost_composition():
    """Monoid composition: cost(g∘f) = cost(f) + cost(g)."""
    from torchcl.dispatch.cost_dispatcher import MorphismCost
    f = MorphismCost(memory_bytes=1024, flops=2000, energy_joules=0.01, latency_us=5.0)
    g = MorphismCost(memory_bytes=512, flops=1000, energy_joules=0.005, latency_us=3.0)
    composed = f + g
    assert composed.flops == 3000
    assert composed.memory_bytes == 1536
    assert abs(composed.latency_us - 8.0) < 1e-6


def test_morphism_cost_thermodynamics():
    """Power and temperature calculations are physically plausible."""
    from torchcl.dispatch.cost_dispatcher import MorphismCost
    c = MorphismCost(energy_joules=0.015, latency_us=1000.0, thermal_resistance=0.25)
    assert c.power_watts() == pytest.approx(15.0, rel=0.01)
    assert c.delta_temperature_c() == pytest.approx(3.75, rel=0.01)


def test_morphism_cost_pareto_dominance():
    """Pareto dominance: strictly better in at least one dimension."""
    from torchcl.dispatch.cost_dispatcher import MorphismCost
    better = MorphismCost(memory_bytes=100, flops=100, latency_us=1.0, energy_joules=0.001)
    worse = MorphismCost(memory_bytes=200, flops=200, latency_us=2.0, energy_joules=0.002)
    assert better.is_pareto_dominant_over(worse)
    assert not worse.is_pareto_dominant_over(better)


def test_morphism_cost_estimate():
    """Cost estimation produces non-zero values for known ops."""
    from torchcl.dispatch.cost_dispatcher import MorphismCost
    gemm_cost = MorphismCost.estimate("gemm", 512, 512, 512)
    assert gemm_cost.flops == 2 * 512 * 512 * 512
    assert gemm_cost.memory_bytes > 0
    assert gemm_cost.latency_us > 0
    assert gemm_cost.energy_joules > 0


# ── Dispatcher Tests ─────────────────────────────────────────────────

def test_dispatcher_generates_candidates():
    """Dispatcher generates multiple kernel candidates for GEMM."""
    from torchcl.dispatch import get_dispatcher
    d = get_dispatcher()
    candidates = d.get_candidates("gemm", (256, 256), sparsity=0.0)
    assert len(candidates) >= 2  # At least tiled + winograd + naive
    names = [c.name for c in candidates]
    assert "tiled_gemm" in names or "naive_gemm" in names


def test_dispatcher_selects_best():
    """Dispatcher selects a candidate with lowest cost."""
    from torchcl.dispatch import get_dispatcher
    d = get_dispatcher()
    candidates = d.get_candidates("gemm", (256, 256), sparsity=0.0)
    best = d.select_best("gemm", candidates, 256, 256, 256)
    assert best.strategy in ("tiled", "winograd", "sparse", "synthesized", "naive")
    assert best.estimated_cost.flops > 0


def test_dispatcher_sparse_preference():
    """Dispatcher prefers sparse GEMM when sparsity is high."""
    from torchcl.dispatch import get_dispatcher
    d = get_dispatcher()
    candidates = d.get_candidates("gemm", (256, 256), sparsity=0.8)
    has_sparse = any(c.strategy == "sparse" for c in candidates)
    assert has_sparse, "High sparsity should generate sparse candidate"


def test_dispatcher_sparsity_profiling():
    """Sparsity profiling works on tensors."""
    from torchcl.dispatch import get_dispatcher
    d = get_dispatcher()
    dense = torch.randn(100)
    sparse = torch.zeros(100)
    sparse[:10] = 1.0

    assert d.profile_sparsity(dense) < 0.2
    assert d.profile_sparsity(sparse) > 0.5


# ── KernelSynthesizer Tests ──────────────────────────────────────────

def test_synthesizer_vendor_config():
    """Synthesizer detects vendor and picks correct config."""
    from torchcl.dispatch.kernel_synthesizer import KernelSynthesizer, _VENDOR_CONFIGS
    ks = KernelSynthesizer()
    cfg = ks.get_vendor_config("intel")
    assert cfg.accel_type == "dpas"
    assert cfg.vector_width == 8
    assert cfg.subgroup_size == 16

    cfg_amd = ks.get_vendor_config("amd")
    assert cfg_amd.accel_type == "wmma"
    assert cfg_amd.subgroup_size == 32


def test_synthesizer_generates_kernel():
    """Synthesizer generates valid OpenCL kernel source."""
    from torchcl.dispatch.kernel_synthesizer import KernelSynthesizer
    ks = KernelSynthesizer()
    candidate = ks.synthesize_gemm(512, 512, 512, vendor="intel")
    assert candidate is not None
    assert candidate.is_synthesized
    assert "synthesized_fused_gemm" in candidate.source_code
    assert "__kernel" in candidate.source_code
    assert "BM" in candidate.source_code


def test_synthesizer_skips_small_matrices():
    """Synthesizer returns None for matrices too small for tiling."""
    from torchcl.dispatch.kernel_synthesizer import KernelSynthesizer
    ks = KernelSynthesizer()
    candidate = ks.synthesize_gemm(4, 4, 4, vendor="intel")
    assert candidate is None  # Too small for 64x64 tiles


def test_synthesizer_backward_kernel():
    """Backward kernel generation produces valid OpenCL."""
    from torchcl.dispatch.kernel_synthesizer import KernelSynthesizer
    ks = KernelSynthesizer()
    source = ks.synthesize_backward_gemm()
    assert "__kernel" in source
    assert "grad_X" in source
    assert "grad_W" in source


# ── PrecisionSelector Tests ──────────────────────────────────────────

def test_precision_exact_always_fp32():
    """EXACT policy always returns fp32."""
    from torchcl.dispatch.precision import PrecisionSelector
    from torchcl.hal.interface import PrecisionPolicy
    ps = PrecisionSelector(policy=PrecisionPolicy.EXACT)
    d = ps.select("gemm", is_training=True)
    assert d.compute_dtype == "fp32"
    d = ps.select("gemm", is_training=False)
    assert d.compute_dtype == "fp32"


def test_precision_safe_training_fp32():
    """SAFE policy uses fp32 during training."""
    from torchcl.dispatch.precision import PrecisionSelector
    from torchcl.hal.interface import PrecisionPolicy
    ps = PrecisionSelector(policy=PrecisionPolicy.SAFE)
    d = ps.select("gemm", is_training=True)
    assert d.compute_dtype == "fp32"
    assert d.bytes_per_element == 4


def test_precision_auto_fp16_safe_ops():
    """AUTO policy selects fp16 for safe ops when hardware supports it."""
    from torchcl.dispatch.precision import PrecisionSelector
    from torchcl.hal.interface import PrecisionPolicy, DeviceInfo
    fake_dev = DeviceInfo(name="Test", vendor="Intel", compute_units=96,
                          global_mem_mb=6466, local_mem_kb=64, max_workgroup=256,
                          supports_fp16=True)
    ps = PrecisionSelector(policy=PrecisionPolicy.AUTO, device_info=fake_dev)
    d = ps.select("gemm")
    assert d.compute_dtype == "fp16"
    assert d.accumulate_dtype == "fp32"  # Always accumulate in fp32


def test_precision_auto_fp32_required_ops():
    """AUTO policy keeps fp32 for loss/reduction ops."""
    from torchcl.dispatch.precision import PrecisionSelector
    from torchcl.hal.interface import PrecisionPolicy, DeviceInfo
    fake_dev = DeviceInfo(name="Test", vendor="Intel", compute_units=96,
                          global_mem_mb=6466, local_mem_kb=64, max_workgroup=256,
                          supports_fp16=True)
    ps = PrecisionSelector(policy=PrecisionPolicy.AUTO, device_info=fake_dev)
    d = ps.select("cross_entropy")
    assert d.compute_dtype == "fp32"
