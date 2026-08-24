"""
CostAwareDispatcher — The brain of OjasX v3.
Consults the MorphismCost model and KernelSynthesizer before every GPU operation
to pick the optimal kernel strategy, precision, and workgroup configuration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from torchcl.hal.interface import DeviceInfo, PrecisionPolicy


# ── MorphismCost Monoid (unified Python implementation) ──────────────

@dataclass
class MorphismCost:
    """Thermodynamic cost of a GPU morphism. Mirrors C++ cost_monoid.hpp exactly.
    
    The Monoid laws:
        Identity:    MorphismCost.identity() + c == c
        Composition: cost(g ∘ f) = cost(f) + cost(g)
    """
    memory_bytes: int = 0
    flops: int = 0
    energy_joules: float = 0.0
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    thermal_resistance: float = 0.25

    @staticmethod
    def identity() -> MorphismCost:
        return MorphismCost()

    def __add__(self, other: MorphismCost) -> MorphismCost:
        return MorphismCost(
            memory_bytes=self.memory_bytes + other.memory_bytes,
            flops=self.flops + other.flops,
            energy_joules=self.energy_joules + other.energy_joules,
            bandwidth_gbps=max(self.bandwidth_gbps, other.bandwidth_gbps),
            latency_us=self.latency_us + other.latency_us,
            thermal_resistance=max(self.thermal_resistance, other.thermal_resistance),
        )

    def power_watts(self) -> float:
        if self.latency_us <= 0.0:
            return 0.0
        return self.energy_joules / (self.latency_us * 1e-6)

    def delta_temperature_c(self) -> float:
        return self.power_watts() * self.thermal_resistance

    def score(self, w_latency: float = 1.0, w_energy: float = 0.5,
              w_mem: float = 0.001) -> float:
        return (self.latency_us * w_latency +
                self.energy_joules * w_energy +
                float(self.memory_bytes) * w_mem)

    def is_pareto_dominant_over(self, other: MorphismCost) -> bool:
        be = (self.energy_joules <= other.energy_joules and
              self.latency_us <= other.latency_us and
              self.memory_bytes <= other.memory_bytes)
        sb = (self.energy_joules < other.energy_joules or
              self.latency_us < other.latency_us or
              self.memory_bytes < other.memory_bytes)
        return be and sb

    @staticmethod
    def estimate(op: str, *dims: int, device: DeviceInfo | None = None) -> MorphismCost:
        """Estimate cost for an operation given its dimensions."""
        cu = device.compute_units if device else 96
        bw = device.max_bandwidth_gbps if device and device.max_bandwidth_gbps > 0 else 68.0
        tdp_watts = 15.0  # Conservative iGPU TDP

        if op in ("gemm", "matmul") and len(dims) >= 3:
            M, N, K = dims[0], dims[1], dims[2]
            flops = 2 * M * N * K
            mem = (M * K + K * N + M * N) * 4
            latency_us = (flops / (cu * 1e7))  # Rough model
            energy_j = (latency_us * 1e-6) * tdp_watts
        elif op in ("relu", "sigmoid", "gelu", "silu", "tanh", "softmax",
                     "add", "sub", "mul", "div", "exp", "log", "sqrt"):
            numel = dims[0] if dims else 1024
            flops = numel
            mem = numel * 8  # read + write
            latency_us = (numel / (bw * 1e9 / 4)) * 1e6
            energy_j = (latency_us * 1e-6) * tdp_watts * 0.5
        elif op == "attention":
            B, H, S, D = dims if len(dims) >= 4 else (1, 8, 512, 64)
            flops = 4 * B * H * S * S * D  # Q@K + softmax + V
            mem = B * H * S * D * 4 * 3
            latency_us = (flops / (cu * 1e7))
            energy_j = (latency_us * 1e-6) * tdp_watts
        else:
            numel = dims[0] if dims else 1024
            flops = numel
            mem = numel * 4
            latency_us = numel / 1e6
            energy_j = (latency_us * 1e-6) * tdp_watts * 0.3
            
        return MorphismCost(
            memory_bytes=mem, flops=flops, energy_joules=energy_j,
            bandwidth_gbps=bw, latency_us=latency_us,
            thermal_resistance=0.25,
        )


# ── Kernel Strategy Candidate ───────────────────────────────────────

@dataclass
class KernelCandidate:
    """A candidate kernel strategy for the dispatcher to evaluate."""
    name: str
    strategy: str               # "tiled", "winograd", "sparse", "synthesized", "naive"
    kernel_file: str            # .cl file or "synthesized"
    kernel_function: str        # OpenCL kernel function name
    workgroup_size: int = 256
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    is_synthesized: bool = False
    source_code: str = ""       # Only for synthesized kernels
    estimated_cost: MorphismCost = field(default_factory=MorphismCost)

    def __repr__(self) -> str:
        return f"Candidate({self.name}, strategy={self.strategy}, wg={self.workgroup_size})"


# ── CostAwareDispatcher ─────────────────────────────────────────────

class CostAwareDispatcher:
    """The brain: profiles data, evaluates candidates, picks the best kernel.
    
    Flow:
        1. Profile input tensors (shape, sparsity, dtype) — O(1)
        2. Generate candidate kernel strategies from KernelSynthesizer
        3. Estimate MorphismCost for each candidate
        4. Select Pareto-optimal candidate (lowest cost score)
        5. Execute via HAL and log timing for online learning
    """

    def __init__(self) -> None:
        from torchcl.dispatch.kernel_synthesizer import KernelSynthesizer
        from torchcl.dispatch.precision import PrecisionSelector

        self._synthesizer = KernelSynthesizer()
        self._precision = PrecisionSelector()
        self._device_info: DeviceInfo | None = None
        self._timing_history: list[dict] = []

    @property
    def device_info(self) -> DeviceInfo:
        if self._device_info is None:
            from torchcl.hal import get_hal
            self._device_info = get_hal().device_info
        return self._device_info

    def get_candidates(self, op: str, shape: tuple,
                       sparsity: float = 0.0) -> list[KernelCandidate]:
        """Generate candidate kernel strategies for an operation."""
        candidates = []

        if op in ("gemm", "matmul"):
            M = int(shape[0]) if len(shape) > 0 else 1
            N = int(shape[1]) if len(shape) > 1 else 1
            K = int(shape[2]) if len(shape) > 2 else N

            if M >= 32 and N >= 32 and K >= 16:
                candidates.append(KernelCandidate(
                    name="tiled_gemm", strategy="tiled",
                    kernel_file="matmul.cl",
                    kernel_function="matmul_reg_tiled_f32",
                    workgroup_size=256, tile_m=64, tile_n=64, tile_k=16,
                ))
            else:
                candidates.append(KernelCandidate(
                    name="naive_gemm", strategy="naive",
                    kernel_file="matmul.cl",
                    kernel_function="matmul_naive_f32",
                    workgroup_size=256,
                ))
            # Winograd / sparse are opt-in. Auto-picking them used to beat
            # tiled in the paper cost model while losing on the wire.

        elif op in ("relu", "sigmoid", "gelu", "silu", "tanh",
                     "leaky_relu"):
            candidates.append(KernelCandidate(
                name=f"{op}_standard", strategy="standard",
                kernel_file="activation.cl",
                kernel_function=f"{op}_f32",
                workgroup_size=min(256, max(1, shape[0] if shape else 256)),
            ))

        elif op in ("add", "sub", "mul", "div"):
            candidates.append(KernelCandidate(
                name=f"{op}_ewise", strategy="standard",
                kernel_file="elementwise.cl",
                kernel_function=f"{op}_f32",
                workgroup_size=256,
            ))

        elif op == "attention":
            candidates.append(KernelCandidate(
                name="flash_attention", strategy="tiled",
                kernel_file="flash_attention.cl",
                kernel_function="flash_attention_f32",
                workgroup_size=256,
            ))

        return candidates

    def select_best(self, op: str, candidates: list[KernelCandidate],
                    *dims: int) -> KernelCandidate:
        """Select the candidate with lowest MorphismCost score."""
        if not candidates:
            raise ValueError(f"No kernel candidates for op '{op}'")
        if len(candidates) == 1:
            return candidates[0]

        best = candidates[0]
        best_score = float("inf")

        for c in candidates:
            cost = MorphismCost.estimate(op, *dims, device=self.device_info)

            # Adjust cost based on strategy strengths
            if c.strategy == "winograd":
                cost.flops = int(cost.flops * 0.875)  # 7/8 multiplications
            elif c.strategy == "sparse":
                # Benefit scales with sparsity
                cost.flops = int(cost.flops * 0.5)
            elif c.strategy == "synthesized":
                cost.latency_us *= 0.9  # Vendor-tuned = ~10% faster
            elif c.strategy == "naive":
                cost.latency_us *= 2.0  # Penalty for naive approach

            c.estimated_cost = cost
            s = cost.score()
            if s < best_score:
                best_score = s
                best = c

        return best

    def log_timing(self, candidate: KernelCandidate, actual_us: float) -> None:
        """Record actual timing for online cost model learning."""
        self._timing_history.append({
            "name": candidate.name,
            "strategy": candidate.strategy,
            "estimated_us": candidate.estimated_cost.latency_us,
            "actual_us": actual_us,
            "ratio": actual_us / max(candidate.estimated_cost.latency_us, 1e-6),
        })

    def profile_sparsity(self, tensor: torch.Tensor, max_samples: int = 256) -> float:
        """O(1) sparsity estimate. Never download an OpenCL tensor to do it."""
        if tensor.numel() == 0:
            return 0.0
        from torchcl.api import is_opencl_tensor
        if is_opencl_tensor(tensor):
            return 0.0
        flat = tensor.detach().cpu().flatten()
        n = min(max_samples, flat.numel())
        sample = flat[:n]
        return float((sample == 0).float().mean().item())
