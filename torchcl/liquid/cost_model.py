"""
Cost Model — Pure Python scikit-learn-free model for predicting kernel performance.
Learns weights online from micro-profiler timings.
"""

from __future__ import annotations
import numpy as np
import os
import json


class DataProfile:
    """Statistical summary of tensor data features."""
    def __init__(self, shape: tuple, sparsity: float, mean: float, std: float, dtype: np.dtype):
        self.shape = shape
        self.size = int(np.prod(shape))
        self.sparsity = sparsity
        self.mean = mean
        self.std = std
        self.dtype = dtype


class HardwareProfile:
    """Hardware device features."""
    def __init__(self, compute_units: int, local_mem_kb: int, max_workgroup: int,
                 global_mem_mb: int, preferred_vector_width: int):
        self.compute_units = compute_units
        self.local_mem_kb = local_mem_kb
        self.max_workgroup = max_workgroup
        self.global_mem_mb = global_mem_mb
        self.preferred_vector_width = preferred_vector_width


class KernelConfig:
    """Execution configuration for a kernel."""
    def __init__(self, workgroup_size: int, tile_size: int = 1, strategy: str = "default", precision: str = "float"):
        self.workgroup_size = workgroup_size
        self.tile_size = tile_size
        self.strategy = strategy
        self.precision = precision

    def to_tuple(self) -> tuple:
        return (self.workgroup_size, self.tile_size, self.strategy, self.precision)

    def __repr__(self) -> str:
        return f"KernelConfig(wg={self.workgroup_size}, tile={self.tile_size}, strat={self.strategy}, prec={self.precision})"


class MorphismCost:
    """Python reflection of C++ MorphismCost Monoid."""
    def __init__(
        self,
        memory_bytes: int = 0,
        flops: int = 0,
        energy_joules: float = 0.0,
        bandwidth_gbps: float = 0.0,
        latency_us: float = 0.0,
        thermal_resistance: float = 0.25,
    ) -> None:
        self.memory_bytes = memory_bytes
        self.flops = flops
        self.energy_joules = energy_joules
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_us = latency_us
        self.thermal_resistance = thermal_resistance

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

    def score(self, w_latency: float = 1.0, w_energy: float = 0.5, w_mem: float = 0.001) -> float:
        return (self.latency_us * w_latency) + (self.energy_joules * w_energy) + (self.memory_bytes * w_mem)

    def is_pareto_dominant_over(self, other: MorphismCost) -> bool:
        better_or_equal = (
            self.energy_joules <= other.energy_joules
            and self.latency_us <= other.latency_us
            and self.memory_bytes <= other.memory_bytes
        )
        strictly_better = (
            self.energy_joules < other.energy_joules
            or self.latency_us < other.latency_us
            or self.memory_bytes < other.memory_bytes
        )
        return better_or_equal and strictly_better


class CostModel:
    """Lightweight analytical cost model with online regression capabilities."""

    @staticmethod
    def morphism_cost(op: str, *dims: int) -> MorphismCost:
        """Estimate 5D morphism cost for an operation."""
        if op == "gemm" and len(dims) >= 3:
            M, N, K = dims[0], dims[1], dims[2]
            flops = 2 * M * N * K
            mem = (M * K + K * N + M * N) * 4
            latency_us = (flops / 1e9) * 20.0  # Approx 50 GFLOPS on iGPU
            energy_j = (latency_us * 1e-6) * 15.0  # 15W TDP
            return MorphismCost(memory_bytes=mem, flops=flops, energy_joules=energy_j, bandwidth_gbps=68.0, latency_us=latency_us)
        else:
            numel = dims[0] if dims else 1024
            flops = numel
            mem = numel * 8
            latency_us = (numel / 1e6) * 5.0
            energy_j = (latency_us * 1e-6) * 10.0
            return MorphismCost(memory_bytes=mem, flops=flops, energy_joules=energy_j, bandwidth_gbps=68.0, latency_us=latency_us)

    def __init__(self) -> None:
        # Weights for compute cost, memory cost, and launch overhead
        self.weights = np.array([1.0, 1.0, 0.05], dtype=np.float32)
        self.history: list[tuple[np.ndarray, float]] = []
        self._db_path = os.path.expanduser("~/.cache/torchcl/cost_profiles.json")
        self.load_history()

    def load_history(self):
        try:
            if os.path.exists(self._db_path):
                with open(self._db_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        feats = np.array(item["features"], dtype=np.float32)
                        ms = float(item["actual_ms"])
                        self.history.append((feats, ms))
                if self.history:
                    self.retrain()
        except Exception:
            pass

    def save_history(self):
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            data = [
                {"features": h[0].tolist(), "actual_ms": h[1]}
                for h in self.history
            ]
            with open(self._db_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def feature_vector(self, op: str, data: DataProfile, hw: HardwareProfile, config: KernelConfig) -> np.ndarray:
        """Create a feature vector [expected_compute, expected_memory, launch_overhead]."""
        size = data.size
        # Estimate ops (simplistic rules depending on the operation)
        if op == "matmul":
            # Matmul shape is typically M, N, K
            M = data.shape[0] if len(data.shape) > 0 else 1
            N = data.shape[1] if len(data.shape) > 1 else 1
            K = data.shape[1] if len(data.shape) > 1 else 1  # approximation
            ops = 2 * M * N * K
        else:
            ops = size

        # 1. Compute time estimate (arbitrary scale)
        # Higher workgroup size or vectorization reduces compute time
        wg_eff = min(1.0, hw.max_workgroup / config.workgroup_size)
        precision_scale = 0.5 if config.precision in ("half", "fp16") else 1.0
        expected_compute = (ops * precision_scale) / (hw.compute_units * config.workgroup_size * wg_eff + 1e-5)

        # 2. Memory time estimate (arbitrary scale)
        item_size = 2 if config.precision in ("half", "fp16") else 4
        # Tiled matmul uses local memory, saving global bandwidth
        mem_savings = 4.0 if config.strategy == "tiled" else 1.0
        expected_memory = (size * item_size) / (mem_savings * 1e6)

        # 3. Launch overhead constant
        launch_overhead = 1.0

        return np.array([expected_compute, expected_memory, launch_overhead], dtype=np.float32)

    def predict_cost(self, op: str, data: DataProfile, hw: HardwareProfile, config: KernelConfig) -> float:
        """Predict execution cost in milliseconds."""
        features = self.feature_vector(op, data, hw, config)
        return float(np.dot(features, self.weights))

    def select_best(self, op: str, data: DataProfile, hw: HardwareProfile,
                    candidates: list[KernelConfig]) -> KernelConfig:
        """Find the configuration with the lowest predicted cost."""
        best_cfg = candidates[0]
        best_cost = float("inf")
        for cfg in candidates:
            cost = self.predict_cost(op, data, hw, cfg)
            if cost < best_cost:
                best_cost = cost
                best_cfg = cfg
        return best_cfg

    def log_timing(self, op: str, data: DataProfile, hw: HardwareProfile, config: KernelConfig, actual_ms: float):
        """Record a profile point to self.history."""
        features = self.feature_vector(op, data, hw, config)
        self.history.append((features, actual_ms))
        # Trigger retraining online when history is sufficient
        if len(self.history) >= 10 and len(self.history) % 5 == 0:
            self.retrain()
            self.save_history()

    def retrain(self):
        """Fit weights using linear regression (least squares) with L2 regularization."""
        if not self.history:
            return
        X = np.stack([h[0] for h in self.history])
        y = np.array([h[1] for h in self.history])

        # Solve via normal equation with L2 penalty (Ridge)
        lambda_val = 1e-2
        XTX = np.dot(X.T, X) + lambda_val * np.eye(X.shape[1])
        XTy = np.dot(X.T, y)
        try:
            new_weights = np.linalg.solve(XTX, XTy)
            # Ensure weights remain non-negative
            self.weights = np.clip(new_weights, 1e-4, 100.0)
        except np.linalg.LinAlgError:
            pass  # Keep current weights if matrix is singular
