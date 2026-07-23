"""
OjasX Benchmark Suite — Compares OjasX v0.2 (Liquid Engine) vs PyTorch CPU.

Measures end-to-end training step time, forward-only, backward-only, and
per-operation latency for a small MLP on MNIST-like synthetic data.

Run:
    python tests/benchmark.py
    python tests/benchmark.py --quick       # fewer iterations
    python tests/benchmark.py --full        # more iterations
    python tests/benchmark.py --save-json   # write results to bench_results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import torch

# Force UTF-8 stdout so unicode in prints doesn't break on Windows cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torchcl


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    name: str
    cpu_ms: float
    ocl_ms: float
    speedup: float           # ocl_ms / cpu_ms ; <1.0 means OjasX is faster
    iterations: int
    notes: str = ""


@dataclass
class BenchReport:
    device_name: str
    opencl_version: str
    platform: str
    python: str
    torch: str
    results: list[BenchResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def time_fn(fn: Callable, warmup: int = 3, iters: int = 20) -> float:
    """Time a callable in milliseconds. Returns median to reduce noise."""
    def _force_materialize(res):
        if hasattr(res, "_elem"):
            res = res._elem
        if hasattr(torchcl, "is_opencl_tensor") and torchcl.is_opencl_tensor(res):
            if hasattr(torchcl.api, "materialize_lazy_tensor") and torchcl.api._is_lazy(res):
                torchcl.api.materialize_lazy_tensor(res)

    for _ in range(warmup):
        res = fn()
        _force_materialize(res)
    torchcl.synchronize() if hasattr(torchcl, "synchronize") else None

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        res = fn()
        _force_materialize(res)
        if hasattr(torchcl, "synchronize"):
            torchcl.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))


def _alloc(shape, dtype=torch.float32):
    engine = torchcl.ops.engine.get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    handle = torchcl.api._wrap_output(cl_buf, shape, dtype)
    return handle, cl_buf


# ═══════════════════════════════════════════════════════════════════════
# Per-op micro-benchmarks
# ═══════════════════════════════════════════════════════════════════════

def bench_add(iters: int) -> BenchResult:
    n = 1024 * 1024
    a_cpu = torch.randn(n)
    b_cpu = torch.randn(n)
    a_cl = torchcl.to_opencl(a_cpu)
    b_cl = torchcl.to_opencl(b_cpu)

    def cpu():
        return a_cpu + b_cpu

    def ocl():
        return torchcl.add(a_cl, b_cl)

    cpu_ms = time_fn(cpu, iters=iters)
    ocl_ms = time_fn(ocl, iters=iters)
    return BenchResult("add (1M elem)", cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters,
                       "includes H2D + D2H for OCL if used standalone")


def bench_mul(iters: int) -> BenchResult:
    n = 1024 * 1024
    a_cpu = torch.randn(n)
    b_cpu = torch.randn(n)
    a_cl = torchcl.to_opencl(a_cpu)
    b_cl = torchcl.to_opencl(b_cpu)

    cpu_ms = time_fn(lambda: a_cpu * b_cpu, iters=iters)
    ocl_ms = time_fn(lambda: torchcl.mul(a_cl, b_cl), iters=iters)
    return BenchResult("mul (1M elem)", cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters)


def bench_relu(iters: int) -> BenchResult:
    n = 1024 * 1024
    x_cpu = torch.randn(n)
    x_cl = torchcl.to_opencl(x_cpu)

    cpu_ms = time_fn(lambda: torch.relu(x_cpu), iters=iters)
    ocl_ms = time_fn(lambda: torchcl.relu(x_cl), iters=iters)
    return BenchResult("relu (1M elem)", cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters)


def bench_matmul(iters: int) -> BenchResult:
    M = K = N = 512
    a_cpu = torch.randn(M, K)
    b_cpu = torch.randn(K, N)
    a_cl = torchcl.to_opencl(a_cpu)
    b_cl = torchcl.to_opencl(b_cpu)

    cpu_ms = time_fn(lambda: a_cpu @ b_cpu, iters=iters)
    ocl_ms = time_fn(lambda: torchcl.matmul(a_cl, b_cl), iters=iters)
    return BenchResult(f"matmul {M}x{K} @ {K}x{N}", cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters)


def bench_softmax(iters: int) -> BenchResult:
    n, c = 256, 1024
    x_cpu = torch.randn(n, c)
    x_cl = torchcl.to_opencl(x_cpu)

    cpu_ms = time_fn(lambda: torch.softmax(x_cpu, dim=-1), iters=iters)
    ocl_ms = time_fn(lambda: torchcl.softmax(x_cl, dim=-1), iters=iters)
    return BenchResult(f"softmax ({n}x{c})", cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters)


def bench_layer_norm(iters: int) -> BenchResult:
    n, d = 256, 1024
    x_cpu = torch.randn(n, d)
    w_cpu = torch.ones(d)
    b_cpu = torch.zeros(d)
    x_cl = torchcl.to_opencl(x_cpu)
    w_cl = torchcl.to_opencl(w_cpu)
    b_cl = torchcl.to_opencl(b_cpu)

    def cpu():
        return torch.nn.functional.layer_norm(x_cpu, [d], w_cpu, b_cpu)

    def ocl():
        return torchcl.layer_norm(x_cl, w_cl, b_cl, d)

    cpu_ms = time_fn(cpu, iters=iters)
    ocl_ms = time_fn(ocl, iters=iters)
    return BenchResult(f"layer_norm ({n}x{d})", cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters)


# ═══════════════════════════════════════════════════════════════════════
# End-to-end: 3-layer MLP forward (OjasX nn vs PyTorch nn)
# ═══════════════════════════════════════════════════════════════════════

def bench_mlp_forward(iters: int) -> BenchResult:
    """Forward-only of a 784→256→128→10 MLP, batch=32."""
    from torchcl.nn import OpenCLLinear, OpenCLReLU

    in_dim, h1, h2, out = 784, 256, 128, 10
    batch = 32

    # OjasX model
    lin1 = OpenCLLinear(in_dim, h1)
    lin2 = OpenCLLinear(h1, h2)
    lin3 = OpenCLLinear(h2, out)
    relu1 = OpenCLReLU()
    relu2 = OpenCLReLU()
    x_cl = torchcl.to_opencl(torch.randn(batch, in_dim))

    def ocl():
        h = relu1(lin1(x_cl))
        h = relu2(lin2(h))
        return lin3(h)

    # PyTorch CPU model (same weights copied)
    cpu_lin1 = torch.nn.Linear(in_dim, h1)
    cpu_lin2 = torch.nn.Linear(h1, h2)
    cpu_lin3 = torch.nn.Linear(h2, out)
    with torch.no_grad():
        cpu_lin1.weight.copy_(lin1.weight.data)
        cpu_lin1.bias.copy_(lin1.bias.data)
        cpu_lin2.weight.copy_(lin2.weight.data)
        cpu_lin2.bias.copy_(lin2.bias.data)
        cpu_lin3.weight.copy_(lin3.weight.data)
        cpu_lin3.bias.copy_(lin3.bias.data)
    x_cpu = torch.randn(batch, in_dim)

    def cpu():
        h = torch.relu(cpu_lin1(x_cpu))
        h = torch.relu(cpu_lin2(h))
        return cpu_lin3(h)

    cpu_ms = time_fn(cpu, iters=iters)
    ocl_ms = time_fn(ocl, iters=iters)
    return BenchResult(
        f"MLP fwd 784→256→128→10 (b={batch})",
        cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters,
        "raw GPU compute, no tensor movement amortization",
    )


# ═══════════════════════════════════════════════════════════════════════
# End-to-end: training step on MNIST-like data
# ═══════════════════════════════════════════════════════════════════════

def gen_synthetic_batch(batch_size: int, num_classes: int = 10):
    images = torch.zeros(batch_size, 784)
    labels = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        label = i % num_classes
        labels[i] = label
        img = torch.zeros(28, 28)
        r = (label * 2) % 20
        c = (label * 3) % 20
        img[r:r + 8, c:c + 8] = torch.randn(8, 8) * 0.5 + 1.0
        images[i] = img.flatten()
    return images, labels


def one_hot(labels, num_classes=10):
    out = torch.zeros(labels.shape[0], num_classes)
    for i in range(labels.shape[0]):
        out[i, labels[i]] = 1.0
    return out


def bench_training_step(iters: int) -> BenchResult:
    """One full train step: forward + loss + backward + SGD update.

    OCL path: forward on GPU, loss on GPU, backward + update on CPU
    (full GPU autograd isn't end-to-end in v0.2 — we measure what works).
    """
    from torchcl.nn import OpenCLLinear, OpenCLReLU

    in_dim, h1, h2, out = 784, 256, 128, 10
    batch = 64
    lr = 0.01

    # --- OjasX setup ---
    lin1 = OpenCLLinear(in_dim, h1)
    lin2 = OpenCLLinear(h1, h2)
    lin3 = OpenCLLinear(h2, out)
    relu1 = OpenCLReLU()
    relu2 = OpenCLReLU()

    # --- PyTorch CPU setup ---
    cpu_lin1 = torch.nn.Linear(in_dim, h1)
    cpu_lin2 = torch.nn.Linear(h1, h2)
    cpu_lin3 = torch.nn.Linear(h2, out)

    def ocl_step():
        images, labels = gen_synthetic_batch(batch)
        targets = one_hot(labels)

        x = torchcl.to_opencl(images)

        # Forward on GPU
        h1_out = relu1(lin1(x))
        h2_out = relu2(lin2(h1_out))
        logits = lin3(h2_out)
        probs = torchcl.softmax(logits, dim=-1)

        # Backward + update on CPU (V0.2 limitation)
        probs_cpu = torchcl.to_cpu(probs)
        targets_cpu = targets.numpy()
        h1_cpu = torchcl.to_cpu(h1_out)
        h2_cpu = torchcl.to_cpu(h2_out)

        # Linear layers store weights as [out, in], so W3 is [10, 128]
        # grad_W3 = h2.T @ (probs - targets) / batch  -> [128, 10]
        grad_out = (probs_cpu - targets_cpu) / batch
        grad_w3 = h2_cpu.T @ grad_out
        # grad_h2 = grad_out @ W3  (W3 is [10, 128], not transposed)  -> [64, 128]
        w3_cpu = torchcl.to_cpu(lin3._weight_gpu)
        grad_h2 = grad_out @ w3_cpu
        grad_h2 = grad_h2 * (h2_cpu > 0).float()
        # grad_W2 = h1.T @ grad_h2  -> [256, 128]
        grad_w2 = h1_cpu.T @ grad_h2
        # grad_h1 = grad_h2 @ W2  -> [64, 256]
        w2_cpu = torchcl.to_cpu(lin2._weight_gpu)
        grad_h1 = grad_h2 @ w2_cpu
        grad_h1 = grad_h1 * (h1_cpu > 0).float()
        # grad_W1 = images.T @ grad_h1  -> [784, 256]
        # (then we want it in the layout of weight, which is [out, in] = [256, 784]
        # so we transpose at the end)
        grad_w1 = images.T @ grad_h1

        with torch.no_grad():
            lin1.weight.data -= lr * grad_w1.T
            lin2.weight.data -= lr * grad_w2.T
            lin3.weight.data -= lr * grad_w3.T

    def cpu_step():
        images, labels = gen_synthetic_batch(batch)
        targets = one_hot(labels)

        h1_out = torch.relu(cpu_lin1(images))
        h2_out = torch.relu(cpu_lin2(h1_out))
        logits = cpu_lin3(h2_out)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum() / batch

        # Full PyTorch autograd on CPU
        for p in [cpu_lin1.weight, cpu_lin2.weight, cpu_lin3.weight,
                  cpu_lin1.bias, cpu_lin2.bias, cpu_lin3.bias]:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in [cpu_lin1.weight, cpu_lin2.weight, cpu_lin3.weight]:
                p.data -= lr * p.grad

    cpu_ms = time_fn(cpu_step, warmup=2, iters=max(3, iters // 3))
    ocl_ms = time_fn(ocl_step, warmup=2, iters=max(3, iters // 3))
    return BenchResult(
        f"Train step MLP 784→256→128→10 (b={batch})",
        cpu_ms, ocl_ms, ocl_ms / cpu_ms, max(3, iters // 3),
        "OCL fwd on GPU, bwd+update on CPU (V0.2); CPU uses full torch.autograd",
    )


# ═══════════════════════════════════════════════════════════════════════
# Liquid pillar benchmarks — unique to OjasX
# ═══════════════════════════════════════════════════════════════════════

def bench_ckt_evolve(iters: int) -> BenchResult:
    """CKT: evolve a state from 0 → 2.0 via ODE integration. No CPU equivalent."""
    ckt = torchcl.liquid.ckt_engine.get_ckt_engine(tau=1.0, eps=1e-3, method="rk2")
    n = 4096

    state = ckt.create_state((n,), initial_dt=0.1)
    input_data = torch.ones(n) * 2.0
    input_cl = torchcl.to_opencl(input_data)
    input_buf = torchcl.api._get_buf(input_cl)

    def step():
        ckt.evolve("relu", input_buf, state, max_steps=20, tol=1e-3)

    ocl_ms = time_fn(step, warmup=1, iters=iters)
    torchcl.liquid.state.get_state_manager().release(state._id)
    return BenchResult(
        f"CKT evolve 0→2.0 (n={n}, 20 steps)",
        cpu_ms=0.0, ocl_ms=ocl_ms, speedup=float("nan"), iterations=iters,
        notes="OjasX-exclusive feature (stateful ODE integration)",
    )


def bench_liquid_tensor_growth(iters: int) -> BenchResult:
    """LiquidTensor.append: grow without reallocation. vs torch.cat reallocation."""
    from torchcl.liquid.memory import LiquidTensor

    chunk_cpu = torch.randn(64, 8)
    chunk_cl = torchcl.to_opencl(chunk_cpu)
    chunk_buf = torchcl.api._get_buf(chunk_cl)

    def ocl():
        lt = LiquidTensor(max_shape=(1024, 8), dtype=np.float32)
        for _ in range(16):
            lt.append(chunk_buf, axis=0)
        lt.release()

    def cpu():
        out = torch.zeros(0, 8)
        for _ in range(16):
            out = torch.cat([out, chunk_cpu], dim=0)

    cpu_ms = time_fn(cpu, iters=iters)
    ocl_ms = time_fn(ocl, iters=iters)
    return BenchResult(
        "LiquidTensor.append (16x64x8) vs torch.cat",
        cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters,
        "OCL ring-buffer write; CPU reallocates 16 times",
    )


def bench_aps_packing(iters: int) -> BenchResult:
    """AdaptivePrecision: pack FP32 → FP16 on GPU. CPU equivalent is numpy."""
    ap = torchcl.liquid.precision.AdaptivePrecision()
    n = 1024 * 1024
    x_cpu = torch.randn(n).numpy().astype(np.float32)
    x_cl = torchcl.to_opencl(torch.from_numpy(x_cpu))
    x_buf = torchcl.api._get_buf(x_cl)

    def ocl():
        ap.pack_to_fp16(x_buf, n)

    def cpu():
        x_cpu.astype(np.float16)

    cpu_ms = time_fn(cpu, iters=iters)
    ocl_ms = time_fn(ocl, iters=iters)
    return BenchResult(
        f"FP32→FP16 packing (n={n})",
        cpu_ms, ocl_ms, ocl_ms / cpu_ms, iters,
        "OCL kernel vs numpy copy+cast",
    )


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

ALL_BENCHMARKS = [
    ("Per-op", bench_add),
    ("Per-op", bench_mul),
    ("Per-op", bench_relu),
    ("Per-op", bench_matmul),
    ("Per-op", bench_softmax),
    ("Per-op", bench_layer_norm),
    ("End-to-end", bench_mlp_forward),
    ("End-to-end", bench_training_step),
    ("Liquid", bench_ckt_evolve),
    ("Liquid", bench_liquid_tensor_growth),
    ("Liquid", bench_aps_packing),
]


def run_all(preset: str) -> BenchReport:
    iters = {"quick": 5, "normal": 20, "full": 50}[preset]

    info = torchcl.get_device_info()
    report = BenchReport(
        device_name=info["name"],
        opencl_version=info["version"],
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch=torch.__version__,
    )

    print("=" * 72)
    print("  OjasX v0.2.0 — Benchmark Suite")
    print("=" * 72)
    print(f"  Device:     {report.device_name}")
    print(f"  OpenCL:     {report.opencl_version}")
    print(f"  Python:     {report.python}")
    print(f"  PyTorch:    {report.torch}")
    print(f"  Iterations: {iters} (median)")
    print("=" * 72)

    for category, fn in ALL_BENCHMARKS:
        try:
            r = fn(iters)
            report.results.append(r)
            print(f"\n[{category}] {r.name}")
            if r.cpu_ms > 0:
                verdict = "[OCL faster]" if r.speedup < 1.0 else "[CPU faster]"
                print(f"  CPU: {r.cpu_ms:7.2f} ms    OCL: {r.ocl_ms:7.2f} ms    "
                      f"speedup: {r.speedup:5.2f}x   {verdict}")
            else:
                print(f"  OCL: {r.ocl_ms:7.2f} ms    (no CPU equivalent)")
            if r.notes:
                print(f"  Note: {r.notes}")
        except Exception as e:
            print(f"\n[{category}] {fn.__name__}: SKIPPED — {type(e).__name__}: {e}")

    # Summary
    comparable = [r for r in report.results if r.cpu_ms > 0 and not np.isnan(r.speedup)]
    if comparable:
        wins = sum(1 for r in comparable if r.speedup < 1.0)
        losses = sum(1 for r in comparable if r.speedup >= 1.0)
        report.summary = {
            "total_benchmarks": len(report.results),
            "comparable_to_cpu": len(comparable),
            "ocx_wins": wins,
            "cpu_wins": losses,
            "median_speedup": float(np.median([r.speedup for r in comparable])),
            "best_speedup": min(r.speedup for r in comparable),
            "worst_speedup": max(r.speedup for r in comparable),
        }

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    if report.summary:
        s = report.summary
        print(f"  Total benchmarks:  {s['total_benchmarks']}")
        print(f"  Comparable to CPU: {s['comparable_to_cpu']}")
        print(f"  OjasX wins:        {s['ocx_wins']}")
        print(f"  PyTorch wins:      {s['cpu_wins']}")
        print(f"  Median speedup:    {s['median_speedup']:.2f}x  "
              f"(<1.0 = OjasX faster)")
        print(f"  Best case:         {s['best_speedup']:.2f}x")
        print(f"  Worst case:        {s['worst_speedup']:.2f}x")
    print("=" * 72)

    return report


def main():
    parser = argparse.ArgumentParser(description="OjasX v0.2 benchmark suite")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_const", const="quick", dest="preset",
                       default="normal")
    group.add_argument("--normal", action="store_const", const="normal", dest="preset")
    group.add_argument("--full", action="store_const", const="full", dest="preset")
    parser.add_argument("--save-json", metavar="PATH", help="write results to JSON")
    args = parser.parse_args()

    report = run_all(args.preset)

    if args.save_json:
        payload = asdict(report)
        for r in payload["results"]:
            if np.isnan(r["speedup"]):
                r["speedup"] = None
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {args.save_json}")

    return 0 if report.summary.get("ocx_wins", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
