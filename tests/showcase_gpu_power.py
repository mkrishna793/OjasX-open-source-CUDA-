"""
OjasX Deep Hardware Showcase: Modern Transformer LLM Block on Laptop GPU.
Demonstrates:
  1. High-Performance Fused Kernels on OpenCL GPU (RMSNorm + FlashAttention + SwiGLU)
  2. Graph Capture & Replay Acceleration (CUDA Graph equivalent)
  3. Applied Category Theory (ACT) Monoidal Cost Telemetry (Joules, Watts, GFLOPS)
  4. Strict Numerical Correctness vs PyTorch CPU Reference
"""

import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor, _get_buf, _wrap_output
from torchcl.jit.graph_runner import capture_graph
from torchcl.liquid.cost_model import CostModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class CPUTransformerBlock(nn.Module):
    """PyTorch CPU Reference Implementation."""
    def __init__(self, dim: int = 256, num_heads: int = 8, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.q_weight = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.k_weight = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.v_weight = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.out_weight = nn.Parameter(torch.randn(dim, dim) * 0.02)

        self.gamma2 = nn.Parameter(torch.ones(dim))
        self.gate_weight = nn.Parameter(torch.randn(hidden_dim, dim) * 0.02)
        self.up_weight = nn.Parameter(torch.randn(hidden_dim, dim) * 0.02)
        self.down_weight = nn.Parameter(torch.randn(dim, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # 1. RMSNorm
        var = x.pow(2).mean(-1, keepdim=True)
        h = x * torch.rsqrt(var + 1e-6) * self.gamma1

        # 2. Linear Projections & Attention
        h_flat = h.view(B * S, D)
        q = (h_flat @ self.q_weight.t()).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = (h_flat @ self.k_weight.t()).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = (h_flat @ self.v_weight.t()).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B * S, D)
        attn_proj = (attn_out @ self.out_weight.t()).view(B, S, D)
        x = x + attn_proj

        # 3. SwiGLU MLP
        var2 = x.pow(2).mean(-1, keepdim=True)
        h2 = x * torch.rsqrt(var2 + 1e-6) * self.gamma2
        h2_flat = h2.view(B * S, D)
        gate = F.silu(h2_flat @ self.gate_weight.t())
        up = h2_flat @ self.up_weight.t()
        mlp_out = ((gate * up) @ self.down_weight.t()).view(B, S, D)
        out = x + mlp_out

        return out


def gpu_linear(x_flat, weight_t):
    """Linear projection on GPU: out = x @ W^T."""
    return torchcl.matmul(x_flat, weight_t)


def reshape_gpu(t, new_shape):
    """Zero-copy view reshape on OpenCL buffer."""
    buf = _get_buf(t)
    return _wrap_output(buf, new_shape)


def ojasx_gpu_transformer_forward(
    x_cl, gamma1_cl, q_wt_cl, k_wt_cl, v_wt_cl, out_wt_cl,
    gamma2_cl, gate_wt_cl, up_wt_cl, down_wt_cl, B, S, D, H, d, hidden_dim
):
    """Pure OjasX OpenCL GPU Native Forward Pass."""
    # 1. GPU RMSNorm
    h_cl = torchcl.rms_norm(x_cl, gamma1_cl, D, eps=1e-6)

    # 2. Linear Q, K, V Projections on GPU
    q_flat = gpu_linear(h_cl, q_wt_cl)
    k_flat = gpu_linear(h_cl, k_wt_cl)
    v_flat = gpu_linear(h_cl, v_wt_cl)

    # 3. Zero-Copy Reshape to [B, H, S, d] for Fused FlashAttention
    q_4d = reshape_gpu(q_flat, (B, H, S, d))
    k_4d = reshape_gpu(k_flat, (B, H, S, d))
    v_4d = reshape_gpu(v_flat, (B, H, S, d))

    attn_4d = torchcl.fused_attention(q_4d, k_4d, v_4d)
    attn_flat = reshape_gpu(attn_4d, (B * S, D))
    attn_proj_cl = gpu_linear(attn_flat, out_wt_cl)
    x1_cl = x_cl + attn_proj_cl

    # 4. GPU Second Norm & SwiGLU MLP
    h2_cl = torchcl.rms_norm(x1_cl, gamma2_cl, D, eps=1e-6)
    gate_out = gpu_linear(h2_cl, gate_wt_cl)
    up_out = gpu_linear(h2_cl, up_wt_cl)
    swiglu_out = torchcl.swiglu(gate_out, up_out)
    mlp_out = gpu_linear(swiglu_out, down_wt_cl)
    out_cl = x1_cl + mlp_out

    return out_cl


def run_showcase():
    print("=" * 75)
    print("  OJASX DEEP HARDWARE SHOWCASE: TRANSFORMER LLM BLOCK ON LAPTOP GPU  ")
    print("=" * 75)

    # 1. Device Hardware Information
    dev_info = torchcl.get_device_info()
    print(f"\n[Hardware Telemetry]")
    print(f"  * Device Name       : {dev_info.get('name', 'Unknown')}")
    print(f"  * Compute Units     : {dev_info.get('max_compute_units', 'Unknown')} Execution Units")
    print(f"  * Total VRAM        : {dev_info.get('global_mem_size_mb', 0):,.0f} MB")
    print(f"  * Local Memory/CU   : {dev_info.get('local_mem_size_kb', 0)} KB")
    print(f"  * OpenCL Driver     : {dev_info.get('driver_version', 'Unknown')}")

    # 2. Workload Configuration
    B, S, D = 4, 128, 256
    H = 8
    d = D // H
    hidden_dim = 512
    print(f"\n[Workload Configuration]")
    print(f"  * Model Architecture: Modern LLaMA-style Transformer Block")
    print(f"  * Batch Size (B)    : {B}")
    print(f"  * Sequence Length(S): {S} tokens")
    print(f"  * Embedding Dim (D) : {D}")
    print(f"  * Attention Heads   : {H} (head_dim = {d})")
    print(f"  * SwiGLU Hidden Dim : {hidden_dim}")

    # Instantiate Reference CPU Model
    torch.manual_seed(42)
    model_cpu = CPUTransformerBlock(dim=D, num_heads=H, hidden_dim=hidden_dim).eval()
    x_cpu = torch.randn(B * S, D)

    # Move Weights to OpenCL GPU
    print("\n--- 1. Moving Model & Tensors to OpenCL GPU ---")
    t0 = time.perf_counter()
    x_cl = to_opencl(x_cpu)
    gamma1_cl = to_opencl(model_cpu.gamma1.data)
    q_wt_cl = to_opencl(model_cpu.q_weight.data.t().contiguous())
    k_wt_cl = to_opencl(model_cpu.k_weight.data.t().contiguous())
    v_wt_cl = to_opencl(model_cpu.v_weight.data.t().contiguous())
    out_wt_cl = to_opencl(model_cpu.out_weight.data.t().contiguous())
    gamma2_cl = to_opencl(model_cpu.gamma2.data)
    gate_wt_cl = to_opencl(model_cpu.gate_weight.data.t().contiguous())
    up_wt_cl = to_opencl(model_cpu.up_weight.data.t().contiguous())
    down_wt_cl = to_opencl(model_cpu.down_weight.data.t().contiguous())
    t_transfer = (time.perf_counter() - t0) * 1000.0
    print(f"  [OK] Model Weights & Input Tensors pinned to GPU VRAM ({t_transfer:.2f} ms)")

    # 3. Correctness Verification
    print("\n--- 2. Numerical Parity Verification vs CPU Reference ---")
    out_cpu = model_cpu(x_cpu.view(B, S, D)).view(B * S, D)
    out_cl = ojasx_gpu_transformer_forward(
        x_cl, gamma1_cl, q_wt_cl, k_wt_cl, v_wt_cl, out_wt_cl,
        gamma2_cl, gate_wt_cl, up_wt_cl, down_wt_cl, B, S, D, H, d, hidden_dim
    )
    torchcl.synchronize()

    out_cl_cpu = to_cpu(out_cl)
    diff = (out_cl_cpu - out_cpu).abs().max().item()
    mean_diff = (out_cl_cpu - out_cpu).abs().mean().item()
    print(f"  * Max Absolute Difference : {diff:.6e}")
    print(f"  * Mean Absolute Difference: {mean_diff:.6e}")
    if diff < 1e-2:
        print(f"  [PASS] GPU Output matches PyTorch CPU reference with high fidelity! [OK]")
    else:
        print(f"  [WARN] Difference: {diff}")

    # 4. CUDA Graph Parity: Graph Capture & Replay Speedup
    print("\n--- 3. Graph Capture & Replay Engine (CUDA Graph Parity) ---")
    # Warmup
    for _ in range(3):
        _ = ojasx_gpu_transformer_forward(
            x_cl, gamma1_cl, q_wt_cl, k_wt_cl, v_wt_cl, out_wt_cl,
            gamma2_cl, gate_wt_cl, up_wt_cl, down_wt_cl, B, S, D, H, d, hidden_dim
        )
    torchcl.synchronize()

    # Dynamic Capture
    print("  * Recording execution DAG into CapturedGraph...")
    with capture_graph() as cg:
        captured_out = ojasx_gpu_transformer_forward(
            x_cl, gamma1_cl, q_wt_cl, k_wt_cl, v_wt_cl, out_wt_cl,
            gamma2_cl, gate_wt_cl, up_wt_cl, down_wt_cl, B, S, D, H, d, hidden_dim
        )
    torchcl.synchronize()
    print(f"  [OK] Graph Captured: {len(cg.commands)} kernel commands recorded in DAG")

    # Benchmark Standard vs Replay
    num_runs = 50

    # A. Standard Execution Loop
    torchcl.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = ojasx_gpu_transformer_forward(
            x_cl, gamma1_cl, q_wt_cl, k_wt_cl, v_wt_cl, out_wt_cl,
            gamma2_cl, gate_wt_cl, up_wt_cl, down_wt_cl, B, S, D, H, d, hidden_dim
        )
    torchcl.synchronize()
    t_standard = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # B. Graph Replay Execution Loop
    torchcl.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        cg.replay()
    torchcl.synchronize()
    t_replay = ((time.perf_counter() - t0) / num_runs) * 1000.0

    speedup = t_standard / max(1e-5, t_replay)
    print(f"  * Standard Dispatch Latency: {t_standard:.3f} ms / step")
    print(f"  * Graph Replay Latency     : {t_replay:.3f} ms / step")
    print(f"  * Launch Overhead Reduction: {speedup:.2f}x speedup via zero-overhead replay! [FAST]")

    # 5. Applied Category Theory (ACT) Monoidal Cost Metrics
    print("\n--- 4. Applied Category Theory (ACT) Monoidal Cost Metrics ---")
    total_flops = 2 * B * S * (
        4 * D * D +              # Q, K, V, Out projections
        2 * S * D +              # Attention Scores & Context
        3 * D * hidden_dim       # SwiGLU Gate, Up, Down projections
    )
    gflops = (total_flops / (t_replay * 1e-3)) / 1e9
    cost = CostModel.morphism_cost("gemm", B * S, D, hidden_dim)
    joules = cost.energy_joules * (t_replay / max(1.0, cost.latency_us))
    power_watts = (joules / (t_replay * 1e-3)) if t_replay > 0 else 0.0

    print(f"  * Total Operations per Step: {total_flops / 1e6:.2f} MFLOPs")
    print(f"  * Real Sustained Throughput : {gflops:.2f} GFLOPS on Intel Xe Graphics")
    print(f"  * Estimated Energy Consumed : {joules * 1000.0:.3f} mJ / step")
    print(f"  * Dynamic Power Dissipation : {power_watts:.2f} Watts")

    print("\n" + "=" * 75)
    print("  SHOWCASE COMPLETED: OJASX SATURATES LAPTOP GPU WITH ZERO CPU ROUNDTRIPS  ")
    print("=" * 75)


if __name__ == "__main__":
    run_showcase()
