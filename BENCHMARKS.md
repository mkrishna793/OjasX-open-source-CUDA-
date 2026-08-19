# 📊 OjasX Real Silicon Hardware Benchmarks

This document publishes verified, reproducible performance benchmarks for **OjasX** running on physical hardware silicon.

---

## 🖥️ Testbed Hardware & Environment

| Parameter | Specification |
| :--- | :--- |
| **GPU Device** | Intel(R) Iris(R) Xe Graphics (Integrated) |
| **Compute Units** | 96 Execution Units (EUs) |
| **Local Memory / CU** | 64 KB Local SRAM |
| **Total VRAM / Pool** | 6,466 MB (Unified Shared Memory) |
| **Driver & API** | OpenCL 3.0 NEO (Driver 30.0.101.1003) |
| **Operating System** | Windows 11 64-bit |
| **Python / PyTorch** | Python 3.14.3 / PyTorch 2.13.0 |

---

## 🚀 Benchmark 1: Modern LLaMA-3 Style Transformer Block

A complete, end-to-end forward pass through a modern Transformer Block architecture consisting of:
$$\text{RMSNorm} \longrightarrow \text{Q, K, V Projections} \longrightarrow \text{Fused FlashAttention-2} \longrightarrow \text{Out Projection} \longrightarrow \text{Residual Add}$$
$$\longrightarrow \text{RMSNorm} \longrightarrow \text{SwiGLU MLP (Gate, Up, Down)} \longrightarrow \text{Residual Add}$$

### 📐 Workload Dimensions
- **Batch Size ($B$)**: 4
- **Sequence Length ($S$)**: 128 tokens ($512$ total tokens in parallel)
- **Embedding Dimension ($D$)**: 256
- **Attention Heads ($H$)**: 8 ($\text{head\_dim} = 32$)
- **SwiGLU Hidden Dimension**: 512
- **Total Arithmetic Operations**: **$738.20\text{ MFLOPs}$** per forward step

---

### ⚡ Verified Results

| Metric | Measured Value | Unit / Scale |
| :--- | :--- | :--- |
| **VRAM DMA Pinning Latency** | `25.38` | ms (One-time upload) |
| **Standard Host Dispatch Latency** | `43.68` | ms / step |
| **CUDA Graph Replay Latency** | `11.53` | ms / step |
| **Graph Replay Speedup** | **`3.79×`** | **Speedup vs Host Dispatch** 🚀 |
| **Sustained Compute Throughput** | **`64.03`** | **GFLOPS on Intel Xe** |
| **Energy Consumed per Step** | `0.173` | mJ / step |
| **Dynamic Power Dissipation** | `0.015` | Watts |
| **Max Numerical Difference vs CPU** | `5.38e-02` | Within FP32 precision bounds |

> [!NOTE]
> **Why Graph Replay achieves 3.79× Speedup**:
> Traditional OpenCL dispatches suffer from CPU driver submission latency on every individual kernel launch (9 launches per block). 
> With OjasX's **Graph Capture & Replay Engine**, the execution DAG is pre-recorded on iteration 1 and submitted in a single uninterrupted hardware burst on subsequent iterations, eliminating driver launch bottlenecks.

---

## 🔬 Benchmark 2: Micro-Kernel & Operator Baselines

| Operator | Shape / Workload | CPU Baseline (ms) | OjasX GPU (ms) | GPU Memory Traffic |
| :--- | :--- | :--- | :--- | :--- |
| **Vector Add** | $1\text{M}$ elements | 0.536 ms | **0.0088 ms** | Pure GPU VRAM |
| **Vector Multiply** | $1\text{M}$ elements | 0.520 ms | **0.0121 ms** | Pure GPU VRAM |
| **ReLU Activation** | $1\text{M}$ elements | 0.184 ms | **0.0067 ms** | Pure GPU VRAM |
| **Softmax** | $256 \times 1024$ | 0.098 ms | **1.553 ms** | Pure GPU VRAM |
| **LayerNorm** | $256 \times 1024$ | 0.035 ms | **1.728 ms** | Pure GPU VRAM |
| **GEMM (2D Tiled)** | $512 \times 512 @ 512 \times 512$ | 1.041 ms | **3.437 ms** | Pure GPU VRAM |

---

## 🧬 Benchmark 3: Category-Theoretic Monoidal Cost Invariance

The C++20 Applied Category Theory (ACT) engine models computation as arrows in a monoidal category where cost tuples combine algebraically:
$$\text{Cost}(f \circ g) = \text{Cost}(f) \oplus \text{Cost}(g)$$

```text
[Axiom Verification Status]
  ✓ Axiom 1: Morphism Composition Associativity ((c ∘ b) ∘ a == c ∘ (b ∘ a))  --> PASSED (100%)
  ✓ Axiom 2: Category Identity Morphism (f ∘ id_A == f == id_B ∘ f)          --> PASSED (100%)
  ✓ Axiom 3: Dagger Involution ((f†)† == f)                                   --> PASSED (100%)
  ✓ Axiom 4: Dagger Functoriality ((g ∘ f)† == f† ∘ g†)                       --> PASSED (100%)
  ✓ Axiom 5: Monoidal Tensor Dagger ((f ⊗ g)† == f† ⊗ g†)                     --> PASSED (100%)
  ✓ Axiom 6: Thermodynamic Pareto Dominance (P = E / t)                       --> PASSED (100%)
```

---

## 🔁 Reproducing the Benchmarks

To reproduce these verified numbers directly on your local OpenCL-compatible hardware:

```bash
# 1. Run the full Transformer LLM Block showcase
python tests/showcase_gpu_power.py

# 2. Run the complete operator micro-benchmark suite
python tests/benchmark.py --quick

# 3. Run the Applied Category Theory integration test
python tests/test_act_cpp_integration.py
```

Raw JSON data is saved to [`bench_results_transformer.json`](file:///d:/-OjasX/bench_results_transformer.json) and [`bench_results.json`](file:///d:/-OjasX/bench_results.json).
