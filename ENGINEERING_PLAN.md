# OjasX Engineering Plan: Breaking CUDA with Real Superiority

## Executive Vision

**Goal**: Make OjasX the first compute stack that is simultaneously:
- **Truly portable** — runs on AMD, Intel, Qualcomm, ARM, NVIDIA GPUs, and FPGAs
- **Energy-proportional** — 2-10× less power than equivalent CUDA workloads  
- **Hardware-light** — runs on integrated GPUs, laptops, edge devices (no $40K H100 needed)
- **Sovereign/Offline** — AGPL guarantees no vendor lock-in, works air-gapped
- **Production-ready** — matches CUDA feature-completeness + novel capabilities

**The "Impossible Trinity" we're solving**:
1. Portability = CUDA's weakness (vendor lock-in)
2. Efficiency = CUDA's weakness (400W TDP for AI that runs fine on 25W)
3. Production readiness = CUDA's strength (but we match it + add novel features)

---

## I. FUNDAMENTAL ARCHITECTURAL REWRITE

### 1.1 Unified Memory Fabric (Eliminate the CPU Bottleneck)

**Current Problem**: Every operation forces:
```
PyTorch Tensor → .cpu().numpy() → cl.enqueue_copy → CLBuffer → Kernel → 
cl.enqueue_copy → numpy → torch.from_numpy
```
This is 2-4× memcpy per op, dominating latency on discrete GPUs (PCIe bandwidth ~16 GB/s vs HBM2 ~800+ GB/s).

**Solution**: Zero-copy tensor ↔ CLBuffer interop with explicit memory spaces.

**Architecture**:
```
                    +--------------------------+
                    |     PyTorch Storage      |
                    +--------------------------+
                               |
                   CL_MEM_USE_HOST_PTR  (zero-copy on SVM)
                               |
                    +--------------------------+
                    |     CLBuffer (host-shared)|
                    +--------------------------+
                               |
          +--------------------+--------------------+
          |                                     |
   CL_GLOBAL_MEM (discrete GPU)          CL_LOCAL_MEM (integrated GPU APU)
          |                                     |
          +-------------------------------------+
                               |
                   OpenCL Kernel executes directly
          on memory mapped into its address space
```

**Implementation Plan**:

1. **`torchcl/runtime/memory.py`** — Major rewrite:
   - `CLBuffer.from_pytorch_storage(tensor, share_type="zero_copy")` 
     - Creates `clCreateBufferWithProperties` with `CL_MEM_USE_HOST_PTR`
     - Maps host pointer into device address space via `clEnqueueMapBuffer`
     - Tracks `cl_context` lifetime with explicit `refcount`
   - `CLBuffer.to_pytorch_tensor(buffer, dtype, shape)` 
     - Zero-copy view via `torch.from_numpy` on shared memory
     - No memcpy when buffer already device-resident
   - `engine.tensor_to_buffer_zero_copy(tensor)` 
     - Detects if tensor already has shared CLBuffer; wraps or creates new
     - Handles both discrete GPU (mmap) and integrated GPU (SVM) cases

2. **`torchcl/ops/engine.py`** — Rewrite data movement:
   - Remove all `.cpu().numpy()` roundtrips
   - `engine.run_kernel(kernel, buffers)` — accepts CLBuffer directly
   - `engine.tensor_op(tensor, op)` — automatic zero-copy wrapping
   - Per-stream command queues with `CL_QUEUE_OUT_OF_ORDER_ENABLE_KHR` for overlap

3. **`torchcl/tensor.py`** — Native device="opencl" tensor creation:
   - `__init__` auto-creates zero-copy CLBuffer
   - `torch.zeros(..., device="opencl")` works without monkeypatching
   - `tensor.to("opencl")` returns zero-copy wrapper, not CPU→GPU copy

**Impact**: 10-100× speedup for small-op workloads, enables GPU-only pipelines.

### 1.2 Graph-Native Execution (Replace Per-Op Launches)

**Current Problem**: Every operation = `clEnqueueNDRangeKernel` launch (~5-15 μs overhead). 
For a transformer with 100+ ops/step: 500-1500 μs launch overhead alone = 10-30% of step time.

**Solution**: Adopt a minimal IR + persistent kernel execution model.

**Architecture**:
```
                     +--------------------------+
                     |   PyTorch FX Graph / FXA |
                     +--------------------------+
                                |
                                v
                     +--------------------------+
                     |   OjasX Graph IR (minimal)|
                     +--------------------------+
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
   +--------------------+          +--------------------+  +--------------------+
   |   Persistent Kernel|          |   Grid-Stride      |  |   Work-stealing    |
   |   Thread Blocks    |          |   Reduction        |  |   (AWM generalization)|
   +--------------------+          +--------------------+  +--------------------+
          |                     |                     |
          v                     v                     v
   +--------------------+      +------------------+  +-------------------+
   |   Compiled OpenCL  |      |   Device-Side    |  |   Kernel Cache    |
   |   Kernel (SPIR-V) |      |   Work Queue     |  |   (LRU by signature)|
   +--------------------+      +------------------+  +-------------------+
                                |
                                v
                     +--------------------------+
                     |   Command Queue per Stream|
                     +--------------------------+
                                |
                                v
                     +--------------------------+
                     |   OpenCL Device          |
                     +--------------------------+
```

**Implementation Plan**:

1. **New file `torchcl/jit/ir.py`** — Minimal Graph IR:
   - `Node` class with `op_type`, `inputs`, `outputs`, `attrs`
   - `Graph` class with `topo_sort()`, `critical_path()`, `estimated_runtime()`
   - Operations: `MatMul`, `Add`, `Relu`, `LayerNorm`, `FlashAttention`, etc.
   - Lowering: `ir_graph → SPIR-V` or optimized OpenCL C string (with proper IR, NOT template substitution)

2. **Rewrite `torchcl/jit/compiler.py`** — IR-based compilation:
   - Replace template-substitution approach with proper code generation
   - Support multi-output fusion: `fuse("matmul", "add", "relu")` → 3 outputs in 1 kernel
   - Generate SPIR-V via `llvm-spirv` or build custom minimal IR → OpenCL C
   - Add `compile(graph, device_profiling)` → cached compiled artifact

3. **New file `torchcl/jit/persistence.py`** — Persistent Kernel Runtime:
   - `PersistentKernelBlock` — device-side thread block that persists across launches
   - Grid-stride pattern: `gid = get_global_id(0); if (gid < n) { loop() }`
   - Work-stealing queue on device: enqueue sub-tasks dynamically
   - Lifecycle: `acquire()` / `release()` / `refcount` management
   - Use AWM (Adaptive Workgroup Morphing) concepts from `liquid/awm.py` generalized

4. **Rewrite `torchcl/ops/engine.py`** — Graph execution:
   - `engine.execute_graph(graph, inputs)` — single queue launch
   - Auto-insert zero-copy buffer management
   - `engine.graph_capture(func)` — capture a Python function into Graph IR
   - `engine.graph_replay(graph, new_inputs)` — execute with different data

5. **`torchcl/_backend.py`** — `torch.compile` integration:
   - FX region fusion: fuse adjacent regions into single Graph IR
   - Lower to SPIR-V or optimized OpenCL C
   - Handle dynamic shapes via runtime shape inference
   - Cache compiled graphs by hash(repr(graph))

**Impact**: 5-15× speedup for pipeline-heavy workloads (transformers, RNNs, attention); eliminates launch overhead entirely.

### 1.3 Persistent Kernel Runtime (Amortize Overhead + Enable State)

**Current Problem**: Stateless kernel launches. No persistent state between operations. Cannot do "warm GPU" optimization.

**Solution**: Device-side persistent state + adaptive work distribution.

**Key Innovations**:

1. **`torchcl/liquid/ckt_engine.py`** — Generalize Continuous Kernel Time:
   - `LiquidState` — persistent GPU memory region that survives launches
   - `StatefulKernel` — base class for kernels that maintain state (RNN hidden state, optimizer moments, etc.)
   - `adaptive_ode_step()` — RK2 with error control on device, reduces compute by skipping unchanged state

2. **`torchcl/liquid/awm.py`** — Generalize Adaptive Workgroup Morphing:
   - Not just "convergent ops" — any kernel can declare workgroup size flexibility
   - Runtime queries `CL_DEVICE_MAX_WORK_GROUP_SIZE`, auto-tunes
   - Work-stealing: idle workgroups steal from congested ones (requires `cl_khr DynamicScratchPad`)

3. **`torchcl/ops/engine.py`** — Persistent kernel support:
   - `engine.acquire_persistent_block(op_type, shape)` → returns token
   - `engine.release_persistent_block(token)` 
   - `engine.stateful_kernel(token, inputs, state)` — kernel that reads/writes persistent state
   - Enables: RNNs with O(1) per-step (hidden state never leaves GPU), optimizer state on GPU

**Impact**: Enables truly efficient RNNs, stateful computation, optimizer state entirely on GPU.

---

## II. INTELLIGENCE LAYER: Auto-Placement & Data-Dependent Specialization

### 2.1 Auto-Device Placement Optimizer

**Current Problem**: Single-device only. No understanding of where each tensor should live for optimal efficiency.

**Solution**: Cost-model-driven auto-placement across heterogeneous devices.

**Architecture**:
```
                     +--------------------------+
                     |   Tensor DAG (from Graph IR)|
                     +--------------------------+
                                |
                                v
                     +--------------------------+
                     |   Hardware Profiles      |
         GPU_A (discrete):     |  - Bandwidth: 800 GB/s   |
         GPU_B (integrated):  |  - Power: 25 W          |
         NPU:                 |  - INT8 TOPS: 50          |
         CPU:                 |  - AVX-512 FLOPS: 1 TFLOP|
                     +--------------------------+
                                |
                                v
                     +--------------------------+
                     |   Cost Model:            |
         minimize:             |  - data_movement_cost    |
         α*compute + β*comm + γ*power  |  - kernel_launch_cost    |
                                |  - precision_penalty     |
                     +--------------------------+
                                |
                                v
                     +--------------------------+
                     |   Placement Decision:   |
         - Tensor A → GPU_A      |  (matmul goes to discrete)|
         - Tensor B → NPU        |  (activations → efficient silicon)|
         - Tensor C → CPU        |  (control flow, small ops)|
                     +--------------------------+
```

**Implementation Plan**:

1. **New file `torchcl/dispatch.py`** — Differential Dispatcher (generalize from `liquid/dispatch.py`):
   - `DataProfile` — sparsity, dynamic range, size, dtype, gradient-flow
   - `HardwareProfile` — device type, bandwidth, FLOPS, power, precision support
   - `CostModel` — online learning from micro-benchmarks, predicts runtime for each op+device combo
   - `place_tensor(tensor, available_devices)` → optimal device placement
   - `select_kernel(op, tensor_profile, target_device)` → best kernel config

2. **`torchcl/liquid/dispatch.py`** — Enhancements:
   - Add `cl_khr_subgroups` query for SIMD width
   - Add `cl_khr_fp16` / `cl_khr_bf16` detection
   - Add memory bandwidth estimation from device info
   - Online cost model: observe actual timings, update predictions

3. **`torchcl/ops/engine.py`** — Placement-aware execution:
   - `engine.tensor_to(tensor, target_device)` — moves tensor with optimal strategy
   - `engine.auto_placement(graph, devices)` — placements entire graph
   - `engine.run_with_placement(graph, devices)` — execute with optimal device mapping

4. **`torchcl/runtime/context.py`** — Enhanced device discovery:
   - Query all OpenCL platforms/devices on system
   - Benchmark bandwidth/flops for each (quick calibration)
   - Cache `HardwareProfile` for cost model initialization

**Impact**: Single model automatically splits across available hardware — e.g., transformer: matmuls on discrete GPU, embeddings on NPU, activations on integrated GPU/laptop, control flow on CPU. Enables "AI on any hardware" without user manual placement.

### 2.2 Data-Dependent Kernel Specialization

**Current Problem**: Single kernel configuration for all data. No adaptation to actual tensor values.

**Solution**: Runtime data analysis → kernel parameter selection.

**Architecture**:
```
                     +--------------------------+
                     |   Tensor Profile         |
         sparsity: 0.3%       |  - mean, std             |
         range: [-2.1, 5.7]    |  - dynamic_range         |
         dtype: bfloat16        |  - outlier_fraction      |
         shape: [1,512,512]     |  - dominant_freq (FFT)   |
                     +--------------------------+
                                |
                                v
                     +--------------------------+
                     |   Specialization Options |
         - FP32 kernel          |  (full precision)        |
         - FP16 kernel          |  (2× bandwidth, ~1% err) |
         - INT8 quantized       |  (8× bandwidth, calibrated)|
         - BFloat16 kernel      |  (native on newer devices)|
         - Mixed-precision     |  - accumulate in FP32,   |
                              |    multiply in FP16        |
                     +--------------------------+
                                |
                                v
                     +--------------------------+
                     |   Select Best           |
         Cost model predicts   |  based on profile +       |
         runtime for each     |    hardware profile       |
         option               +--------------------------+
```

**Implementation Plan**:

1. **`torchcl/liquid/precision.py`** — Adaptive Precision Streaming (generalize):
   - `TensorProfile.analyze()` — computes sparsity, range, outlier fraction in O(1) via 256-element sample
   - `APS.suggest_precision(profile, hardware)` → `{"compute": "fp16", "accumulate": "fp32", "accumulate_overflow": "fp64"}`
   - On-the-fly pack/unpack kernels: `pack_fp32_to_fp16`, `unpack_fp16_to_fp32`
   - Dynamic range tracking across iterations

2. **`torchcl/kernels/precision.cl`** — Enhanced precision ops:
   - `fp32_fp16_pack` / `fp16_fp32_unpack` with saturation handling
   - `bf16_fp32_pack` / `fp32_bf16_unpack`
   - Quantization/Dequantization with learned scales (or symmetric min/max)

3. **`torchcl/ops/engine.py`** — Precision-aware dispatch:
   - `engine.matmul(a, b, precision_config)` → selects optimal precision path
   - `engine.elementwise(x, op, precision)` → fp16-aware ReLU, etc.
   - Automatic downcast when dynamic range permits

4. **`torchcl/liquid/cost_model.py`** — Online cost learning:
   - Observe actual kernel timing per precision config
   - Update predictive model: `runtime = f(precision, shape, sparsity, hardware)`
   - Continually refine suggestions

**Impact**: Automatically uses INT8/FP16 where numerically safe, FP32 where needed. 2-4× speedup on capable hardware + improved accuracy stability.

### 2.3 Sparse-Aware Computation

**Current Problem**: No native sparse support. All ops dense even when 90% zeros.

**Solution**: Sparse kernel paths triggered by profile.

**Implementation**:

1. **Profile detects sparsity > 10%** → routes to sparse kernel
2. **Sparse matmul** — compressed row format (CSR) on OpenCL
3. **Sparse reduction** — masked work-items, early exit when value==0
4. **Sparse attention** — only compute QKV for non-zero entries

**New kernels in `torchcl/kernels/`**:
- `sparse_matmul.cl` — CSR/COO matmul with sparse/dense hybrid
- `sparse_reduction.cl` — sparse sum/max with bitmask
- `sparse_attention.cl` — attention with sparsity pattern

---

## III. PRODUCTION HARDENING LAYER

### 3.1 Complete ATen Dispatch Table

**Current Problem**: ~30 ops mapped; ~200+ ATen ops missing. `torch.compile` falls back to CPU per-node.

**Solution**: Map all critical ops; build FX region fusion.

**Implementation Plan**:

1. **`torchcl/tensor.py`** — Complete dispatch table:
   Map all ATen operations. Priority order (most critical first):
   ```
   Priority 1 (essential for training):
   - conv2d, conv1d, conv_transpose
   - batch_norm, instance_norm, layer_norm (backward)
   - embedding, embedding_bag
   - gelu, gelu_backward
   - dropout (with RNG state)
   - addmm, baddbmm
   - mm, addmm
   - transpose, contiguous
   - squeeze, unsqueeze
   - split, chunk
   - stack, cat
   - view, reshape
   - permute
   - index_select, gather, scatter
   - where, setwhere
   - bmm, matmul (already have)
   
   Priority 2 (essential for inference):
   - layer_norm (forward)
   - rms_norm (forward + backward)
   - gelu, silu, tanh, sigmoid (already have)
   - clamp, relu, relu6, leaky_relu
   - mul, add, sub, div
   - neg, abs
   - sqrt, exp, log, sin, cos
   - clamp, hardtanh
   - alphasort, sort (partial)
   - topk (top values, not indices)
   - index_add, scatter_add
   - nll_loss, cross_entropy_loss (already have)
   
   Priority 3 (nice-to-have):
   - einsum
   - layer_norm (with affine)
   - rsqrt, rcp
   - erf, erfc
   - pow
   - lerp, blend
   - interpolation (bilinear, nearest)
   - grid_sample
   - batch_dropout
   ```

2. **`torchcl/autograd.py`** — Add missing backward passes:
   - `Conv2dBackward` — already partially there, needs full correctness
   - `EmbeddingBagBackward`
   - `TopKBackward`
   - `SortBackward`
   - `TopKGradIndex` (gradient w.r.t. indices)

3. **`torchcl/_backend.py`** — `torch.compile` region fusion:
   - FX-based region grouping: identify consecutive ops that can share buffers
   - Lower entire regions to Graph IR + compile to single SPIR-V module
   - Handle dynamic shapes via runtime shape parameters
   - Cache by (region_hash, device_profile)

**Impact**: `torch.compile` works end-to-end; users can `model = torch.compile(model.to("opencl"))` and get real speedup.

### 3.2 Multi-GPU / Distributed Backend

**Current Problem**: Single-device only. No `torch.distributed` support.

**Solution**: NCCL-equivalent via OpenCL P2P + MPI.

**Implementation Plan**:

1. **New file `torchcl/distributed.py`** — TorchCl distributed backend:
   - `init_process(rank, world_size, backend="opencl")` 
   - `torch.distributed.DistributedDataParallel(mesh="opencl")` 
   - `torch.distributed.PipelineParallel(layers_per_rank)` 
   - `torch.distributed.FSDP(mesh="opencl", optimize=True)`

2. **Communication Primitives**:
   - `torchcl.comm.allreduce(tensor, op="sum")` — ring allreduce on OpenCL
   - `torchcl.comm.allgather(tensor)` — gather from all ranks
   - `torchcl.comm.broadcast(tensor, src=0)` — broadcast from root
   - `torchcl.comm.scatter(tensor, indices)` — scatter to ranks

3. **Backend Implementation**:
   - **P2P enabled**: `cl_khr_p2p` — direct GPU-to-GPU DMA (if same platform)
   - **MPI fallthrough**: `mpi4py` + `clEnqueueMigrateMemObjects` for cross-platform
   - **Naive CPU collectives** (fallback): gather→CPU→distribute (slow but works)

4. **`torchcl/liquid/dispatch.py`** — Distributed cost model:
   - Add network bandwidth to HardwareProfile
   - Cost model includes comm cost: `α*compute + β*comm + γ*sync`
   - Auto-optimize partition sizes for pipeline parallelism

5. **New file `torchcl/liquid/state_manager.py`** — Distributed state:
   - Optimizer state sharding (ZeRO stages 1-3)
   - Gradient accumulation across ranks
   - Checkpoint serialization with device placement

**Impact**: OjasX supports real distributed training — first open-source framework to bring DDP/FSDP to heterogeneous OpenCL devices.

### 3.3 Debugging & Profiling Ecosystem

**Current Problem**: No visibility. "It runs slow, IDK why."

**Solution**: Full stack profiling + debugging.

**Implementation Plan**:

1. **`torchcl/profiler.py`** — Profiling stack:
   - `profile_scope(name)` — context manager, records kernel time, buffer transfers, CPU compute
   - `profile.step()` — emits JSON with: `{"kernels": [...], "transfers": [...], "cpu_time": ..., "vram_peak": ...}`
   - `profile.compare(before, after)` — text diff of two profiles
   - Integration with `torch.autograd.profiler`

2. **Kernel introspection**:
   - `kernel.get_workgroup_size()` → reported
   - `kernel.get_local_memory_usage()` → reported
   - `kernel.get_register_usage()` → reported
   - `engine.list_kernels()` → show all compiled kernels + signatures

3. **`torchcl/debugger.py`** — Debugging tools:
   - `engine.memory_snapshot(tensor)` → VRAM→host dump (limited, for debugging)
   - `kernel.emulate_host()` → runs kernel on CPU with same params (for correctness verification)
   - `tensor.assert_close_cuda(ref_tolerance=1e-2)` — compare vs PyTorch CPU (reference)

4. **`torchcl/runtime/memory.py`** — Enhanced telemetry:
   - Real-time VRAM pressure polling via `clGetDeviceInfo`
   - OOM prediction: "will OOM in N more ops at current rate"
   - Memory leak detector with call-stack tracking
   - Buffer lifetime graph (who's holding reference to what)

5. **Integration with `torch.utils.collectives`**:
   - `torch.autograd.graph_soup` compatibility
   - `torch.profiler.emit_nvtabulate`-style output (adapted for OpenCL)

**Impact**: Developers can actually optimize and debug. No more "mystery slowdown."

### 3.4 AOT Compilation Pipeline

**Current Problem**: Everything JIT-compiled at runtime. Cold start, no offline deployment.

**Solution**: AOT compile OpenCL kernels before deployment.

**Implementation Plan**:

1. **New file `torchcl/aot_compiler.py`** — AOT compilation:
   - `aot_compile(graph, target_device, output_dir)` — compile ahead of time
   - Generates: `.cl` source + `spirv` + `json(profile)` + `json(kernel_config)`
   - Output: directory deployable to offline target

2. **Workflow**:
   ```bash
   # Train/optimize in Python
   import torchcl
   model = torch.compile(torchcl.Model().to("opencl"))
   
   # AOT compile for deployment
   torchcl.aot_compile(
       graph=model.graph,
       target_device="intel_irisxe",
       output_dir="build/intel_irisxe/"
   )
   
   # Deploy: just load compiled kernels, no Python needed for compilation
   python -c "import torchcl; torchcl.aot_run('build/intel_irisxe/', inputs)"
   ```

3. **Cache integration**: 
   - AOT output goes into `torchcl/jit/cache.py` LRU
   - Recompile only when graph signature changes OR hardware profile changes
   - `torchcl.jit.load_compiled(device)` → returns ready-to-execute module

**Impact**: Zero-cold-start deployment. Edge devices without Python still run optimized kernels.

---

## IV. DIFFERENTIATION: THE "REAL SUPERIORITY" FEATURES

### 4.1 Energy-Proportional AI (Power Budget API)

**The Problem**: CUDA has no power control. Models assume 400W H100. On laptop battery (15W), either crashes or throttles.

**The OjasX Innovation**: Explicit power budgeting that the system respects.

**Implementation**:

1. **New file `torchcl/power.py`** — Power budget layer:
   ```python
   torchcl.set_power_budget(watts=15)    # Laptop on battery
   torchcl.set_power_budget(watts=25)    # Laptop plugged in, efficient mode
   torchcl.set_power_budget(watts=400)   # Datacenter H100 mode
   torchcl.get_current_power_budget()
   ```

2. **How it works internally**:
   - `power_budget → adjusts precision config → APS activates`
   - `watts=15`: FP16 → INT8 where safe, batch size reduced 4×, gradient accumulation 4×
   - `watts=25`: FP16 primary, FP32 accumulate, some ops FP32-critical
   - `watts=400`: FP32 primary, FP16 only for non-critical paths
   - Queries `clGetDeviceInfo` for max clock, adjusts `clSetDeviceClock` if supported
   - Falls back gracefully: if device can't underclock, just restricts precision

3. **Energy-proportional metrics** (exposed):
   ```python
   torchcl.energy.total_joules_consumed()
   torchcl.energy.energy_per_op(operation="matmul", shape=[512,512])
   torchcl.energy.carbon_footprint(region="us_east")  # approximate
   ```

**Impact**: Same model runs 10× more energy-efficiently on efficient hardware. The differentiator CUDA cannot match.

### 4.2 Sovereign/Offline AI (AGPL Guarantee)

**The Problem**: NVIDIA cloud APIs, telemetry, remote-only features, export controls.

**The OjasX Innovation**: AGPL-licensed full stack. Runs completely offline. No API keys. No telemetry.

**Implementation**: (Already AGPL, but make it explicit)

1. **`LICENSE`** — Explicit AGPL-3.0 with patent clause
2. **`torchcl/telemetry.py`** — Zero telemetry by default:
   - `torchcl.telemetry.enabled` = `False` (hard default)
   - If user opts-in: anonymized aggregation only, no device IDs, no model info
   - `torchcl.set_telemetry(opts)` — explicit opt-in for feedback-driven improvements

3. **Air-gapped deployment guarantee**:
   ```python
   # Verify no network access needed
   torchcl.verify_offline_capable(model)
   # Returns: True + list of required files
   ```

4. **Sovereign AI workflow**:
   ```python
   # Download OjasX once, never need network again
   import torchcl
   model = torchcl.aot_compile(...)
   # Deploy to air-gapped system:
   scp build/ user@airgapped-server:
   ssh user@airgapped-server "python run_deploy.py"
   # Works with zero network access
   ```

**Impact**: Geopolitical differentiator. Countries/enterprises under export controls can build AI sovereignty.

### 4.3 Heterogeneous Scheduling (The "Real" Multi-Device)

**The Problem**: CUDA = one vendor, one hardware type. "Run on any GPU" means "run on any NVIDIA GPU."

**The OjasX Innovation**: True heterogeneous scheduling — model automatically splits across CPU + integrated GPU + discrete GPU + NPU, optimal placement per tensor/op.

**Implementation** (building on sections 2.1+):

1. **`torchcl.auto.device_map(model, available_devices)`** → full tensor DAG placement

2. **Example: Laptop with Intel Iris Xe + AMD Radeon**:
   ```python
   import torchcl
   
   # Auto-placement figures this out:
   devices = torchcl.get_all_devices()
   # [Intel Iris Xe (integrated), AMD Radeon (discrete)]
   
   model = torchcl.auto.device_map(my_transformer, devices)
   
   # Result placement:
   # - Token embeddings → Intel Iris Xe (efficient, low power)
   # - All matmuls (QKV, proj) → AMD Radeon (high bandwidth)
   # - Activations/gelu → Intel Iris Xe (already there, zero-copy)
   # - Control flow/slicing → CPU (AVX-512, minimal overhead)
   
   output = model(input)
   # Automatically executes across both devices with async overlaps
   ```

3. **Example: Edge device (Qualcomm Snapdragon NPU + ARM CPU)**:
   ```python
   devices = torchcl.get_all_devices()
   # [Qualcomm Adreno GPU, Hexagon NPU, ARM CPU]
   
   # Model auto-places:
   # - INT8 inference → Hexagon NPU (3-8W, best for INT8)
   # - FP16 matmuls → Adreno GPU
   # - Control flow → ARM CPU
   ```

**Impact**: "Write once, run optimally on anything." The genuine "write once, run anywhere" for AI compute.

### 4.4 Native ONNX / TorchScript / Keras Export

**The Problem**: Framework lock-in. "I trained in PyTorch, but I need to deploy ONNX."

**The OjasX Innovation**: OjasX as a compilation target + runtime, not just PyTorch wrapper.

**Implementation**:

1. **Export to OjasX IR**:
   ```python
   # Export trained model
   traced = torch.jit.trace(model, example_input)
   torchcl.export_ojasx(traced, "model.ojax")
   
   # Deploy anywhere OjasX runtime exists
   # No PyTorch needed at inference time
   ```

2. **ONNX → OjasX conversion**:
   ```python
   import onnx
   import torchcl
   
   onnx_model = onnx.load("model.onnx")
   ojasx_model = torchcl.from_onnx(onnx_model)
   torchcl.save_ojasx(ojax_model, "model.ojax")
   ```

3. **TensorFlow/Keras → OjasX**:
   ```python
   import tensorflow as tf
   import torchcl
   
   keras_model = tf.keras.models.load_model("model.h5")
   torchcl.from_keras(keras_model, "model.ojax")
   ```

**Impact**: OjasX becomes neutral compilation target for all frameworks. Framework vendors adopt OjasX as "run everywhere" runtime.

### 4.5 Continuous Compute (The "Liquid" Advantage)

**The Problem**: Stateless kernels. RNNs have O(steps) overhead because hidden state must be read/written from GPU memory each step. No concept of "compute that remembers."

**The OjasX Innovation**: `LiquidContinuous` — kernel state persists across launches, adaptive step-size ODE integration.

**Implementation** (building on `torchcl/liquid/ckt_engine.py`):

1. **`torchcl.liquid.LiquidModule`** — persistent state module:
   ```python
   class LiquidRNN(LiquidModule):
       def __init__(self, input_size, hidden_size):
           super().__init__()
           self.hidden_state = None  # Persistent on device
           
       def forward(self, x):
           # Hidden state never leaves GPU
           self.hidden_state = liquid_ode_step(
               self.hidden_state, x, 
               step_size=adaptive_rk2(self.hidden_state, x)
           )
           return self.hidden_state
   ```

2. **Adaptive ODE integration on device**:
   - RK2 with error estimator: if change < threshold, accept larger step
   - If change > threshold, reject and retry with smaller step
   - Reduces compute for "smooth" dynamics, increases where needed

3. **Use cases**:
   - RNNs/LSTMs/GRUs with 2-3× speedup (fewer kernel launches)
   - ODE solving for neural ODEs
   - State-space models (S4, etc.)
   - Reinforcement learning actor-critic with persistent critic

**Impact**: Truly novel compute paradigm. No other framework has persistent GPU-stateful kernels that persist across Python-kernel boundaries.

---

## V. PHASED ROADMAP

### Phase 0 (Current → 2 weeks): Foundation Fixes
- [ ] Zero-copy tensor↔buffer (memory.py rewrite)
- [ ] Remove all `.cpu().numpy()` roundtrips in engine.py
- [ ] Complete ATen dispatch table (tensor.py Priority 1)
- [ ] Basic multi-queue support (context.py + engine.py)
- [ ] Profile vs CUDA correctness on smoke tests

### Phase 1 (Weeks 2-6): Graph IR + Persistent Kernels
- [ ] New `torchcl/jit/ir.py` — Minimal Graph IR
- [ ] Rewrite `torchcl/jit/compiler.py` — IR-based compilation (SPIR-V or optimized OpenCL C)
- [ ] New `torchcl/jit/persistence.py` — Persistent kernel blocks
- [ ] Graph execution in `engine.py` — `execute_graph()`, `graph_capture()`, `graph_replay()`
- [ ] `torch.compile` region fusion in `_backend.py`

### Phase 2 (Weeks 6-12): Auto-Placement + Precision
- [ ] `torchcl/dispatch.py` — Differential dispatcher (generalized from liquid)
- [ ] `torchcl/liquid/precision.py` — Adaptive Precision Streaming
- [ ] Auto-device-mapping (`torchcl.auto.device_map`)
- [ ] Sparse kernel paths triggered by profile
- [ ] Online cost model integration

### Phase 3 (Weeks 12-20): Distributed + Production
- [ ] `torchcl/distributed.py` — DDP/FSDP/PipelineParallel on OpenCL
- [ ] Communication primitives (allreduce, allgather, broadcast)
- [ ] `torchcl/profiler.py` — Full stack profiling
- [ ] `torchcl/debugger.py` — Memory snapshots, kernel emulation
- [ ] Numerical verification vs CUDA on shared hardware

### Phase 4 (Weeks 20-32): Differentiation Features
- [ ] `torchcl/power.py` — Energy-proportional power budget API
- [ ] AGPO (AGPL) sovereignty guarantees formalized
- [ ] `torchcl.auto.heterogeneous_map` — True multi-device scheduling
- [ ] AOT compilation pipeline (`torchcl/aot_compiler.py`)
- [ ] ONNX/exports pipeline
- [ ] Liquid continuous compute (`torchcl.liquid.LiquidModule`)

### Phase 5 (Weeks 32-40+): Production Hardening & V1.0
- [ ] Stress tests, memory leak detection, 100% gradient correctness
- [ ] Multi-device CI (AMD, Intel, Qualcomm, ARM, NVIDIA)
- [ ] Benchmark suite vs CUDA on equivalent hardware
- [ ] Documentation, tutorials, examples
- [ ] V1.0 release: "Production-ready, truly portable AI compute stack"

---

## VI. KEY METRICS FOR SUCCESS

| Metric | Target | How Measured |
|--------|--------|--------------|
| **Portability** | Runs on 5+ vendor GPUs without code changes | CI across AMD/Intel/Qualcomm/ARM/NVIDIA |
| **Energy efficiency** | 2-5× less power than CUDA equivalent | `power.py` budget + measured watts |
| **Hardware minimum** | Runs on integrated GPU (Intel Iris Xe, Apple M2 GPU) | `pip install -e .` + smoke test |
| **Launch overhead** | < 1 μs per kernel (after warmup) | Profile kernel launch timing |
| **Memory movement** | 0 bytes for ops that don't need it | Track memcpy bytes vs baseline |
| **Graph execution** | 10-30× fewer kernel launches vs per-op | Graph IR benchmark |
| **Dispatch coverage** | 200+ ATen ops supported | `tensor.py` dispatch table count |
| **Distributed** | DDP trains ResNet-50 across 2+ devices | `torchdistributed` benchmark |
| **Sovereignty** | Zero network required for inference | `verify_offline_capable()` → True |
| **Developer productivity** | `torch.compile(model)` works end-to-end | `torch.compile` benchmark |

---

## VII. THE "WHY THIS WILL WORK" ARGUMENT

### Why OjasX Can Succeed Where Others Failed:

1. **We're not just "CUDA but open"** — we solve real problems CUDA doesn't address:
   - Energy proportionality (power budget API)
   - Hardware heterogeneity (auto-placement)
   - Sovereignty (AGPL guarantee, offline capability)
   - Debugging/visibility (profilers, memory snapshots)

2. **We build on existing strength**, not from scratch:
   - JIT fusion already works (8/10 quality)
   - Autograd has 20+ backward passes (7/10)
   - Liquid primitives are novel (8/10)
   - Memory slab allocator is solid (8.5/10)

3. **The gaps are well-understood** — we identified them precisely:
   - Data movement bottleneck (quantified)
   - Launch overhead (quantified) 
   - Dispatch incompleteness (counted missing ops)
   - Memory management gaps (listed)
   - Distributed absent (marked missing)

4. **Incremental path to V1.0** — each phase builds on previous:
   - Phase 0 makes it faster immediately
   - Phase 1 enables graph execution
   - Phase 2 adds intelligence
   - Phase 3 makes it production
   - Phase 4 adds differentiation
   - Phase 5 releases V1.0

5. **The differentiators are defensible** — no other open-source framework has:
   - Energy-proportional power API ✓
   - True heterogeneous scheduling ✓
   - AGPL sovereign guarantee ✓
   - Persistent kernel state ✓
   - AOT compilation for edge ✓
   - Offline-zero-telemetry ✓

---

## VIII. IMMEDIATE NEXT STEPS (What to Do Today)

1. **Read the key files** to current state:
   - `torchcl/runtime/memory.py` — plan zero-copy rewrite
   - `torchcl/ops/engine.py` — plan graph execution integration  
   - `torchcl/tensor.py` — plan dispatch table completion
   - `torchcl/jit/compiler.py` — plan IR rewrite
   - `torchcl/liquid/dispatch.py` — plan generalization

2. **Set up benchmark environment**:
   ```bash
   # Install test hardware
   pip install -e ".[dev]"
   # Verify runs on your GPU
   python -c "import torchcl; print(torchcl.get_device_info())"
   python tests/test_smoke.py
   
   # Set up CUDA comparison if available
   pip install torch  # CUDA version for comparison
   ```

3. **Choose Phase 0 starting point**:
   - If data movement is your priority → start with `memory.py` zero-copy
   - If speed is your priority → start with `engine.py` graph execution
   - If completeness is your priority → start with `tensor.py` dispatch table

4. **Recruit / assign** based on skill:
   - Low-level OpenCL kernel writer → kernel files + precision.cl
   - IR/compiler engineer → jit/ir.py + compiler.py rewrite
   - Systems architect → dispatch.py + power.py + distributed.py
   - QA/test engineer → benchmark suite + CI across devices

---

## IX. CLOSING THOUGHT

The difference between "CUDA but open" and "something genuinely superior" is ** solving the problems CUDA can't or won't solve.**

OjasX can be that something if we:
1. **Eliminate the CPU bottleneck** (zero-copy memory fabric)
2. **Make it run on any hardware** (heterogeneous auto-placement)  
3. **Make it energy-proportional** (power budget API)
4. **Make it sovereign/offline** (AGPL guarantee)
5. **Make it production-ready** (full feature set + debugging)

Each of these is achievable in 2-8 weeks of focused engineering. Together, they create a compute stack that doesn't just match CUDA — it **surpasses it in ways that matter** to actual users: cost, energy, hardware accessibility, and freedom from vendor lock-in.

This plan is ambitious but the foundation (the existing OjasX code) is genuinely strong. The path is clear. The question is execution depth.

---
*Engineering Plan v0.1 — Generated for OjasX TorchCL Project*
*Current Date: 2026-08-18*