import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torchcl
from torchcl.tensor import OjasXTensor

print("=" * 60)
print("  OjasX V3 Native Integration Test")
print("=" * 60)

passed = 0
failed = 0

def check(name, got, expected, atol=1e-3):
    global passed, failed
    if isinstance(got, torch.Tensor) and torchcl.is_opencl_tensor(got):
        got = torchcl.to_cpu(got)
    if isinstance(expected, torch.Tensor):
        expected = expected.float()
    
    if torch.allclose(got.float(), expected.float(), atol=atol):
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")
        print(f"         Got:      {got.flatten()[:5]}")
        print(f"         Expected: {expected.flatten()[:5]}")

# ── Test 1: Tensor .to("opencl") and subclassing ──
print("\n--- Test 1: Tensor .to('opencl') ---")

x = torch.randn(4, 8)
x_cl = x.to("opencl")

assert isinstance(x_cl, OjasXTensor), f"Expected OjasXTensor, got {type(x_cl)}"
assert x_cl.device.type == "opencl", f"Expected opencl device type, got {x_cl.device.type}"
check("tensor round-trip values", x_cl.to("cpu"), x)

# ── Test 2: Arithmetic with native operators ──
print("\n--- Test 2: Native operations dispatch (x + y, relu) ---")

y = torch.randn(4, 8)
y_cl = y.to("opencl")

res_cl = x_cl + y_cl
check("elementwise addition", res_cl, x + y)

res_relu = torch.relu(res_cl)
check("relu activation", res_relu, torch.relu(x + y))

# ── Test 3: nn.Linear module integration ──
print("\n--- Test 3: nn.Linear module on OpenCL ---")

linear = nn.Linear(8, 4)
linear.to("opencl")

# Forward pass on OpenCL
out_cl = linear(x_cl)

# Expected forward pass on CPU
weight_cpu = torchcl.to_cpu(linear.weight.data)
bias_cpu = torchcl.to_cpu(linear.bias.data) if linear.bias is not None else None
expected_out = x @ weight_cpu.t()
if bias_cpu is not None:
    expected_out += bias_cpu

check("nn.Linear forward output", out_cl, expected_out)

# ── Test 4: GPU Transpose in cl_mm (C++ path simulation) ──
print("\n--- Test 4: GPU Transpose in cl_mm (C++ path simulation) ---")

from torchcl.runtime.privateuse1 import cl_mm
from torchcl.api import _opencl_buffers, to_opencl, to_cpu

# Create CPU tensors
a_cpu = torch.randn(4, 8)
b_cpu = torch.randn(3, 8)

# Move to OpenCL
a_cl = to_opencl(a_cpu)
b_cl = to_opencl(b_cpu)

# Create transposed view on CPU
b_t = b_cl.t()
b_t._torchcl_shape = (b_cpu.shape[1], b_cpu.shape[0])
b_t._elem._torchcl_shape = (b_cpu.shape[1], b_cpu.shape[0])

# Run cl_mm directly, which should invoke GPU-based transpose on b_t
out_gpu_sim = cl_mm(a_cl, b_t)

# Map the returned tensor's data_ptr() to extract it
out_cpu = to_cpu(out_gpu_sim)

print("  a_cpu:\n", a_cpu)
print("  b_cpu:\n", b_cpu)
print("  out_cpu:\n", out_cpu)
print("  expected:\n", a_cpu @ b_cpu.t())
check("cl_mm with GPU transposed input", out_cpu, a_cpu @ b_cpu.t())

# ── Summary ──
print()
print("=" * 60)
print(f"  NATIVE V3 RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
