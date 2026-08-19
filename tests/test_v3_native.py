"""
Test V3 Native: Verifies PyTorch C-extension integration and OpenCL native ops.
"""

import sys
import torch
import torch.nn as nn
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor


def test_v3_native():
    print("\n--- Test 1: Tensor .to('opencl') ---")
    x = torch.randn(32, 64)
    x_cl = x.to("opencl")
    assert is_opencl_tensor(x_cl)
    x_back = x_cl.to("cpu")
    diff = (x - x_back).abs().max().item()
    assert diff < 1e-4, f"Round-trip difference: {diff}"

    print("\n--- Test 2: Native operations dispatch (x + y, relu) ---")
    y = torch.randn(32, 64)
    y_cl = y.to("opencl")
    z_cl = x_cl + y_cl
    assert is_opencl_tensor(z_cl)
    z_cpu = z_cl.to("cpu")
    expected_add = x + y
    diff = (z_cpu - expected_add).abs().max().item()
    assert diff < 1e-4, f"Add difference: {diff}"

    r_cl = torch.relu(z_cl)
    assert is_opencl_tensor(r_cl)
    r_cpu = r_cl.to("cpu")
    expected_relu = torch.relu(expected_add)
    diff = (r_cpu - expected_relu).abs().max().item()
    assert diff < 1e-4, f"ReLU difference: {diff}"

    print("\n--- Test 3: nn.Linear module on OpenCL ---")
    linear = nn.Linear(64, 128)
    linear_cpu_out = linear(x)

    linear_cl = linear.to("opencl")
    out_cl = linear_cl(x_cl)
    out_cpu = out_cl.to("cpu")
    diff = (out_cpu - linear_cpu_out).abs().max().item()
    assert diff < 1e-3, f"Linear difference: {diff}"


if __name__ == "__main__":
    test_v3_native()
    print("ALL V3 NATIVE TESTS PASSED!")
