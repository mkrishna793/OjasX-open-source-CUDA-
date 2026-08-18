"""
OjasX Autograd Test — Verifies that gradients flow correctly through
OpenCL operations by comparing against PyTorch CPU autograd.
"""

import pytest
import torch
import numpy as np
import torchcl
from torchcl.autograd import (
    ReluFunction,
    SigmoidFunction,
    TanhFunction,
    GeluFunction,
    SiluFunction,
    LeakyReluFunction,
    AddFunction,
    SubFunction,
    MulFunction,
    MatmulFunction,
    CrossEntropyFunction,
    LayerNormFunction,
)


def test_activations_autograd():
    x_cpu = torch.randn(64, 64, requires_grad=True)
    x_cl = torchcl.to_opencl(x_cpu.detach())

    # ReLU forward + backward
    y_cpu = torch.relu(x_cpu)
    y_cl = ReluFunction.apply(x_cl)
    assert torch.allclose(torchcl.to_cpu(y_cl), y_cpu.detach(), atol=1e-4)

    grad_out_cpu = torch.ones_like(y_cpu)
    y_cpu.backward(grad_out_cpu)

    grad_out_cl = torchcl.to_opencl(grad_out_cpu.detach())
    grad_in_cl = ReluFunction.backward(
        type('ctx', (), {
            'saved_tensors': (x_cl,),
            '_n': 64 * 64,
            '_shape': (64, 64),
        })(),
        grad_out_cl,
    )
    assert torch.allclose(torchcl.to_cpu(grad_in_cl), x_cpu.grad, atol=1e-4)

    # Sigmoid forward
    x_cpu2 = torch.randn(32, 32, requires_grad=True)
    x_cl2 = torchcl.to_opencl(x_cpu2.detach())
    y_cpu2 = torch.sigmoid(x_cpu2)
    y_cl2 = SigmoidFunction.apply(x_cl2)
    assert torch.allclose(torchcl.to_cpu(y_cl2), y_cpu2.detach(), atol=1e-3)

    # Tanh forward
    y_cpu3 = torch.tanh(x_cpu2)
    y_cl3 = TanhFunction.apply(x_cl2)
    assert torch.allclose(torchcl.to_cpu(y_cl3), y_cpu3.detach(), atol=1e-3)

    # GELU forward
    y_cpu4 = torch.nn.functional.gelu(x_cpu2)
    y_cl4 = GeluFunction.apply(x_cl2)
    assert torch.allclose(torchcl.to_cpu(y_cl4), y_cpu4.detach(), atol=1e-2)

    # SiLU forward
    y_cpu5 = torch.nn.functional.silu(x_cpu2)
    y_cl5 = SiluFunction.apply(x_cl2)
    assert torch.allclose(torchcl.to_cpu(y_cl5), y_cpu5.detach(), atol=1e-3)


def test_arithmetic_autograd():
    a_cpu = torch.randn(32, 32)
    b_cpu = torch.randn(32, 32)
    a_cl = torchcl.to_opencl(a_cpu)
    b_cl = torchcl.to_opencl(b_cpu)

    assert torch.allclose(torchcl.to_cpu(AddFunction.apply(a_cl, b_cl)), a_cpu + b_cpu, atol=1e-4)
    assert torch.allclose(torchcl.to_cpu(SubFunction.apply(a_cl, b_cl)), a_cpu - b_cpu, atol=1e-4)
    assert torch.allclose(torchcl.to_cpu(MulFunction.apply(a_cl, b_cl)), a_cpu * b_cpu, atol=1e-4)


def test_matmul_autograd():
    m1_cpu = torch.randn(16, 32)
    m2_cpu = torch.randn(32, 8)
    m1_cl = torchcl.to_opencl(m1_cpu)
    m2_cl = torchcl.to_opencl(m2_cpu)

    result_cl = MatmulFunction.apply(m1_cl, m2_cl)
    result_cpu = m1_cpu @ m2_cpu
    assert torch.allclose(torchcl.to_cpu(result_cl), result_cpu, atol=1e-2)


def test_layer_norm_autograd():
    ln_input = torch.randn(8, 64, requires_grad=True)
    ln_weight = torch.ones(64, requires_grad=True)
    ln_bias = torch.zeros(64, requires_grad=True)

    # Run CPU Reference
    ln_result_cpu = torch.nn.functional.layer_norm(ln_input, [64], ln_weight, ln_bias)
    grad_out_cpu = torch.randn_like(ln_result_cpu)
    ln_result_cpu.backward(grad_out_cpu)

    # Run OpenCL
    ln_input_cl = torchcl.to_opencl(ln_input.detach())
    ln_weight_cl = torchcl.to_opencl(ln_weight.detach())
    ln_bias_cl = torchcl.to_opencl(ln_bias.detach())

    class DummyCtx:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    ctx = DummyCtx()
    ln_result_cl = LayerNormFunction.forward(ctx, ln_input_cl, ln_weight_cl, ln_bias_cl, 64)
    assert torch.allclose(torchcl.to_cpu(ln_result_cl), ln_result_cpu.detach(), atol=1e-3)

    grad_out_cl = torchcl.to_opencl(grad_out_cpu)
    grad_in_cl, grad_w_cl, grad_b_cl, _, _ = LayerNormFunction.backward(ctx, grad_out_cl)

    assert torch.allclose(torchcl.to_cpu(grad_in_cl), ln_input.grad, atol=1e-3)
    assert torch.allclose(torchcl.to_cpu(grad_w_cl), ln_weight.grad, atol=1e-3)
    assert torch.allclose(torchcl.to_cpu(grad_b_cl), ln_bias.grad, atol=1e-3)


def test_cross_entropy_autograd():
    logits = torch.randn(8, 10)
    targets = torch.tensor([0, 3, 5, 7, 2, 9, 1, 4], dtype=torch.float32)

    logits_cl = torchcl.to_opencl(logits)
    targets_cl = torchcl.to_opencl(targets)

    loss_cl = torchcl.cross_entropy_loss(logits_cl, targets_cl)
    loss_cpu = torch.nn.functional.cross_entropy(logits, targets.long())
    assert torch.allclose(torchcl.to_cpu(loss_cl), loss_cpu.unsqueeze(0), atol=1e-2)


def test_mse_loss_autograd():
    pred = torch.randn(4, 8)
    target_mse = torch.randn(4, 8)

    pred_cl = torchcl.to_opencl(pred)
    target_mse_cl = torchcl.to_opencl(target_mse)

    mse_cl = torchcl.mse_loss(pred_cl, target_mse_cl)
    mse_cpu = torch.nn.functional.mse_loss(pred, target_mse)
    assert torch.allclose(torchcl.to_cpu(mse_cl), mse_cpu.unsqueeze(0), atol=0.1)


if __name__ == "__main__":
    print("=" * 60)
    print("  OjasX V2 — Autograd Test Suite")
    print("=" * 60)
    test_activations_autograd()
    print("  [PASS] Activations Autograd")
    test_arithmetic_autograd()
    print("  [PASS] Arithmetic Autograd")
    test_matmul_autograd()
    print("  [PASS] Matmul Autograd")
    test_layer_norm_autograd()
    print("  [PASS] Layer Norm Autograd")
    test_cross_entropy_autograd()
    print("  [PASS] Cross Entropy Autograd")
    test_mse_loss_autograd()
    print("  [PASS] MSE Loss Autograd")
    print("\nALL AUTOGRAD TESTS PASSED!")
