"""
PyTorch 2.x torch.compile Integration Tests for OjasX Backend.
"""

import pytest
import torch
import torchcl


class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(32, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ActivationNet(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(torch.relu(x * 2.0 - 1.0))


def test_torch_compile_mlp():
    torch.manual_seed(42)
    model = SimpleMLP()
    x = torch.randn(8, 32)

    with torch.no_grad():
        ref_out = model(x)

    opt_model = torch.compile(model, backend="ojasx")
    with torch.no_grad():
        opt_out = opt_model(x)

    assert opt_out.shape == (8, 10)
    assert torch.allclose(opt_out, ref_out, atol=1e-3)


def test_torch_compile_activations():
    net = ActivationNet()
    x = torch.randn(16, 64)

    with torch.no_grad():
        ref_out = net(x)

    opt_net = torch.compile(net, backend="ojasx")
    with torch.no_grad():
        opt_out = opt_net(x)

    assert opt_out.shape == (16, 64)
    assert torch.allclose(opt_out, ref_out, atol=1e-3)


def test_torch_compile_opencl_backend():
    model = SimpleMLP()
    x = torch.randn(4, 32)
    opt_model = torch.compile(model, backend="opencl")
    out = opt_model(x)
    assert out.shape == (4, 10)


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing PyTorch 2.x torch.compile with OjasX Backend")
    print("=" * 60)
    test_torch_compile_mlp()
    print("  [PASS] torch.compile SimpleMLP")
    test_torch_compile_activations()
    print("  [PASS] torch.compile ActivationNet")
    test_torch_compile_opencl_backend()
    print("  [PASS] torch.compile opencl backend")
    print("\nALL TORCH.COMPILE TESTS PASSED!")
