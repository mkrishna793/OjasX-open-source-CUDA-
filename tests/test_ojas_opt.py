"""
Test Suite for ojasOpt: In-Kernel Fused Thermodynamic Optimizers.
"""

import pytest
import torch
import torchcl
import torchcl.optim as optim


def test_ojas_opt_adamw_parity():
    torch.manual_seed(42)
    # CPU Reference Model & Optimizer
    w_ref = torch.nn.Parameter(torch.randn(32, 32))
    opt_ref = torch.optim.AdamW([w_ref], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)

    # OjasX GPU Optimizer
    w_cl = torch.nn.Parameter(w_ref.clone().detach())
    opt_cl = optim.AdamW([w_cl], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)

    # 5 Optimization steps
    for step in range(5):
        grad = torch.randn(32, 32)
        w_ref.grad = grad.clone()
        w_cl.grad = grad.clone()

        opt_ref.step()
        opt_cl.step()

        diff = (w_cl - w_ref).abs().max().item()
        assert diff < 1e-4, f"Step {step}: AdamW difference {diff} exceeds threshold"


def test_ojas_opt_lion_convergence():
    torch.manual_seed(42)
    w = torch.nn.Parameter(torch.randn(16, 16))
    opt = optim.Lion([w], lr=1e-3, weight_decay=1e-2)

    initial_norm = w.norm().item()
    for _ in range(10):
        w.grad = torch.randn(16, 16) * 0.1
        opt.step()

    final_norm = w.norm().item()
    assert not torch.isnan(w).any(), "Lion resulted in NaNs"
    assert final_norm != initial_norm, "Weights did not update"


def test_ojas_opt_sgd_parity():
    torch.manual_seed(42)
    w_ref = torch.nn.Parameter(torch.randn(16, 16))
    opt_ref = torch.optim.SGD([w_ref], lr=1e-2, momentum=0.9, weight_decay=1e-3)

    w_cl = torch.nn.Parameter(w_ref.clone().detach())
    opt_cl = optim.SGD([w_cl], lr=1e-2, momentum=0.9, weight_decay=1e-3)

    for step in range(5):
        grad = torch.randn(16, 16)
        w_ref.grad = grad.clone()
        w_cl.grad = grad.clone()

        opt_ref.step()
        opt_cl.step()

        diff = (w_cl - w_ref).abs().max().item()
        assert diff < 1e-4, f"Step {step}: SGD difference {diff} exceeds threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
