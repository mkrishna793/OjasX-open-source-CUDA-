"""
Test Suite for ojasDNN: Deep Neural Network & Categorical Attention Primitives.
"""

import pytest
import torch
import torchcl
import torchcl.dnn as dnn


def test_ojas_dnn_rms_norm():
    torch.manual_seed(42)
    x = torch.randn(4, 32, 64)
    weight = torch.ones(64)

    # PyTorch CPU reference
    rms_expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight

    out_cl = dnn.rms_norm(x, weight, eps=1e-5)
    out_cpu = torchcl.to_cpu(out_cl)

    diff = (out_cpu - rms_expected).abs().max().item()
    assert diff < 1e-3, f"Difference too high in RMSNorm: {diff}"


def test_ojas_dnn_swiglu():
    torch.manual_seed(42)
    gate = torch.randn(4, 32, 64)
    up = torch.randn(4, 32, 64)

    expected = torch.nn.functional.silu(gate) * up

    out_cl = dnn.swiglu(gate, up)
    out_cpu = torchcl.to_cpu(out_cl)

    diff = (out_cpu - expected).abs().max().item()
    assert diff < 1e-3, f"Difference too high in SwiGLU: {diff}"


def test_ojas_dnn_flash_attention():
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 32, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    # Reference SDPA
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    out_cl = dnn.flash_attention(q, k, v, is_causal=True)
    out_cpu = torchcl.to_cpu(out_cl)

    diff = (out_cpu - expected).abs().max().item()
    assert diff < 1e-2, f"Difference too high in FlashAttention: {diff}"


def test_ojas_dnn_transformer_block():
    torch.manual_seed(42)
    block = dnn.TransformerBlock(dim=64, num_heads=4, hidden_dim=128)
    x = torch.randn(2, 16, 64)

    out_cl = block(x)
    out_cpu = torchcl.to_cpu(out_cl)

    assert out_cpu.shape == (2, 16, 64)
    assert not torch.isnan(out_cpu).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
