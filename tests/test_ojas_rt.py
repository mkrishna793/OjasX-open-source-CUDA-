"""
Test Suite for ojasRT: Liquid Paged-KV Cache & Streaming Inference Runtime.
"""

import pytest
import torch
import torchcl
import torchcl.dnn as dnn
import torchcl.rt as rt


def test_ojas_rt_paged_kv_allocation():
    cache = rt.PagedKVCache(num_blocks=16, block_size=4, num_heads=2, head_dim=8)

    seq_0 = 101
    k1 = torch.randn(6, 2, 8)
    v1 = torch.randn(6, 2, 8)

    cache.append_kv(seq_0, k1, v1)
    assert len(cache.block_tables[seq_0]) == 2  # 6 tokens -> 2 blocks (size 4 each)
    assert cache.seq_lengths[seq_0] == 6

    # Append 3 more tokens (total 9 tokens -> 3 blocks)
    k2 = torch.randn(3, 2, 8)
    v2 = torch.randn(3, 2, 8)
    cache.append_kv(seq_0, k2, v2)
    assert len(cache.block_tables[seq_0]) == 3
    assert cache.seq_lengths[seq_0] == 9

    # Retrieve reconstructed view
    k_view, v_view = cache.get_kv_view(seq_0)
    k_view_cpu = torchcl.to_cpu(k_view)
    assert k_view_cpu.shape == (9, 2, 8)

    # Free sequence
    cache.free_sequence(seq_0)
    assert len(cache.free_blocks) == 16


def test_ojas_rt_llm_generator():
    torch.manual_seed(42)
    model = dnn.TransformerBlock(dim=32, num_heads=4, hidden_dim=64)
    generator = rt.LLMGenerator(model=model)

    prompt = torch.randn(1, 4, 32)
    generated = generator.generate(prompt, max_new_tokens=4)
    gen_cpu = torchcl.to_cpu(generated)

    # 4 initial tokens + 4 new tokens = 8 tokens
    assert gen_cpu.shape == (1, 8, 32)
    assert not torch.isnan(gen_cpu).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
