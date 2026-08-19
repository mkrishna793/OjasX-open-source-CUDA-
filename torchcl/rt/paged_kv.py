"""
ojasRT — Liquid Paged-KV Cache Runtime for High-Throughput LLM Inference.
Eliminates memory fragmentation and enables dynamic sequence scaling on OpenCL GPUs.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import torch
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor


class PagedKVCache:
    """Manages non-contiguous physical memory blocks for streaming Key-Value attention states."""
    def __init__(
        self,
        num_blocks: int = 128,
        block_size: int = 16,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Pre-allocated physical memory pools [num_blocks, block_size, num_heads, head_dim]
        self.k_pool = torchcl.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype)
        self.v_pool = torchcl.zeros(num_blocks, block_size, num_heads, head_dim, dtype=dtype)

        # Free block list
        self.free_blocks = list(range(num_blocks))
        # Logical sequence -> list of physical block IDs
        self.block_tables: Dict[int, List[int]] = {}
        self.seq_lengths: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int) -> None:
        """Initialize a new sequence context."""
        if seq_id in self.block_tables:
            self.free_sequence(seq_id)
        self.block_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def append_kv(self, seq_id: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """
        Append new key and value states for a sequence.
        k_new, v_new: [num_tokens, num_heads, head_dim]
        """
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id)

        num_tokens = k_new.shape[0]
        cur_len = self.seq_lengths[seq_id]
        total_len = cur_len + num_tokens
        self.seq_lengths[seq_id] = total_len

        # Determine required number of blocks
        needed_blocks = math.ceil(total_len / self.block_size)
        while len(self.block_tables[seq_id]) < needed_blocks:
            if not self.free_blocks:
                raise RuntimeError("Out of GPU Paged-KV memory blocks!")
            self.block_tables[seq_id].append(self.free_blocks.pop(0))

        # Copy data into allocated blocks
        k_cpu = torchcl.to_cpu(k_new) if is_opencl_tensor(k_new) else k_new
        v_cpu = torchcl.to_cpu(v_new) if is_opencl_tensor(v_new) else v_new

        k_pool_cpu = torchcl.to_cpu(self.k_pool)
        v_pool_cpu = torchcl.to_cpu(self.v_pool)

        for i in range(num_tokens):
            global_pos = cur_len + i
            block_idx = global_pos // self.block_size
            offset = global_pos % self.block_size
            phys_block = self.block_tables[seq_id][block_idx]

            k_pool_cpu[phys_block, offset] = k_cpu[i]
            v_pool_cpu[phys_block, offset] = v_cpu[i]

        self.k_pool = torchcl.to_opencl(k_pool_cpu)
        self.v_pool = torchcl.to_opencl(v_pool_cpu)

    def get_kv_view(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve contiguous key and value tensors for attention computation."""
        length = self.seq_lengths[seq_id]
        blocks = self.block_tables[seq_id]

        k_pool_cpu = torchcl.to_cpu(self.k_pool)
        v_pool_cpu = torchcl.to_cpu(self.v_pool)

        k_out = torch.zeros(length, self.num_heads, self.head_dim, dtype=self.dtype)
        v_out = torch.zeros(length, self.num_heads, self.head_dim, dtype=self.dtype)

        for i in range(length):
            block_idx = i // self.block_size
            offset = i % self.block_size
            phys_block = blocks[block_idx]
            k_out[i] = k_pool_cpu[phys_block, offset]
            v_out[i] = v_pool_cpu[phys_block, offset]

        return torchcl.to_opencl(k_out), torchcl.to_opencl(v_out)

    def free_sequence(self, seq_id: int) -> None:
        """Recycle all physical blocks allocated to a sequence."""
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]
