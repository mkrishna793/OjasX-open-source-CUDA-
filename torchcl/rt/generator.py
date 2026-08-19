"""
ojasRT — High-Performance Streaming LLM Generator for OjasX.
"""

from __future__ import annotations
from typing import List, Generator
import torch
import torch.nn as nn
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor
from torchcl.rt.paged_kv import PagedKVCache


class LLMGenerator:
    """Autoregressive streaming generator powered by OpenCL GPU execution."""
    def __init__(self, model: nn.Module, paged_kv: PagedKVCache | None = None):
        self.model = model
        self.paged_kv = paged_kv or PagedKVCache()

    @torch.no_grad()
    def generate(
        self,
        prompt_tensor: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate new token representations autoregressively.
        prompt_tensor: [1, seq_len, dim]
        """
        curr = prompt_tensor
        if not is_opencl_tensor(curr):
            curr = to_opencl(curr)

        outputs = [curr]
        for step in range(max_new_tokens):
            # Run model forward pass on GPU
            logits = self.model(curr)
            # Take last token representation
            next_token = logits[:, -1:, :]
            if temperature != 1.0 and temperature > 0.0:
                next_token = next_token / temperature

            outputs.append(next_token)
            curr = torch.cat([curr, next_token], dim=1)

        return curr


__all__ = ["LLMGenerator"]
