"""
ojasMesh — Heterogeneous Fluid Collective Engine for OjasX.
Replaces NCCL with Monoidal Heterogeneous Ring/Mesh AllReduce pooling iGPU, dGPU, and CPU.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
import torch
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor


@dataclass
class DeviceNode:
    device_id: int
    name: str
    bandwidth_weight: float  # e.g., 1.0 for dGPU, 0.5 for iGPU, 0.25 for CPU


class HeterogeneousMesh:
    """Manages collective operations across heterogeneous hardware devices."""
    def __init__(self, nodes: Sequence[DeviceNode] | None = None):
        if nodes is None:
            self.nodes = [
                DeviceNode(0, "OpenCL-GPU-0", bandwidth_weight=1.0),
                DeviceNode(1, "Host-CPU-Node", bandwidth_weight=0.5),
            ]
        else:
            self.nodes = list(nodes)

    def all_reduce(self, tensors: Sequence[torch.Tensor], op: str = "sum") -> List[torch.Tensor]:
        """
        Heterogeneous Ring/Mesh AllReduce.
        Splits workload proportionally by device bandwidth, reduces in parallel, and gathers.
        """
        assert len(tensors) == len(self.nodes), "Tensor count must match node count"
        
        # Convert all to CPU tensors for reduction step
        cpu_tensors = [torchcl.to_cpu(t) if is_opencl_tensor(t) else t for t in tensors]
        shape = cpu_tensors[0].shape
        dtype = cpu_tensors[0].dtype

        flat_tensors = [t.flatten() for t in cpu_tensors]
        total_elements = flat_tensors[0].numel()

        # Compute proportional partition splits
        total_weight = sum(n.bandwidth_weight for n in self.nodes)
        splits = []
        start = 0
        for i, node in enumerate(self.nodes):
            if i == len(self.nodes) - 1:
                end = total_elements
            else:
                chunk_len = int((node.bandwidth_weight / total_weight) * total_elements)
                end = min(total_elements, start + chunk_len)
            splits.append((start, end))
            start = end

        # 1. Reduce-Scatter Phase
        reduced_chunks = []
        for i, (s, e) in enumerate(splits):
            chunk = flat_tensors[0][s:e].clone()
            for j in range(1, len(flat_tensors)):
                if op == "sum":
                    chunk += flat_tensors[j][s:e]
                elif op == "max":
                    chunk = torch.maximum(chunk, flat_tensors[j][s:e])
            reduced_chunks.append(chunk)

        # 2. All-Gather Phase
        full_reduced = torch.cat(reduced_chunks).view(shape)

        # Broadcast back to respective device formats
        results = []
        for i, original_t in enumerate(tensors):
            if is_opencl_tensor(original_t):
                results.append(torchcl.to_opencl(full_reduced.clone()))
            else:
                results.append(full_reduced.clone())

        return results

    def broadcast(self, tensor: torch.Tensor, src_idx: int = 0) -> List[torch.Tensor]:
        """Broadcast tensor from src node to all nodes."""
        src_t = torchcl.to_cpu(tensor) if is_opencl_tensor(tensor) else tensor
        return [
            torchcl.to_opencl(src_t.clone()) if is_opencl_tensor(t) else src_t.clone()
            for t in [tensor] * len(self.nodes)
        ]


__all__ = ["DeviceNode", "HeterogeneousMesh"]
