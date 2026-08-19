"""
Test Suite for ojasMesh: Heterogeneous Collective Engine.
"""

import pytest
import torch
import torchcl
import torchcl.mesh as mesh


def test_ojas_mesh_all_reduce_heterogeneous():
    nodes = [
        mesh.DeviceNode(0, "OpenCL-IrisXe", bandwidth_weight=1.0),
        mesh.DeviceNode(1, "Intel-CPU-Node", bandwidth_weight=0.5),
    ]
    cluster = mesh.HeterogeneousMesh(nodes)

    t_gpu = torchcl.to_opencl(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    t_cpu = torch.tensor([10.0, 20.0, 30.0, 40.0])

    expected = torch.tensor([11.0, 22.0, 33.0, 44.0])

    reduced_gpu, reduced_cpu = cluster.all_reduce([t_gpu, t_cpu], op="sum")

    gpu_res = torchcl.to_cpu(reduced_gpu)
    assert torch.allclose(gpu_res, expected)
    assert torch.allclose(reduced_cpu, expected)


def test_ojas_mesh_broadcast():
    cluster = mesh.HeterogeneousMesh()
    src = torchcl.to_opencl(torch.tensor([5.0, 10.0, 15.0]))

    bcast_gpu, bcast_cpu = cluster.broadcast(src, src_idx=0)
    assert torch.allclose(torchcl.to_cpu(bcast_gpu), torch.tensor([5.0, 10.0, 15.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
