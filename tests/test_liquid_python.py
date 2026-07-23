import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchcl
from torchcl.liquid import get_liquid_engine
from torchcl.api import to_cpu

print("============================================================")
print("  OjasX Liquid Compute & Monoidal Ring AllReduce Test")
print("============================================================")

liquid = get_liquid_engine()

# 1. Telemetry
telemetry = liquid.get_telemetry()
print("Discovering multi-GPU cluster topology:")
for dev in telemetry:
    print(f"  - Device #{dev['id']}: {dev['name']} [{dev['vram_mb']} MB VRAM | {dev['bw_gbps']} GB/s]")

# 2. Fluid Partition
x = torch.randn(12, 8)
gpu_chunks = liquid.fluid_partition(x)
print(f"\nFluidly partitioned tensor {tuple(x.shape)} into {len(gpu_chunks)} GPU chunks.")
assert len(gpu_chunks) == 3

# 3. Monoidal Ring AllReduce
allreduced_gpu = liquid.ring_allreduce(gpu_chunks, op="SUM")
allreduced_cpu = to_cpu(allreduced_gpu)

expected = gpu_chunks[0] + gpu_chunks[1] + gpu_chunks[2]
expected_cpu = to_cpu(expected)

diff = torch.abs(allreduced_cpu - expected_cpu).max()
print(f"Ring AllReduce max diff: {diff}")
assert torch.allclose(allreduced_cpu, expected_cpu, atol=1e-4)

print("\n============================================================")
print("  LIQUID COMPUTE & RING ALLREDUCE PASSED 100%!")
print("============================================================")
