"""
TorchCL — Universal OpenCL Backend for PyTorch
================================================

Enables PyTorch to run tensor operations on ANY OpenCL-capable GPU:
AMD, Intel, Qualcomm, ARM Mali, and more.

Usage:
    import torchcl

    # Check device
    print(torchcl.get_device_info())

    # Move tensors to OpenCL
    x = torchcl.to_opencl(torch.randn(512, 512))
    y = torchcl.to_opencl(torch.randn(512, 512))

    # Compute on GPU
    z = torchcl.matmul(x, y)
    z = torchcl.relu(z)

    # Get results back
    result = torchcl.to_cpu(z)
"""

__version__ = "0.2.0"
__author__ = "OjasX Contributors"

# Initialize OpenCL runtime on import
from torchcl.runtime.context import (
    init_opencl,
    get_device_info,
    get_context,
    get_queue,
    get_device,
    synchronize,
    is_available,
)
from torchcl.runtime.stream import (
    Stream,
    Event,
    stream,
    current_stream,
    default_stream,
)

# Public API — tensor operations
from torchcl.api import (
    # Data movement
    to_opencl,
    to_cpu,
    is_opencl_tensor,
    # Creation
    zeros,
    zeros_like,
    ones,
    ones_like,
    full,
    randn,
    # Arithmetic
    add,
    sub,
    mul,
    div,
    neg,
    abs_,
    exp,
    log,
    sqrt,
    # Activations
    relu,
    sigmoid,
    tanh_,
    gelu,
    silu,
    leaky_relu,
    softmax,
    # Matrix
    matmul,
    transpose,
    # Reductions
    sum_,
    mean,
    max_,
    min_,
    # Normalization
    layer_norm,
    rms_norm,
    # Loss functions
    cross_entropy_loss,
    mse_loss,
    fused_attention,
    rope,
    swiglu,
)

# Auto-initialize on import
try:
    init_opencl()
    import torchcl.runtime.privateuse1
    _info = get_device_info()
    print(f"[OjasX] OK - Initialized on: {_info['name']}")
    print(f"[OjasX]   OpenCL {_info['version']} | "
          f"{_info['max_compute_units']} CUs | "
          f"{_info['global_mem_size_mb']} MB")
except Exception as e:
    print(f"[OjasX] FAILED - Could not initialize OpenCL: {e}")
    print("[OjasX]   Falling back to CPU-only mode.")

# V3 Native Integration
try:
    from torchcl.tensor import apply_monkeypatches
    apply_monkeypatches()
except Exception as e:
    print(f"[TorchCL] V3 Native Integration Failed: {e}")

# Register torch.compile Dynamo backends
try:
    import torchcl._backend
except Exception as e:
    pass

__all__ = [
    # Info
    "get_device_info",
    "is_available",
    "synchronize",
    # Data movement
    "to_opencl",
    "to_cpu",
    "is_opencl_tensor",
    # Creation
    "zeros", "ones", "full", "randn",
    # Arithmetic
    "add", "sub", "mul", "div", "neg", "abs_", "exp", "log", "sqrt",
    # Activations
    "relu", "sigmoid", "tanh_", "gelu", "silu", "leaky_relu", "softmax",
    # Matrix
    "matmul", "transpose",
    # Reductions
    "sum_", "mean", "max_", "min_",
    # Normalization
    "layer_norm", "rms_norm",
    # LLM & Loss
    "cross_entropy_loss", "mse_loss", "fused_attention", "rope", "swiglu",
    # Streams & Events
    "Stream", "Event", "stream", "current_stream", "default_stream",
]

