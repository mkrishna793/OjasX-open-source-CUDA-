"""
TorchCL Public API — High-level functions for OpenCL tensor operations.

This is what users interact with. All functions accept PyTorch tensors,
run the computation on OpenCL, and return PyTorch tensors.

Usage:
    import torchcl
    x = torchcl.to_opencl(torch.randn(100, 100))
    y = torchcl.to_opencl(torch.randn(100, 100))
    z = torchcl.add(x, y)
    result = torchcl.to_cpu(z)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import weakref
import threading
from torchcl.ops.engine import get_engine
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.runtime.context import synchronize as _sync

# ── Internal storage: maps tensor data_ptr → CLBuffer ────────────────
_opencl_buffers = weakref.WeakValueDictionary()
_buffer_lock = threading.RLock()


def _cleanup_buffer(cl_buf: CLBuffer) -> None:
    get_buffer_pool().free(cl_buf)


_tensor_id_counter = 0


def _next_id() -> int:
    global _tensor_id_counter
    _tensor_id_counter += 1
    return _tensor_id_counter


def _make_handle(shape: tuple, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, int]:
    """Create a CPU placeholder tensor and assign it a unique ID."""
    handle = torch.empty(1, dtype=dtype)  # tiny placeholder
    tid = _next_id()
    handle._torchcl_id = tid  # type: ignore[attr-defined]
    handle._torchcl_shape = shape  # type: ignore[attr-defined]
    handle._torchcl_dtype = dtype  # type: ignore[attr-defined]
    return handle, tid


def _get_buf(tensor: torch.Tensor) -> CLBuffer:
    """Get the OpenCL buffer for a TorchCL tensor handle."""
    if hasattr(tensor, "_elem"):
        return _get_buf(tensor._elem)
    tid = getattr(tensor, "_torchcl_id", None)
    if tid is not None and tid in _opencl_buffers:
        return _opencl_buffers[tid]
    ptr = tensor.data_ptr()
    with _buffer_lock:
        if ptr in _opencl_buffers:
            return _opencl_buffers[ptr]
        try:
            storage_ptr = tensor.untyped_storage().data_ptr()
            if storage_ptr in _opencl_buffers:
                return _opencl_buffers[storage_ptr]
        except Exception:
            pass
        if _is_lazy(tensor):
            materialize_lazy_tensor(tensor)
            return _opencl_buffers[tensor._torchcl_id]
        raise ValueError(
            "This tensor is not on the OpenCL device. "
            "Use torchcl.to_opencl(tensor) first."
        )


def _is_lazy(tensor: torch.Tensor) -> bool:
    if hasattr(tensor, "_elem"):
        return _is_lazy(tensor._elem)
    return getattr(tensor, "_lazy_inputs", None) is not None


def _create_lazy_unary(op_name: str, a: torch.Tensor) -> torch.Tensor:
    a_elem = a._elem if hasattr(a, "_elem") else a
    shape = _get_shape(a_elem)
    dtype = _get_dtype(a_elem)
    handle, tid = _make_handle(shape, dtype)
    
    if _is_lazy(a_elem):
        handle._lazy_inputs = a_elem._lazy_inputs
        handle._lazy_op_type = a_elem._lazy_op_type
        if hasattr(a_elem, "_lazy_binary_op"):
            handle._lazy_binary_op = a_elem._lazy_binary_op
        handle._lazy_ops = list(a_elem._lazy_ops) + [op_name]
    else:
        handle._lazy_inputs = [a_elem]
        handle._lazy_op_type = "unary"
        handle._lazy_ops = [op_name]
        
    from torchcl.tensor import OjasXTensor
    return OjasXTensor(handle)


def _create_lazy_binary(op_name: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_elem = a._elem if hasattr(a, "_elem") else a
    b_elem = b._elem if hasattr(b, "_elem") else b
    
    if _is_lazy(a_elem):
        materialize_lazy_tensor(a_elem)
    if _is_lazy(b_elem):
        materialize_lazy_tensor(b_elem)
        
    shape = _get_shape(a_elem)
    dtype = _get_dtype(a_elem)
    handle, tid = _make_handle(shape, dtype)
    
    handle._lazy_inputs = [a_elem, b_elem]
    handle._lazy_op_type = "binary_then_unary"
    handle._lazy_binary_op = op_name
    handle._lazy_ops = []
    
    from torchcl.tensor import OjasXTensor
    return OjasXTensor(handle)


def materialize_lazy_tensor(tensor: torch.Tensor) -> None:
    tensor_elem = tensor._elem if hasattr(tensor, "_elem") else tensor
    with _buffer_lock:
        if tensor_elem._torchcl_id in _opencl_buffers:
            return

    inputs = tensor_elem._lazy_inputs
    for inp in inputs:
        if _is_lazy(inp):
            materialize_lazy_tensor(inp)

    engine = get_engine()
    shape = tensor_elem._torchcl_shape
    dtype = tensor_elem._torchcl_dtype
    out_buf = engine.allocate_output(shape, dtype)

    from torchcl.jit.compiler import get_jit_compiler
    jit = get_jit_compiler()
    n = int(np.prod(shape))

    with _buffer_lock:
        if tensor_elem._lazy_op_type == "unary":
            in_buf = _opencl_buffers[inputs[0]._torchcl_id].buffer
            jit.fuse_elementwise_chain(tensor_elem._lazy_ops, n, [in_buf], out_buf.buffer)
        elif tensor_elem._lazy_op_type == "binary_then_unary":
            a_buf = _opencl_buffers[inputs[0]._torchcl_id].buffer
            b_buf = _opencl_buffers[inputs[1]._torchcl_id].buffer
            jit.fuse_binary_then_unary(tensor_elem._lazy_binary_op, tensor_elem._lazy_ops, n, a_buf, b_buf, out_buf.buffer)

        _opencl_buffers[tensor_elem._torchcl_id] = out_buf
    weakref.finalize(tensor_elem, _cleanup_buffer, out_buf)


def _get_shape(tensor: torch.Tensor) -> tuple:
    s = getattr(tensor, "_torchcl_shape", None)
    if s is not None:
        return s
    if hasattr(tensor, "_elem"):
        return _get_shape(tensor._elem)
    return tuple(tensor.shape)


def _get_dtype(tensor: torch.Tensor) -> torch.dtype:
    d = getattr(tensor, "_torchcl_dtype", None)
    if d is not None:
        return d
    if hasattr(tensor, "_elem"):
        return _get_dtype(tensor._elem)
    return tensor.dtype


def _wrap_output(cl_buf: CLBuffer, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Wrap an OpenCL buffer as a TorchCL tensor handle."""
    handle, tid = _make_handle(shape, dtype)
    with _buffer_lock:
        _opencl_buffers[tid] = cl_buf
    weakref.finalize(handle, _cleanup_buffer, cl_buf)
    from torchcl.tensor import OjasXTensor
    return OjasXTensor(handle)


def is_opencl_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is stored on OpenCL."""
    if tensor is None:
        return False
    if hasattr(tensor, "_elem"):
        return is_opencl_tensor(tensor._elem)
    if getattr(tensor, "device", None) is not None and tensor.device.type in ("opencl", "privateuseone"):
        return True
    with _buffer_lock:
        if tensor.data_ptr() in _opencl_buffers:
            return True
        tid = getattr(tensor, "_torchcl_id", None)
        return tid is not None and (tid in _opencl_buffers or _is_lazy(tensor))



# ── Data movement ────────────────────────────────────────────────────

def to_opencl(tensor: torch.Tensor) -> torch.Tensor:
    """Move a CPU tensor to OpenCL device."""
    if is_opencl_tensor(tensor):
        return tensor

    engine = get_engine()
    cl_buf = engine.tensor_to_buffer(tensor)
    return _wrap_output(cl_buf, tuple(tensor.shape), tensor.dtype)


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Move an OpenCL tensor back to CPU."""
    if tensor is None:
        return None
    if hasattr(tensor, "_elem"):
        return to_cpu(tensor._elem)
    if not is_opencl_tensor(tensor):
        return tensor

    engine = get_engine()
    shape = _get_shape(tensor)
    dtype = _get_dtype(tensor)
    return engine.buffer_to_tensor(_get_buf(tensor), shape, dtype)


def synchronize() -> None:
    """Wait for all OpenCL operations to complete."""
    _sync()


# ── Tensor creation ──────────────────────────────────────────────────

def zeros(*shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a zero-filled tensor on OpenCL."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    n = int(np.prod(shape))
    engine = get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    engine.run_fill(cl_buf, 0.0, n)
    return _wrap_output(cl_buf, shape, dtype)


def ones(*shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a ones-filled tensor on OpenCL."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    n = int(np.prod(shape))
    engine = get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    engine.run_fill(cl_buf, 1.0, n)
    return _wrap_output(cl_buf, shape, dtype)


def full(*shape, fill_value: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a constant-filled tensor on OpenCL."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    n = int(np.prod(shape))
    engine = get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    engine.run_fill(cl_buf, fill_value, n)
    return _wrap_output(cl_buf, shape, dtype)


def randn(*shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a random normal tensor on OpenCL."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    cpu_tensor = torch.randn(*shape, dtype=dtype)
    return to_opencl(cpu_tensor)


# ── Arithmetic operations ────────────────────────────────────────────

def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition on OpenCL."""
    if not is_opencl_tensor(a) or not is_opencl_tensor(b):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_binary("add", a, b)


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise subtraction on OpenCL."""
    if not is_opencl_tensor(a) or not is_opencl_tensor(b):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_binary("sub", a, b)


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise multiplication on OpenCL."""
    if not is_opencl_tensor(a) or not is_opencl_tensor(b):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_binary("mul", a, b)


def div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise division on OpenCL."""
    if not is_opencl_tensor(a) or not is_opencl_tensor(b):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_binary("div", a, b)


def neg(a: torch.Tensor) -> torch.Tensor:
    """Element-wise negation on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("neg", a)


def abs_(a: torch.Tensor) -> torch.Tensor:
    """Element-wise absolute value on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("abs", a)


def exp(a: torch.Tensor) -> torch.Tensor:
    """Element-wise exp on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("exp", a)


def log(a: torch.Tensor) -> torch.Tensor:
    """Element-wise log on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("log", a)


def sqrt(a: torch.Tensor) -> torch.Tensor:
    """Element-wise sqrt on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("sqrt", a)


# ── Activation functions ─────────────────────────────────────────────

def relu(a: torch.Tensor) -> torch.Tensor:
    """ReLU activation on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("relu", a)


def sigmoid(a: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("sigmoid", a)


def tanh_(a: torch.Tensor) -> torch.Tensor:
    """Tanh activation on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("tanh", a)


def gelu(a: torch.Tensor) -> torch.Tensor:
    """GELU activation on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("gelu", a)


def silu(a: torch.Tensor) -> torch.Tensor:
    """SiLU activation on OpenCL."""
    if not is_opencl_tensor(a):
        raise ValueError("This tensor is not on the OpenCL device. Use torchcl.to_opencl(tensor) first.")
    return _create_lazy_unary("silu", a)


def leaky_relu(a: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """LeakyReLU activation on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_activation("leaky_relu_f32", _get_buf(a), out_buf, n, neg_slope=negative_slope)
    return _wrap_output(out_buf, shape)


def softmax(a: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax on OpenCL (along last dimension)."""
    engine = get_engine()
    shape = _get_shape(a)
    if len(shape) == 1:
        rows, cols = 1, shape[0]
    else:
        rows = int(np.prod(shape[:-1]))
        cols = shape[-1]
    out_buf = engine.allocate_output(shape)
    engine.run_softmax(_get_buf(a), out_buf, rows, cols)
    return _wrap_output(out_buf, shape)


# ── Matrix operations ────────────────────────────────────────────────

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication on OpenCL: C = A @ B."""
    engine = get_engine()
    a_shape = _get_shape(a)
    b_shape = _get_shape(b)

    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"matmul requires 2D tensors, got {a_shape} and {b_shape}")

    M, K = a_shape
    K2, N = b_shape
    if K != K2:
        raise ValueError(f"matmul dimension mismatch: {a_shape} @ {b_shape}")

    out_shape = (M, N)

    from torchcl.liquid.dispatch import get_dispatcher
    from torchcl.liquid.precision import AdaptivePrecision

    dispatcher = get_dispatcher()
    config = dispatcher.dispatch("matmul", a)

    if config.precision == "half":
        ap = AdaptivePrecision()
        a_fp16 = ap.pack_to_fp16(_get_buf(a), M * K)
        b_fp16 = ap.pack_to_fp16(_get_buf(b), K * N)
        out_fp16 = get_buffer_pool().allocate(M * N * 2, np.dtype(np.float16), out_shape)
        engine.run_matmul_fp16(
            a_fp16, b_fp16, out_fp16, M, N, K,
            use_tiled=(config.strategy == "tiled")
        )
        out_buf = ap.unpack_from_fp16(out_fp16, M * N)
        get_buffer_pool().free(a_fp16)
        get_buffer_pool().free(b_fp16)
        get_buffer_pool().free(out_fp16)
    else:
        out_buf = engine.allocate_output(out_shape)
        engine.run_matmul(
            _get_buf(a), _get_buf(b), out_buf, M, N, K,
            use_tiled=(config.strategy == "tiled")
        )

    return _wrap_output(out_buf, out_shape)


def transpose(a: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D tensor on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    if len(shape) != 2:
        raise ValueError(f"transpose requires 2D tensor, got {shape}")
    M, N = shape
    out_shape = (N, M)
    out_buf = engine.allocate_output(out_shape)
    engine.run_transpose(_get_buf(a), out_buf, M, N)
    return _wrap_output(out_buf, out_shape)


# ── Reduction operations ─────────────────────────────────────────────

def sum_(a: torch.Tensor) -> torch.Tensor:
    """Sum all elements on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(())
    engine.run_reduction("sum_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, ())


def mean(a: torch.Tensor) -> torch.Tensor:
    """Mean of all elements on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    # Sum then divide
    sum_buf = engine.allocate_output(())
    engine.run_reduction("sum_f32", _get_buf(a), sum_buf, n)
    out_buf = engine.allocate_output(())
    engine.run_elementwise_scalar("mul_scalar_f32", sum_buf, 1.0 / n, out_buf, 1)
    engine.free_buffer(sum_buf)
    return _wrap_output(out_buf, ())


def max_(a: torch.Tensor) -> torch.Tensor:
    """Max of all elements on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(())
    engine.run_reduction("max_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, ())


def min_(a: torch.Tensor) -> torch.Tensor:
    """Min of all elements on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(())
    engine.run_reduction("min_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, ())


# ── Normalization operations ─────────────────────────────────────────

def layer_norm(
    a: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    normalized_shape: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Layer normalization on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    N = normalized_shape
    M = int(np.prod(shape)) // N

    out_buf = engine.allocate_output(shape)
    mean_buf = engine.allocate_output((M,))
    rstd_buf = engine.allocate_output((M,))

    engine.run_layer_norm(
        _get_buf(a), _get_buf(weight), _get_buf(bias),
        out_buf, mean_buf, rstd_buf,
        M, N, eps,
    )
    return _wrap_output(out_buf, shape)


def rms_norm(
    a: torch.Tensor,
    weight: torch.Tensor,
    normalized_shape: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """RMS normalization on OpenCL (LLaMA-style)."""
    engine = get_engine()
    shape = _get_shape(a)
    N = normalized_shape
    M = int(np.prod(shape)) // N

    out_buf = engine.allocate_output(shape)
    rrms_buf = engine.allocate_output((M,))

    engine.run_rms_norm(
        _get_buf(a), _get_buf(weight),
        out_buf, rrms_buf,
        M, N, eps,
    )
    return _wrap_output(out_buf, shape)


# ── Loss functions ───────────────────────────────────────────────────

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss on OpenCL (mean reduction).

    Args:
        logits:  [batch, num_classes] — raw logits (not softmax'd)
        targets: [batch] — class indices (as float for OpenCL transfer)
    Returns:
        Scalar loss tensor.
    """
    engine = get_engine()
    logits_shape = _get_shape(logits)
    batch_size = logits_shape[0]
    C = logits_shape[1]

    loss_per_sample_buf = engine.allocate_output((batch_size,))
    log_softmax_buf = engine.allocate_output(logits_shape)

    engine.run_cross_entropy_forward(
        _get_buf(logits), _get_buf(targets),
        loss_per_sample_buf, log_softmax_buf,
        batch_size, C,
    )

    # Mean reduction
    sum_buf = engine.allocate_output((1,))
    engine.run_reduction("sum_f32", loss_per_sample_buf, sum_buf, batch_size)
    out_buf = engine.allocate_output((1,))
    engine.run_elementwise_scalar("mul_scalar_f32", sum_buf, 1.0 / batch_size, out_buf, 1)
    engine.free_buffer(sum_buf)
    engine.free_buffer(loss_per_sample_buf)

    return _wrap_output(out_buf, (1,))


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss on OpenCL."""
    engine = get_engine()
    shape = _get_shape(pred)
    n = int(np.prod(shape))

    per_elem_buf = engine.allocate_output(shape)
    engine.run_mse_forward(_get_buf(pred), _get_buf(target), per_elem_buf, n)

    # Mean reduction
    sum_buf = engine.allocate_output((1,))
    engine.run_reduction("sum_f32", per_elem_buf, sum_buf, n)
    out_buf = engine.allocate_output((1,))
    engine.run_elementwise_scalar("mul_scalar_f32", sum_buf, 1.0 / n, out_buf, 1)
    engine.free_buffer(sum_buf)
    engine.free_buffer(per_elem_buf)

    return _wrap_output(out_buf, (1,))


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute fused scaled dot-product attention on OpenCL.

    Q: [B, H, M, D]
    K: [B, H, N, D]
    V: [B, H, N, D]
    """
    engine = get_engine()
    q_shape = _get_shape(q)
    k_shape = _get_shape(k)

    B, H, M, D = q_shape
    _, _, N, _ = k_shape

    if scale is None:
        scale = D ** -0.5

    out_buf = engine.allocate_output((B, H, M, D))
    engine.run_fused_attention(
        _get_buf(q), _get_buf(k), _get_buf(v), out_buf,
        B, H, M, N, D, scale
    )
    return _wrap_output(out_buf, (B, H, M, D))
