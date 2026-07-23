import torch
import numpy as np
import weakref
from torchcl.ops.engine import get_engine
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.runtime.context import synchronize
from torchcl.api import (
    _opencl_buffers,
    _wrap_output,
    _get_buf,
    _get_shape,
    _get_dtype,
    _buffer_lock,
)

HAS_CPP_EXTENSION = False

# ── Memory callbacks for C++ Allocator ───────────────────────────────

_cpp_buffers = {}

def cpp_allocate(size: int) -> int:
    """Invoked by C++ c10::Allocator to allocate OpenCL buffer."""
    cl_buf = get_buffer_pool().allocate(size)
    ptr = cl_buf.buffer.int_ptr
    with _buffer_lock:
        _opencl_buffers[ptr] = cl_buf
        _cpp_buffers[ptr] = cl_buf
    return ptr

def cpp_free(ptr: int) -> None:
    """Invoked by C++ c10::Allocator to free OpenCL buffer."""
    with _buffer_lock:
        _cpp_buffers.pop(ptr, None)
        cl_buf = _opencl_buffers.pop(ptr, None)
    if cl_buf is not None:
        get_buffer_pool().free(cl_buf)

# ── Attempt to load C++ extension ────────────────────────────────────

try:
    import torchcl._C as _C
    torch._C._rename_privateuse1_backend("opencl")
    _C.register_allocator(cpp_allocate, cpp_free)
    HAS_CPP_EXTENSION = True
except ImportError:
    pass

# ── Operator implementations for PrivateUse1 dispatch key ─────────────

# Create PyTorch Library for PrivateUse1 dispatch key implementations
try:
    my_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
except Exception:
    my_lib = None

if my_lib is not None:
    def cl_copy_(self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """Handles CPU <-> GPU copy operations for opencl tensors."""
        engine = get_engine()
        
        # Self is on opencl, src is on CPU
        if self.device.type == "privateuseone" and src.device.type == "cpu":
            self_buf = _opencl_buffers[self.data_ptr()]
            # Ensure src is contiguous and float32
            src_np = src.detach().cpu().numpy().astype(np.float32)
            get_buffer_pool().host_to_device(src_np, self_buf)
            
        # Self is on CPU, src is on opencl
        elif self.device.type == "cpu" and src.device.type == "privateuseone":
            src_buf = _opencl_buffers[src.data_ptr()]
            self_np = np.empty(src.shape, dtype=np.float32)
            get_buffer_pool().device_to_host(self_np, src_buf)
            # Copy data back to self
            self.copy_(torch.from_numpy(self_np))
            
        # Both self and src are on opencl
        elif self.device.type == "privateuseone" and src.device.type == "privateuseone":
            self_buf = _opencl_buffers[self.data_ptr()]
            src_buf = _opencl_buffers[src.data_ptr()]
            # Run simple elementwise copy kernel
            n = int(np.prod(src.shape))
            engine.run_elementwise_unary("copy_f32", src_buf, self_buf, n)
            
        return self

    my_lib.impl("copy_", cl_copy_)

    def cl_empty(size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
        """Standard tensor allocation, handled automatically by the C++ allocator."""
        pass

    my_lib.impl("empty.memory_format", cl_empty)

    # Unary implementations
    def _register_unary(op_name, kernel_name):
        def unary_op(a: torch.Tensor) -> torch.Tensor:
            engine = get_engine()
            out = torch.empty_like(a)
            a_buf = _opencl_buffers[a.data_ptr()]
            out_buf = _opencl_buffers[out.data_ptr()]
            n = int(np.prod(a.shape))
            engine.run_elementwise_unary(kernel_name, a_buf, out_buf, n)
            return out
        my_lib.impl(op_name, unary_op)

    _register_unary("relu", "relu_f32")
    _register_unary("sigmoid", "sigmoid_f32")
    _register_unary("tanh", "tanh_f32")
    _register_unary("gelu", "gelu_f32")
    _register_unary("silu", "silu_f32")
    _register_unary("neg", "neg_f32")
    _register_unary("abs", "abs_f32")
    _register_unary("exp", "exp_f32")
    _register_unary("log", "log_f32")
    _register_unary("sqrt", "sqrt_f32")

    # Binary implementations
    def _register_binary(op_name, kernel_name):
        def binary_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            engine = get_engine()
            out = torch.empty_like(a)
            a_buf = _opencl_buffers[a.data_ptr()]
            b_buf = _opencl_buffers[b.data_ptr()]
            out_buf = _opencl_buffers[out.data_ptr()]
            n = int(np.prod(a.shape))
            engine.run_elementwise_binary(kernel_name, a_buf, b_buf, out_buf, n)
            return out
        my_lib.impl(op_name, binary_op)

    _register_binary("add.Tensor", "add_f32")
    _register_binary("sub.Tensor", "sub_f32")
    _register_binary("mul.Tensor", "mul_f32")
    _register_binary("div.Tensor", "div_f32")

    # Matrix multiplication
    def _get_contiguous_buf(t: torch.Tensor) -> CLBuffer:
        buf = _get_buf(t)
        t_shape = _get_shape(t)
        
        is_transposed = False
        if len(t_shape) == 2 and buf.shape != t_shape:
            is_transposed = True
        elif not t.is_contiguous() and len(t.shape) == 2 and t.stride(0) == 1 and t.stride(1) == t.shape[0]:
            is_transposed = True
            
        if is_transposed:
            engine = get_engine()
            M_orig, N_orig = buf.shape[0], buf.shape[1]
            out_buf = get_buffer_pool().allocate(M_orig * N_orig * 4, np.dtype(np.float32), t_shape)
            engine.run_transpose(buf, out_buf, M_orig, N_orig)
            return out_buf
            
        if t.is_contiguous():
            return buf
            
        t_cpu = t.cpu().contiguous()
        return get_engine().tensor_to_buffer(t_cpu)

    def cl_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        a_shape = _get_shape(a)
        b_shape = _get_shape(b)
        out_shape = (a_shape[0], b_shape[1])
        out = torch.empty(*out_shape, dtype=a.dtype)
        out._torchcl_shape = out_shape
        
        a_buf = _get_contiguous_buf(a)
        b_buf = _get_contiguous_buf(b)
        
        out_ptr = out.data_ptr()
        with _buffer_lock:
            if out_ptr not in _opencl_buffers:
                n = int(np.prod(out_shape))
                out_buf = get_buffer_pool().allocate(n * 4, np.dtype(np.float32), out_shape)
                _opencl_buffers[out_ptr] = out_buf
                _cpp_buffers[out_ptr] = out_buf
                weakref.finalize(out, cpp_free, out_ptr)
            else:
                out_buf = _opencl_buffers[out_ptr]
        
        engine.run_matmul(a_buf, b_buf, out_buf, a_shape[0], b_shape[1], a_shape[1])
        
        a_orig_buf = _get_buf(a)
        b_orig_buf = _get_buf(b)
            
        if a_buf is not a_orig_buf:
            synchronize()
            get_buffer_pool().free(a_buf)
        if b_buf is not b_orig_buf:
            synchronize()
            get_buffer_pool().free(b_buf)
            
        return out

    my_lib.impl("mm", cl_mm)
