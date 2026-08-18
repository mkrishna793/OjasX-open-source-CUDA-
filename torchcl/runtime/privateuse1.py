import torch
import numpy as np
import weakref
from typing import Sequence, Optional, List, Tuple
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
except Exception:
    pass

# ── Helper to ensure contiguous buffer ───────────────────────────────

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


def _allocate_tensor_output(shape: tuple, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, CLBuffer]:
    out = torch.empty(*shape, dtype=dtype)
    out._torchcl_shape = shape
    out_ptr = out.data_ptr()
    n = int(np.prod(shape))
    with _buffer_lock:
        if out_ptr not in _opencl_buffers:
            out_buf = get_buffer_pool().allocate(n * 4, np.dtype(np.float32), shape)
            _opencl_buffers[out_ptr] = out_buf
            _cpp_buffers[out_ptr] = out_buf
            weakref.finalize(out, cpp_free, out_ptr)
        else:
            out_buf = _opencl_buffers[out_ptr]
    return out, out_buf


# ── Operator implementations for PrivateUse1 dispatch key ─────────────

try:
    my_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
except Exception:
    my_lib = None

if my_lib is not None:
    def cl_copy_(self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """Handles CPU <-> GPU copy operations for opencl tensors."""
        engine = get_engine()

        if self.device.type == "privateuseone" and src.device.type == "cpu":
            self_buf = _opencl_buffers.get(self.data_ptr())
            if self_buf is None:
                self_buf = get_buffer_pool().allocate(src.numel() * 4, np.float32, src.shape)
                with _buffer_lock:
                    _opencl_buffers[self.data_ptr()] = self_buf
            src_np = src.detach().cpu().numpy().astype(np.float32)
            get_buffer_pool().host_to_device(src_np, self_buf, non_blocking=non_blocking)

        elif self.device.type == "cpu" and src.device.type == "privateuseone":
            src_buf = _opencl_buffers[src.data_ptr()]
            self_np = np.empty(src.shape, dtype=np.float32)
            get_buffer_pool().device_to_host(src_buf, np.float32, src.shape)
            self.copy_(torch.from_numpy(self_np))

        elif self.device.type == "privateuseone" and src.device.type == "privateuseone":
            self_buf = _opencl_buffers[self.data_ptr()]
            src_buf = _opencl_buffers[src.data_ptr()]
            n = int(np.prod(src.shape))
            engine.run_elementwise_unary("copy_f32", src_buf, self_buf, n)

        return self

    my_lib.impl("copy_", cl_copy_)

    def cl_empty(size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
        pass

    my_lib.impl("empty.memory_format", cl_empty)

    # Unary operations
    def _register_unary(op_name, kernel_name):
        def unary_op(a: torch.Tensor) -> torch.Tensor:
            engine = get_engine()
            shape = _get_shape(a)
            out, out_buf = _allocate_tensor_output(shape, a.dtype)
            a_buf = _get_buf(a)
            n = int(np.prod(shape))
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

    # Binary operations
    def _register_binary(op_name, kernel_name):
        def binary_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            engine = get_engine()
            shape = _get_shape(a)
            out, out_buf = _allocate_tensor_output(shape, a.dtype)
            a_buf = _get_buf(a)
            b_buf = _get_buf(b)
            n = int(np.prod(shape))
            engine.run_elementwise_binary(kernel_name, a_buf, b_buf, out_buf, n)
            return out
        my_lib.impl(op_name, binary_op)

    _register_binary("add.Tensor", "add_f32")
    _register_binary("sub.Tensor", "sub_f32")
    _register_binary("mul.Tensor", "mul_f32")
    _register_binary("div.Tensor", "div_f32")

    # Matrix multiplication: mm
    def cl_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        a_shape = _get_shape(a)
        b_shape = _get_shape(b)
        out_shape = (a_shape[0], b_shape[1])
        out, out_buf = _allocate_tensor_output(out_shape, a.dtype)

        a_buf = _get_contiguous_buf(a)
        b_buf = _get_contiguous_buf(b)

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

    # Matrix multiplication with add: addmm
    def cl_addmm(self: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, beta: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
        mm_res = cl_mm(mat1, mat2)
        if alpha != 1.0:
            mm_res = mm_res * alpha
        if beta != 0.0:
            return self * beta + mm_res
        return mm_res

    my_lib.impl("addmm", cl_addmm)

    # Batched matrix multiply: bmm
    def cl_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_shape = _get_shape(a)
        b_shape = _get_shape(b)
        B, M, K = a_shape[0], a_shape[1], a_shape[2]
        N = b_shape[2]
        out, out_buf = _allocate_tensor_output((B, M, N), a.dtype)
        engine = get_engine()
        for i in range(B):
            a_slice = a[i]
            b_slice = b[i]
            a_buf = _get_contiguous_buf(a_slice)
            b_buf = _get_contiguous_buf(b_slice)
            sub_out = out[i]
            sub_out_ptr = sub_out.data_ptr()
            sub_out_buf = _opencl_buffers.get(sub_out_ptr, out_buf)
            engine.run_matmul(a_buf, b_buf, sub_out_buf, M, N, K)
        return out

    my_lib.impl("bmm", cl_bmm)

    # Layer normalization
    def cl_native_layer_norm(input: torch.Tensor, normalized_shape: Sequence[int], weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, eps: float = 1e-5):
        engine = get_engine()
        in_shape = _get_shape(input)
        N = int(np.prod(normalized_shape))
        M = int(np.prod(in_shape)) // N

        out, out_buf = _allocate_tensor_output(in_shape, input.dtype)
        mean, mean_buf = _allocate_tensor_output((M,), input.dtype)
        rstd, rstd_buf = _allocate_tensor_output((M,), input.dtype)

        in_buf = _get_buf(input)
        w_buf = _get_buf(weight) if weight is not None else get_buffer_pool().allocate(N * 4)
        b_buf = _get_buf(bias) if bias is not None else get_buffer_pool().allocate(N * 4)

        engine.run_layer_norm(in_buf, w_buf, b_buf, out_buf, mean_buf, rstd_buf, M, N, eps)
        return out, mean, rstd

    my_lib.impl("native_layer_norm", cl_native_layer_norm)

    # Scaled dot-product attention
    def cl_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None):
        engine = get_engine()
        q_shape = _get_shape(query)
        B, H, M, D = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
        N = _get_shape(key)[2]
        if scale is None:
            scale = 1.0 / (D ** 0.5)

        out, out_buf = _allocate_tensor_output(q_shape, query.dtype)
        q_buf = _get_buf(query)
        k_buf = _get_buf(key)
        v_buf = _get_buf(value)

        engine.run_fused_attention(q_buf, k_buf, v_buf, out_buf, B, H, M, N, D, scale)
        return out

    my_lib.impl("scaled_dot_product_attention", cl_sdpa)

    # Embedding
    def cl_embedding(weight: torch.Tensor, indices: torch.Tensor, padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False):
        engine = get_engine()
        w_shape = _get_shape(weight)
        idx_shape = _get_shape(indices)
        num_indices = int(np.prod(idx_shape))
        dim = w_shape[1]
        out_shape = tuple(idx_shape) + (dim,)

        out, out_buf = _allocate_tensor_output(out_shape, weight.dtype)
        w_buf = _get_buf(weight)
        idx_buf = _get_buf(indices)

        engine.run_embedding(w_buf, idx_buf, out_buf, num_indices, dim)
        return out

    my_lib.impl("embedding", cl_embedding)

    # Transposition
    def cl_t(self: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(self)
        if len(shape) != 2:
            return self
        M, N = shape[0], shape[1]
        out, out_buf = _allocate_tensor_output((N, M), self.dtype)
        in_buf = _get_buf(self)
        engine.run_transpose(in_buf, out_buf, M, N)
        return out

    my_lib.impl("t", cl_t)
