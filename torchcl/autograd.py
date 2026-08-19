"""
OjasX Autograd — PyTorch autograd.Function implementations for OpenCL ops.

Each class subclasses torch.autograd.Function and wires:
  - forward:  runs the OpenCL kernel, saves tensors for backward
  - backward: runs the backward OpenCL kernel using saved tensors

This enables loss.backward() to flow gradients through OpenCL operations.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch

from torchcl.ops.engine import get_engine
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.api import (
    _get_buf,
    _get_shape,
    _wrap_output,
    is_opencl_tensor,
    to_opencl,
    to_cpu,
)


# ── Helper to allocate and get buffer ────────────────────────────────

def _alloc(shape: tuple, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, CLBuffer]:
    """Allocate an OpenCL buffer and return (handle, cl_buf)."""
    engine = get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    handle = _wrap_output(cl_buf, shape, dtype)
    return handle, cl_buf


# ═══════════════════════════════════════════════════════════════════════
# Activation Functions
# ═══════════════════════════════════════════════════════════════════════

class ReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_handle: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(input_handle)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_activation("relu_f32", _get_buf(input_handle), out_buf, n)
        ctx.save_for_backward(input_handle)
        ctx._n = n
        ctx._shape = shape
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input_handle,) = ctx.saved_tensors
        engine = get_engine()
        grad_in_handle, grad_in_buf = _alloc(ctx._shape)
        engine.run_activation_backward(
            "relu_backward_f32",
            _get_buf(grad_output), _get_buf(input_handle),
            grad_in_buf, ctx._n,
        )
        return grad_in_handle


class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_handle: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(input_handle)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_activation("sigmoid_f32", _get_buf(input_handle), out_buf, n)
        ctx.save_for_backward(out_handle)  # save OUTPUT for sigmoid backward
        ctx._n = n
        ctx._shape = shape
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (output_handle,) = ctx.saved_tensors
        engine = get_engine()
        grad_in_handle, grad_in_buf = _alloc(ctx._shape)
        engine.run_activation_backward(
            "sigmoid_backward_f32",
            _get_buf(grad_output), _get_buf(output_handle),
            grad_in_buf, ctx._n,
        )
        return grad_in_handle


class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_handle: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(input_handle)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_activation("tanh_f32", _get_buf(input_handle), out_buf, n)
        ctx.save_for_backward(out_handle)  # save OUTPUT for tanh backward
        ctx._n = n
        ctx._shape = shape
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (output_handle,) = ctx.saved_tensors
        engine = get_engine()
        grad_in_handle, grad_in_buf = _alloc(ctx._shape)
        engine.run_activation_backward(
            "tanh_backward_f32",
            _get_buf(grad_output), _get_buf(output_handle),
            grad_in_buf, ctx._n,
        )
        return grad_in_handle


class GeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_handle: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(input_handle)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_activation("gelu_f32", _get_buf(input_handle), out_buf, n)
        ctx.save_for_backward(input_handle)  # save INPUT for gelu backward
        ctx._n = n
        ctx._shape = shape
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input_handle,) = ctx.saved_tensors
        engine = get_engine()
        grad_in_handle, grad_in_buf = _alloc(ctx._shape)
        engine.run_activation_backward(
            "gelu_backward_f32",
            _get_buf(grad_output), _get_buf(input_handle),
            grad_in_buf, ctx._n,
        )
        return grad_in_handle


class SiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_handle: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(input_handle)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_activation("silu_f32", _get_buf(input_handle), out_buf, n)
        ctx.save_for_backward(input_handle)  # save INPUT for silu backward
        ctx._n = n
        ctx._shape = shape
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input_handle,) = ctx.saved_tensors
        engine = get_engine()
        grad_in_handle, grad_in_buf = _alloc(ctx._shape)
        engine.run_activation_backward(
            "silu_backward_f32",
            _get_buf(grad_output), _get_buf(input_handle),
            grad_in_buf, ctx._n,
        )
        return grad_in_handle


class LeakyReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_handle: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(input_handle)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_activation("leaky_relu_f32", _get_buf(input_handle), out_buf, n, neg_slope=negative_slope)
        ctx.save_for_backward(input_handle)
        ctx._n = n
        ctx._shape = shape
        ctx._negative_slope = negative_slope
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (input_handle,) = ctx.saved_tensors
        engine = get_engine()
        grad_in_handle, grad_in_buf = _alloc(ctx._shape)
        engine.run_activation_backward(
            "leaky_relu_backward_f32",
            _get_buf(grad_output), _get_buf(input_handle),
            grad_in_buf, ctx._n,
            neg_slope=ctx._negative_slope,
        )
        return grad_in_handle, None  # None for negative_slope grad


# ═══════════════════════════════════════════════════════════════════════
# Arithmetic Operations
# ═══════════════════════════════════════════════════════════════════════

class AddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(a)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_elementwise_binary("add_f32", _get_buf(a), _get_buf(b), out_buf, n)
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # d(a+b)/da = 1, d(a+b)/db = 1 → grad flows through unchanged
        return grad_output, grad_output


class SubFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(a)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_elementwise_binary("sub_f32", _get_buf(a), _get_buf(b), out_buf, n)
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # d(a-b)/da = 1, d(a-b)/db = -1
        engine = get_engine()
        shape = _get_shape(grad_output)
        n = int(np.prod(shape))
        neg_grad_handle, neg_grad_buf = _alloc(shape)
        engine.run_elementwise_unary("neg_f32", _get_buf(grad_output), neg_grad_buf, n)
        return grad_output, neg_grad_handle


class MulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(a)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_elementwise_binary("mul_f32", _get_buf(a), _get_buf(b), out_buf, n)
        ctx.save_for_backward(a, b)
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        engine = get_engine()
        shape = _get_shape(grad_output)
        n = int(np.prod(shape))

        # grad_a = grad_out * b
        grad_a_handle, grad_a_buf = _alloc(shape)
        engine.run_elementwise_binary("mul_f32", _get_buf(grad_output), _get_buf(b), grad_a_buf, n)

        # grad_b = grad_out * a
        grad_b_handle, grad_b_buf = _alloc(shape)
        engine.run_elementwise_binary("mul_f32", _get_buf(grad_output), _get_buf(a), grad_b_buf, n)

        return grad_a_handle, grad_b_handle


# ═══════════════════════════════════════════════════════════════════════
# Matrix Operations
# ═══════════════════════════════════════════════════════════════════════

class MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        a_shape = _get_shape(a)
        b_shape = _get_shape(b)
        M, K = a_shape
        K2, N = b_shape
        assert K == K2, f"matmul dimension mismatch: {a_shape} @ {b_shape}"

        out_shape = (M, N)
        out_handle, out_buf = _alloc(out_shape)
        engine.run_matmul(_get_buf(a), _get_buf(b), out_buf, M, N, K)
        ctx.save_for_backward(a, b)
        ctx._M = M
        ctx._N = N
        ctx._K = K
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        engine = get_engine()
        M, N, K = ctx._M, ctx._N, ctx._K

        # grad_A = grad_output @ B^T   → shape [M, K]
        b_t_handle, b_t_buf = _alloc((N, K))
        engine.run_transpose(_get_buf(b), b_t_buf, K, N)
        grad_a_handle, grad_a_buf = _alloc((M, K))
        engine.run_matmul(_get_buf(grad_output), b_t_buf, grad_a_buf, M, K, N)

        # grad_B = A^T @ grad_output   → shape [K, N]
        a_t_handle, a_t_buf = _alloc((K, M))
        engine.run_transpose(_get_buf(a), a_t_buf, M, K)
        grad_b_handle, grad_b_buf = _alloc((K, N))
        engine.run_matmul(a_t_buf, _get_buf(grad_output), grad_b_buf, K, N, M)

        return grad_a_handle, grad_b_handle


class DivFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(a)
        n = int(np.prod(shape))
        out_handle, out_buf = _alloc(shape)
        engine.run_elementwise_binary("div_f32", _get_buf(a), _get_buf(b), out_buf, n)
        ctx.save_for_backward(a, b)
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        engine = get_engine()
        shape = _get_shape(grad_output)
        n = int(np.prod(shape))
        
        # grad_a = grad_output / b
        grad_a_handle, grad_a_buf = _alloc(shape)
        engine.run_elementwise_binary("div_f32", _get_buf(grad_output), _get_buf(b), grad_a_buf, n)
        
        # grad_b = -grad_output * a / (b * b)
        b_sq_handle, b_sq_buf = _alloc(shape)
        engine.run_elementwise_binary("mul_f32", _get_buf(b), _get_buf(b), b_sq_buf, n)
        
        neg_a_handle, neg_a_buf = _alloc(shape)
        engine.run_elementwise_unary("neg_f32", _get_buf(a), neg_a_buf, n)
        
        factor_handle, factor_buf = _alloc(shape)
        engine.run_elementwise_binary("div_f32", _get_buf(neg_a_handle), _get_buf(b_sq_handle), factor_buf, n)
        
        grad_b_handle, grad_b_buf = _alloc(shape)
        engine.run_elementwise_binary("mul_f32", _get_buf(grad_output), _get_buf(factor_handle), grad_b_buf, n)
        
        return grad_a_handle, grad_b_handle


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        engine = get_engine()
        if not is_opencl_tensor(x): x = to_opencl(x)
        if not is_opencl_tensor(weight): weight = to_opencl(weight)
        if bias is not None and not is_opencl_tensor(bias): bias = to_opencl(bias)

        x_shape = _get_shape(x)
        w_shape = _get_shape(weight)
        
        K = int(x_shape[-1])
        M = int(math.prod(x_shape[:-1])) if len(x_shape) > 1 else 1
        N, K2 = int(w_shape[0]), int(w_shape[1])
        assert K == K2, f"Linear dimension mismatch: input {x_shape}, weight {w_shape}"
        
        out_shape = tuple(x_shape[:-1]) + (N,)
        out_handle, out_buf = _alloc(out_shape)
        
        x_buf = _get_buf(x)
        w_buf = _get_buf(weight)
        bias_buf = _get_buf(bias) if bias is not None else None
        
        engine.run_linear(x_buf, w_buf, bias_buf, out_buf, M, N, K)
        
        ctx.save_for_backward(x, weight, bias)
        ctx._x_shape = x_shape
        ctx._M = M
        ctx._N = N
        ctx._K = K
        ctx._has_bias = bias is not None
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x, weight, bias = ctx.saved_tensors
        engine = get_engine()
        M, N, K = ctx._M, ctx._N, ctx._K
        x_shape = ctx._x_shape
        
        # grad_x = grad_output @ weight
        grad_x_handle, grad_x_buf = _alloc(x_shape)
        engine.run_matmul(_get_buf(grad_output), _get_buf(weight), grad_x_buf, M, K, N)
        
        # grad_weight = grad_output.T @ x
        grad_out_t_handle, grad_out_t_buf = _alloc((N, M))
        engine.run_transpose(_get_buf(grad_output), grad_out_t_buf, M, N)
        
        grad_w_handle, grad_w_buf = _alloc((N, K))
        engine.run_matmul(grad_out_t_buf, _get_buf(x), grad_w_buf, N, K, M)
        
        # grad_bias = grad_output.sum(dim=0)
        grad_bias_handle = None
        if ctx._has_bias:
            grad_bias_handle, grad_bias_buf = _alloc((N,))
            engine.run_sum_columns(_get_buf(grad_output), grad_bias_buf, M, N)
            
        return grad_x_handle, grad_w_handle, grad_bias_handle


# ═══════════════════════════════════════════════════════════════════════
# Layer Normalization
# ═══════════════════════════════════════════════════════════════════════

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_handle: torch.Tensor,
        weight_handle: torch.Tensor,
        bias_handle: torch.Tensor,
        normalized_shape: int,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        engine = get_engine()
        shape = _get_shape(input_handle)
        N = normalized_shape
        M = int(np.prod(shape)) // N

        out_handle, out_buf = _alloc(shape)
        mean_handle, mean_buf = _alloc((M,))
        rstd_handle, rstd_buf = _alloc((M,))

        engine.run_layer_norm(
            _get_buf(input_handle), _get_buf(weight_handle), _get_buf(bias_handle),
            out_buf, mean_buf, rstd_buf,
            M, N, eps,
        )

        ctx.save_for_backward(input_handle, weight_handle, mean_handle, rstd_handle)
        ctx._M = M
        ctx._N = N
        ctx._shape = shape
        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_handle, weight_handle, mean_handle, rstd_handle = ctx.saved_tensors
        engine = get_engine()
        M, N = ctx._M, ctx._N

        # grad_input
        grad_in_handle, grad_in_buf = _alloc(ctx._shape)
        engine.run_layer_norm_backward(
            _get_buf(grad_output), _get_buf(input_handle), _get_buf(weight_handle),
            _get_buf(mean_handle), _get_buf(rstd_handle),
            grad_in_buf, M, N,
        )

        # grad_weight, grad_bias
        grad_w_handle, grad_w_buf = _alloc((N,))
        grad_b_handle, grad_b_buf = _alloc((N,))
        engine.run_layer_norm_grad_weight_bias(
            _get_buf(grad_output), _get_buf(input_handle),
            _get_buf(mean_handle), _get_buf(rstd_handle),
            grad_w_buf, grad_b_buf, M, N,
        )

        return grad_in_handle, grad_w_handle, grad_b_handle, None, None


# ═══════════════════════════════════════════════════════════════════════
# Cross-Entropy Loss
# ═══════════════════════════════════════════════════════════════════════

class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        pool = get_buffer_pool()

        logits_shape = _get_shape(logits)
        batch_size = logits_shape[0]
        C = logits_shape[1]

        # Per-sample losses
        loss_per_sample_handle, loss_buf = _alloc((batch_size,))
        log_softmax_handle, log_softmax_buf = _alloc(logits_shape)

        engine.run_cross_entropy_forward(
            _get_buf(logits), _get_buf(targets),
            loss_buf, log_softmax_buf,
            batch_size, C,
        )

        # Reduce to mean loss (on GPU via sum + scalar multiply)
        sum_buf = engine.allocate_output((1,))
        engine.run_reduction("sum_f32", loss_buf, sum_buf, batch_size)

        mean_buf = engine.allocate_output((1,))
        engine.run_elementwise_scalar(
            "mul_scalar_f32", sum_buf, 1.0 / batch_size, mean_buf, 1,
        )
        engine.free_buffer(sum_buf)

        mean_loss_handle = _wrap_output(mean_buf, (1,))

        ctx.save_for_backward(log_softmax_handle, targets)
        ctx._batch_size = batch_size
        ctx._C = C
        ctx._logits_shape = logits_shape
        return mean_loss_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        log_softmax_handle, targets = ctx.saved_tensors
        engine = get_engine()

        grad_logits_handle, grad_logits_buf = _alloc(ctx._logits_shape)
        engine.run_cross_entropy_backward(
            _get_buf(log_softmax_handle), _get_buf(targets),
            grad_logits_buf,
            ctx._batch_size, ctx._C,
        )
        return grad_logits_handle, None


class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_handle: torch.Tensor,
        weight_handle: torch.Tensor,
        bias_handle: Optional[torch.Tensor],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> torch.Tensor:
        engine = get_engine()
        pool = get_buffer_pool()

        x_shape = _get_shape(input_handle)
        w_shape = _get_shape(weight_handle)
        
        N, C_in, H, W = x_shape
        C_out, C_in_w, kH, kW = w_shape
        assert C_in == C_in_w, f"Conv2d channel mismatch: input {C_in}, weight {C_in_w}"

        stride_h, stride_w = stride
        pad_h, pad_w = padding

        H_out = (H - kH + 2 * pad_h) // stride_h + 1
        W_out = (W - kW + 2 * pad_w) // stride_w + 1

        out_shape = (N, C_out, H_out, W_out)
        out_handle, out_buf = _alloc(out_shape)

        if kH == 3 and kW == 3:
            bias_buf = _get_buf(bias_handle) if bias_handle is not None else None
            engine.run_conv2d_direct(
                _get_buf(input_handle), _get_buf(weight_handle), bias_buf, out_buf,
                N, C_in, C_out, H, W, H_out, W_out,
                stride_h, stride_w, pad_h, pad_w
            )
            ctx.save_for_backward(input_handle, weight_handle, bias_handle)
            ctx._params = (stride, padding, out_shape, x_shape, w_shape)
            ctx._strategy = "direct"
        else:
            cols_rows = N * H_out * W_out
            cols_cols = C_in * kH * kW
            cols_buf = pool.allocate(cols_rows * cols_cols * 4, np.dtype(np.float32), (cols_rows, cols_cols))
            
            engine.run_im2col(
                _get_buf(input_handle), cols_buf,
                C_in, H, W, kH, kW,
                stride_h, stride_w, pad_h, pad_w,
                H_out, W_out, N
            )

            w_t_buf = pool.allocate(cols_cols * C_out * 4, np.dtype(np.float32), (cols_cols, C_out))
            engine.run_transpose(_get_buf(weight_handle), w_t_buf, C_out, cols_cols)

            matmul_out_buf = pool.allocate(cols_rows * C_out * 4, np.dtype(np.float32), (cols_rows, C_out))
            engine.run_matmul(cols_buf, w_t_buf, matmul_out_buf, cols_rows, C_out, cols_cols)

            # Read matmul result in NHWC format
            matmul_out_np = pool.device_to_host(matmul_out_buf, np.float32, (N, H_out, W_out, C_out))
            # Transpose to NCHW format
            output_np = matmul_out_np.transpose(0, 3, 1, 2).copy()
            # Copy to out_buf
            pool.host_to_device(output_np, out_buf)
            
            pool.free(w_t_buf)
            pool.free(matmul_out_buf)
            
            if bias_handle is not None:
                bias_np = to_cpu(bias_handle).numpy()
                tiled_bias = np.tile(bias_np.reshape(1, C_out, 1, 1), (N, 1, H_out, W_out)).astype(np.float32)
                tiled_buf = pool.host_to_device(tiled_bias)
                engine.run_elementwise_binary("add_f32", out_buf, tiled_buf, out_buf, N * C_out * H_out * W_out)
                pool.free(tiled_buf)

            ctx.save_for_backward(input_handle, weight_handle, bias_handle)
            ctx._cols_buf = cols_buf
            ctx._params = (stride, padding, out_shape, x_shape, w_shape)
            ctx._strategy = "im2col"

        return out_handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None]:
        input_handle, weight_handle, bias_handle = ctx.saved_tensors
        engine = get_engine()
        pool = get_buffer_pool()
        
        stride, padding, out_shape, x_shape, w_shape = ctx._params
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        
        N, C_in, H, W = x_shape
        C_out, C_in_w, kH, kW = w_shape
        H_out = out_shape[2]
        W_out = out_shape[3]
        S = H_out * W_out
        
        # 1. Compute grad_bias entirely on GPU
        grad_bias_handle = None
        if bias_handle is not None:
            grad_bias_handle, grad_bias_buf = _alloc((C_out,))
            engine.run_conv2d_bias_backward(_get_buf(grad_output), grad_bias_buf, N, C_out, H_out, W_out)

        # 2. Get/Compute im2col column buffer
        if ctx._strategy == "direct" or getattr(ctx, "_cols_buf", None) is None:
            cols_rows = N * H_out * W_out
            cols_cols = C_in * kH * kW
            cols_buf = pool.allocate(cols_rows * cols_cols * 4, np.dtype(np.float32), (cols_rows, cols_cols))
            engine.run_im2col(
                _get_buf(input_handle), cols_buf,
                C_in, H, W, kH, kW,
                stride_h, stride_w, pad_h, pad_w,
                H_out, W_out, N
            )
        else:
            cols_buf = ctx._cols_buf

        # 3. Reshape/Transpose grad_output on GPU: N, C_out, H_out * W_out -> C_out, N * H_out * W_out
        grad_out_flat_buf = pool.allocate(C_out * N * S * 4, np.dtype(np.float32), (C_out, N * S))
        engine.run_reshape_grad_out(_get_buf(grad_output), grad_out_flat_buf, N, C_out, S)

        # 4. Compute grad_weight entirely on GPU: grad_out_flat @ cols_buf
        grad_weight_handle, grad_w_buf = _alloc(w_shape)
        engine.run_matmul(grad_out_flat_buf, cols_buf, grad_w_buf, C_out, C_in * kH * kW, N * S)

        # 5. Compute grad_cols entirely on GPU: grad_out_flat.T @ weight
        grad_out_flat_t_buf = pool.allocate(N * S * C_out * 4, np.dtype(np.float32), (N * S, C_out))
        engine.run_transpose(grad_out_flat_buf, grad_out_flat_t_buf, C_out, N * S)

        grad_cols_buf = pool.allocate(N * S * C_in * kH * kW * 4, np.dtype(np.float32), (N * S, C_in * kH * kW))
        engine.run_matmul(grad_out_flat_t_buf, _get_buf(weight_handle), grad_cols_buf, N * S, C_in * kH * kW, C_out)

        # 6. Accumulate to input grad using col2im on GPU
        grad_in_handle, grad_in_buf = _alloc(x_shape)
        pool.zero_fill(grad_in_buf)
        
        engine.run_col2im(
            grad_cols_buf, grad_in_buf,
            C_in, H, W, kH, kW,
            stride_h, stride_w, pad_h, pad_w,
            H_out, W_out, N
        )

        # 7. Clean up temporary buffers
        pool.free(grad_out_flat_buf)
        pool.free(grad_out_flat_t_buf)
        pool.free(grad_cols_buf)
        if ctx._strategy != "direct":
            pool.free(cols_buf)
        else:
            pool.free(cols_buf)

        return grad_in_handle, grad_weight_handle, grad_bias_handle, None, None
