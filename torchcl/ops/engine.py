"""
OpenCL Compute Engine — The central execution layer that bridges PyTorch
tensors to OpenCL kernels.

This module:
  1. Converts PyTorch tensors ↔ OpenCL buffers (via numpy)
  2. Launches compiled OpenCL kernels with correct workgroup sizes
  3. Returns results as PyTorch tensors

All operator modules (basic, matrix, activation, reduction) delegate here.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pyopencl as cl
import torch

from torchcl.runtime.context import get_queue, get_device_info, synchronize
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.kernels.registry import get_kernel_registry

# ── Dtype mapping ────────────────────────────────────────────────────
_TORCH_TO_NP = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

_NP_TO_TORCH = {v: k for k, v in _TORCH_TO_NP.items()}


class OpenCLEngine:
    """Central compute engine — converts tensors, launches kernels, returns results."""

    def __init__(self) -> None:
        self._pool = get_buffer_pool()
        self._registry = get_kernel_registry()

    # ── Tensor ↔ Buffer conversion ───────────────────────────────

    def tensor_to_buffer(self, tensor: torch.Tensor) -> CLBuffer:
        """Upload a PyTorch (CPU) tensor to an OpenCL buffer."""
        np_array = tensor.detach().cpu().numpy()
        return self._pool.host_to_device(np_array)

    def buffer_to_tensor(
        self,
        cl_buf: CLBuffer,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Download an OpenCL buffer back to a PyTorch CPU tensor."""
        np_dtype = _TORCH_TO_NP.get(dtype, np.float32)
        np_array = self._pool.device_to_host(cl_buf, dtype=np_dtype, shape=shape)
        return torch.from_numpy(np_array.copy())

    def allocate_output(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> CLBuffer:
        """Allocate an empty output buffer for a given tensor shape/dtype."""
        np_dtype = _TORCH_TO_NP.get(dtype, np.float32)
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        return self._pool.allocate(nbytes, np_dtype, shape)

    def free_buffer(self, cl_buf: CLBuffer) -> None:
        """Return a buffer to the pool."""
        self._pool.free(cl_buf)

    # ── Kernel launching ─────────────────────────────────────────

    def _compute_global_size(self, n: int, local_size: int = 256) -> int:
        """Round up n to the nearest multiple of local_size."""
        return ((n + local_size - 1) // local_size) * local_size

    def run_elementwise_binary(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        b_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run a binary element-wise kernel: out = op(a, b)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("elementwise.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               a_buf.buffer, b_buf.buffer, out_buf.buffer, np.int32(n))

    def run_elementwise_unary(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run a unary element-wise kernel: out = op(a)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("elementwise.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               a_buf.buffer, out_buf.buffer, np.int32(n))

    def run_elementwise_scalar(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        scalar: float,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run a scalar element-wise kernel: out = op(a, scalar)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("elementwise.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               a_buf.buffer, np.float32(scalar), out_buf.buffer, np.int32(n))

    def run_activation(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
        **extra_args,
    ) -> None:
        """Run an activation kernel."""
        queue = get_queue()
        kernel = self._registry.get_kernel("activation.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None

        args = [a_buf.buffer]
        for v in extra_args.values():
            args.append(np.float32(v))
        args.extend([out_buf.buffer, np.int32(n)])
        kernel(queue, global_size, local_size, *args)

    def run_matmul(
        self,
        a_buf: CLBuffer,
        b_buf: CLBuffer,
        out_buf: CLBuffer,
        M: int,
        N: int,
        K: int,
        use_tiled: bool = True,
    ) -> None:
        """Run matrix multiplication: C[M,N] = A[M,K] @ B[K,N]."""
        queue = get_queue()
        device_info = get_device_info()

        if use_tiled and M >= 16 and N >= 16 and K >= 16:
            tile_size = 16
            if device_info["local_mem_size_kb"] < 8:
                tile_size = 8  # Smaller tiles for limited local memory

            kernel = self._registry.get_kernel(
                "matmul.cl", "matmul_tiled_f32",
                build_options=f"-DTILE_SIZE={tile_size}"
            )
            global_size = (
                self._compute_global_size(M, tile_size),
                self._compute_global_size(N, tile_size),
            )
            local_size = (tile_size, tile_size)
        else:
            kernel = self._registry.get_kernel("matmul.cl", "matmul_naive_f32")
            global_size = (
                self._compute_global_size(M, 16),
                self._compute_global_size(N, 16),
            )
            local_size = None

        kernel(queue, global_size, local_size,
               a_buf.buffer, b_buf.buffer, out_buf.buffer,
               np.int32(M), np.int32(N), np.int32(K))

    def run_matmul_fp16(
        self,
        a_buf: CLBuffer,
        b_buf: CLBuffer,
        out_buf: CLBuffer,
        M: int,
        N: int,
        K: int,
        use_tiled: bool = True,
    ) -> None:
        """Run half-precision matrix multiplication: C[M,N] = A[M,K] @ B[K,N]."""
        queue = get_queue()
        device_info = get_device_info()

        if use_tiled and M >= 16 and N >= 16 and K >= 16:
            tile_size = 16
            if device_info["local_mem_size_kb"] < 8:
                tile_size = 8

            kernel = self._registry.get_kernel(
                "matmul.cl", "matmul_tiled_fp16",
                build_options=f"-DTILE_SIZE={tile_size}"
            )
            global_size = (
                self._compute_global_size(M, tile_size),
                self._compute_global_size(N, tile_size),
            )
            local_size = (tile_size, tile_size)
        else:
            kernel = self._registry.get_kernel("matmul.cl", "matmul_naive_fp16")
            global_size = (
                self._compute_global_size(M, 16),
                self._compute_global_size(N, 16),
            )
            local_size = None

        kernel(queue, global_size, local_size,
               a_buf.buffer, b_buf.buffer, out_buf.buffer,
               np.int32(M), np.int32(N), np.int32(K))

    def run_reduction(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run a reduction kernel (sum, max, min).

        Uses a two-pass approach: first reduce within workgroups,
        then reduce the workgroup results on CPU (simple and correct).
        """
        queue = get_queue()
        kernel = self._registry.get_kernel("reduction.cl", kernel_name)

        local_size = min(256, n)
        num_groups = (n + local_size - 1) // local_size
        global_size = num_groups * local_size

        # Allocate partial results buffer
        partial_buf = self._pool.allocate(num_groups * 4)  # float32 = 4 bytes

        kernel(queue, (global_size,), (local_size,),
               a_buf.buffer, partial_buf.buffer,
               cl.LocalMemory(local_size * 4),
               np.int32(n))

        # Read partial results and finish on CPU
        partials = self._pool.device_to_host(partial_buf, np.float32, (num_groups,))
        self._pool.free(partial_buf)

        if kernel_name == "sum_f32":
            result = np.array([partials.sum()], dtype=np.float32)
        elif kernel_name == "max_f32":
            result = np.array([partials.max()], dtype=np.float32)
        elif kernel_name == "min_f32":
            result = np.array([partials.min()], dtype=np.float32)
        else:
            result = np.array([partials.sum()], dtype=np.float32)

        self._pool.host_to_device(result, out_buf)

    def run_softmax(
        self,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        rows: int,
        cols: int,
    ) -> None:
        """Run row-wise softmax."""
        queue = get_queue()
        kernel = self._registry.get_kernel("reduction.cl", "softmax_f32")
        global_size = (self._compute_global_size(rows),)
        local_size = None
        kernel(queue, global_size, local_size,
               a_buf.buffer, out_buf.buffer,
               np.int32(rows), np.int32(cols))

    def run_fill(self, out_buf: CLBuffer, value: float, n: int) -> None:
        """Fill a buffer with a constant value."""
        queue = get_queue()
        kernel = self._registry.get_kernel("elementwise.cl", "fill_f32")
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               out_buf.buffer, np.float32(value), np.int32(n))

    def run_transpose(
        self,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        M: int,
        N: int,
    ) -> None:
        """Transpose a matrix: out[N,M] = a[M,N]^T."""
        queue = get_queue()
        kernel = self._registry.get_kernel("matmul.cl", "transpose_f32")
        global_size = (
            self._compute_global_size(M, 16),
            self._compute_global_size(N, 16),
        )
        local_size = None
        kernel(queue, global_size, local_size,
               a_buf.buffer, out_buf.buffer,
               np.int32(M), np.int32(N))

    def run_linear(
        self,
        a_buf: CLBuffer,
        b_buf: CLBuffer,
        bias_buf: CLBuffer | None,
        out_buf: CLBuffer,
        M: int,
        N: int,
        K: int,
        use_tiled: bool = True,
    ) -> None:
        """Run fused linear layer: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]."""
        queue = get_queue()
        device_info = get_device_info()
        has_bias = 1 if bias_buf is not None else 0
        bias_raw = bias_buf.buffer if bias_buf is not None else a_buf.buffer

        if use_tiled and M >= 16 and N >= 16 and K >= 16:
            tile_size = 16
            if device_info["local_mem_size_kb"] < 8:
                tile_size = 8

            kernel = self._registry.get_kernel(
                "matmul.cl", "linear_forward_tiled_f32",
                build_options=f"-DTILE_SIZE={tile_size}"
            )
            global_size = (
                self._compute_global_size(M, tile_size),
                self._compute_global_size(N, tile_size),
            )
            local_size = (tile_size, tile_size)
        else:
            kernel = self._registry.get_kernel("matmul.cl", "linear_forward_f32")
            global_size = (
                self._compute_global_size(M, 16),
                self._compute_global_size(N, 16),
            )
            local_size = None

        kernel(queue, global_size, local_size,
               a_buf.buffer, b_buf.buffer, bias_raw, out_buf.buffer,
               np.int32(M), np.int32(N), np.int32(K), np.int32(has_bias))

    # ── Activation backward ──────────────────────────────────────

    def run_activation_backward(
        self,
        kernel_name: str,
        grad_out_buf: CLBuffer,
        saved_buf: CLBuffer,
        grad_in_buf: CLBuffer,
        n: int,
        **extra_args,
    ) -> None:
        """Run an activation backward kernel.

        Args:
            kernel_name: e.g. 'relu_backward_f32'
            grad_out_buf: gradient from upstream
            saved_buf: saved input or output from forward
            grad_in_buf: output gradient buffer
            n: number of elements
            extra_args: e.g. neg_slope for leaky_relu
        """
        queue = get_queue()
        kernel = self._registry.get_kernel("activation.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None

        args = [grad_out_buf.buffer, saved_buf.buffer]
        for v in extra_args.values():
            args.append(np.float32(v))
        args.extend([grad_in_buf.buffer, np.int32(n)])
        kernel(queue, global_size, local_size, *args)

    # ── Convolution (im2col + matmul) ────────────────────────────

    def run_im2col(
        self,
        input_buf: CLBuffer,
        cols_buf: CLBuffer,
        C_in: int, H: int, W: int,
        kH: int, kW: int,
        stride_h: int, stride_w: int,
        pad_h: int, pad_w: int,
        H_out: int, W_out: int,
        batch_size: int,
    ) -> None:
        """Unfold image patches into column matrix."""
        queue = get_queue()
        kernel = self._registry.get_kernel("conv.cl", "im2col_f32")
        total = batch_size * H_out * W_out
        global_size = (self._compute_global_size(total),)
        local_size = (min(256, total),) if total >= 256 else None
        kernel(queue, global_size, local_size,
               input_buf.buffer, cols_buf.buffer,
               np.int32(C_in), np.int32(H), np.int32(W),
               np.int32(kH), np.int32(kW),
               np.int32(stride_h), np.int32(stride_w),
               np.int32(pad_h), np.int32(pad_w),
               np.int32(H_out), np.int32(W_out),
               np.int32(batch_size))

    def run_col2im(
        self,
        cols_buf: CLBuffer,
        grad_input_buf: CLBuffer,
        C_in: int, H: int, W: int,
        kH: int, kW: int,
        stride_h: int, stride_w: int,
        pad_h: int, pad_w: int,
        H_out: int, W_out: int,
        batch_size: int,
    ) -> None:
        """Fold column matrix back into image (backward)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("conv.cl", "col2im_f32")
        total = batch_size * H_out * W_out
        global_size = (self._compute_global_size(total),)
        local_size = (min(256, total),) if total >= 256 else None
        kernel(queue, global_size, local_size,
               cols_buf.buffer, grad_input_buf.buffer,
               np.int32(C_in), np.int32(H), np.int32(W),
               np.int32(kH), np.int32(kW),
               np.int32(stride_h), np.int32(stride_w),
               np.int32(pad_h), np.int32(pad_w),
               np.int32(H_out), np.int32(W_out),
               np.int32(batch_size))

    def run_conv2d_direct(
        self,
        input_buf: CLBuffer,
        weight_buf: CLBuffer,
        bias_buf: CLBuffer | None,
        out_buf: CLBuffer,
        batch_size: int,
        C_in: int,
        C_out: int,
        H: int,
        W: int,
        H_out: int,
        W_out: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
    ) -> None:
        """Run direct 3x3 convolution."""
        queue = get_queue()
        kernel = self._registry.get_kernel("conv.cl", "conv2d_direct_3x3_f32")
        
        # gid.0 = output position (H_out * W_out)
        # gid.1 = (batch_size * C_out)
        total_pos = H_out * W_out
        total_nc = batch_size * C_out
        
        global_size = (
            self._compute_global_size(total_pos),
            self._compute_global_size(total_nc)
        )
        local_size = None  # compiler-selected
        
        has_bias = 1 if bias_buf is not None else 0
        bias_raw = bias_buf.buffer if bias_buf is not None else input_buf.buffer
        
        kernel(queue, global_size, local_size,
               input_buf.buffer, weight_buf.buffer, bias_raw, out_buf.buffer,
               np.int32(batch_size), np.int32(C_in), np.int32(C_out),
               np.int32(H), np.int32(W), np.int32(H_out), np.int32(W_out),
               np.int32(stride_h), np.int32(stride_w),
               np.int32(pad_h), np.int32(pad_w),
               np.int32(has_bias))

    # ── Layer normalization ──────────────────────────────────────

    def run_layer_norm(
        self,
        input_buf: CLBuffer,
        weight_buf: CLBuffer,
        bias_buf: CLBuffer,
        output_buf: CLBuffer,
        mean_buf: CLBuffer,
        rstd_buf: CLBuffer,
        M: int, N: int,
        eps: float = 1e-5,
    ) -> None:
        """Run layer normalization forward."""
        queue = get_queue()
        kernel = self._registry.get_kernel("norm.cl", "layer_norm_f32")
        global_size = (self._compute_global_size(M),)
        local_size = (min(256, M),) if M >= 256 else None
        kernel(queue, global_size, local_size,
               input_buf.buffer, weight_buf.buffer, bias_buf.buffer,
               output_buf.buffer, mean_buf.buffer, rstd_buf.buffer,
               np.int32(M), np.int32(N), np.float32(eps))

    def run_layer_norm_backward(
        self,
        grad_out_buf: CLBuffer,
        input_buf: CLBuffer,
        weight_buf: CLBuffer,
        mean_buf: CLBuffer,
        rstd_buf: CLBuffer,
        grad_input_buf: CLBuffer,
        M: int, N: int,
    ) -> None:
        """Run layer normalization backward (grad_input only)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("norm.cl", "layer_norm_backward_f32")
        global_size = (self._compute_global_size(M),)
        local_size = (min(256, M),) if M >= 256 else None
        kernel(queue, global_size, local_size,
               grad_out_buf.buffer, input_buf.buffer, weight_buf.buffer,
               mean_buf.buffer, rstd_buf.buffer,
               grad_input_buf.buffer,
               np.int32(M), np.int32(N))

    def run_layer_norm_grad_weight_bias(
        self,
        grad_out_buf: CLBuffer,
        input_buf: CLBuffer,
        mean_buf: CLBuffer,
        rstd_buf: CLBuffer,
        grad_weight_buf: CLBuffer,
        grad_bias_buf: CLBuffer,
        M: int, N: int,
    ) -> None:
        """Compute grad_weight and grad_bias for layer norm."""
        self.run_fill(grad_weight_buf, 0.0, N)
        self.run_fill(grad_bias_buf, 0.0, N)
        queue = get_queue()
        kernel = self._registry.get_kernel("norm.cl", "layer_norm_grad_weight_bias_f32")
        global_size = (self._compute_global_size(N, 16), self._compute_global_size(M, 16))
        local_size = (16, 16)
        kernel(queue, global_size, local_size,
               grad_out_buf.buffer, input_buf.buffer,
               mean_buf.buffer, rstd_buf.buffer,
               grad_weight_buf.buffer, grad_bias_buf.buffer,
               np.int32(M), np.int32(N))

    # ── Batch normalization ──────────────────────────────────────

    def run_batch_norm(
        self,
        input_buf: CLBuffer,
        weight_buf: CLBuffer,
        bias_buf: CLBuffer,
        output_buf: CLBuffer,
        mean_buf: CLBuffer,
        var_buf: CLBuffer,
        batch_size: int, C: int, spatial: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        """Run batch normalization forward."""
        queue = get_queue()
        kernel = self._registry.get_kernel("norm.cl", "batch_norm_f32")
        global_size = (self._compute_global_size(C),)
        local_size = (min(256, C),) if C >= 256 else None
        kernel(queue, global_size, local_size,
               input_buf.buffer, weight_buf.buffer, bias_buf.buffer,
               output_buf.buffer, mean_buf.buffer, var_buf.buffer,
               np.int32(batch_size), np.int32(C), np.int32(spatial),
               np.float32(eps), np.float32(momentum))

    # ── Cross-entropy loss ───────────────────────────────────────

    def run_cross_entropy_forward(
        self,
        logits_buf: CLBuffer,
        targets_buf: CLBuffer,
        loss_buf: CLBuffer,
        log_softmax_buf: CLBuffer,
        batch_size: int, C: int,
    ) -> None:
        """Run cross-entropy forward."""
        queue = get_queue()
        kernel = self._registry.get_kernel("loss.cl", "cross_entropy_forward_f32")
        global_size = (self._compute_global_size(batch_size),)
        local_size = (min(256, batch_size),) if batch_size >= 256 else None
        kernel(queue, global_size, local_size,
               logits_buf.buffer, targets_buf.buffer,
               loss_buf.buffer, log_softmax_buf.buffer,
               np.int32(batch_size), np.int32(C))

    def run_cross_entropy_backward(
        self,
        log_softmax_buf: CLBuffer,
        targets_buf: CLBuffer,
        grad_logits_buf: CLBuffer,
        batch_size: int, C: int,
    ) -> None:
        """Run cross-entropy backward."""
        queue = get_queue()
        kernel = self._registry.get_kernel("loss.cl", "cross_entropy_backward_f32")
        global_size = (self._compute_global_size(batch_size),)
        local_size = (min(256, batch_size),) if batch_size >= 256 else None
        kernel(queue, global_size, local_size,
               log_softmax_buf.buffer, targets_buf.buffer,
               grad_logits_buf.buffer,
               np.int32(batch_size), np.int32(C),
               np.float32(1.0 / batch_size))

    # ── MSE loss ─────────────────────────────────────────────────

    def run_mse_forward(
        self,
        pred_buf: CLBuffer,
        target_buf: CLBuffer,
        loss_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run MSE forward (per-element squared error)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("loss.cl", "mse_forward_f32")
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               pred_buf.buffer, target_buf.buffer, loss_buf.buffer,
               np.int32(n))

    def run_mse_backward(
        self,
        pred_buf: CLBuffer,
        target_buf: CLBuffer,
        grad_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run MSE backward."""
        queue = get_queue()
        kernel = self._registry.get_kernel("loss.cl", "mse_backward_f32")
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               pred_buf.buffer, target_buf.buffer, grad_buf.buffer,
               np.int32(n))

    # ── Embedding ────────────────────────────────────────────────

    def run_embedding(
        self,
        weight_buf: CLBuffer,
        indices_buf: CLBuffer,
        output_buf: CLBuffer,
        N: int, D: int,
    ) -> None:
        """Run embedding lookup."""
        queue = get_queue()
        kernel = self._registry.get_kernel("embedding.cl", "embedding_lookup_f32")
        total = N * D
        global_size = (self._compute_global_size(total),)
        local_size = (min(256, total),) if total >= 256 else None
        kernel(queue, global_size, local_size,
               weight_buf.buffer, indices_buf.buffer, output_buf.buffer,
               np.int32(N), np.int32(D))

    def run_embedding_backward(
        self,
        grad_out_buf: CLBuffer,
        indices_buf: CLBuffer,
        grad_weight_buf: CLBuffer,
        N: int, D: int,
    ) -> None:
        """Run embedding backward (scatter-add)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("embedding.cl", "embedding_backward_f32")
        total = N * D
        global_size = (self._compute_global_size(total),)
        local_size = (min(256, total),) if total >= 256 else None
        kernel(queue, global_size, local_size,
               grad_out_buf.buffer, indices_buf.buffer, grad_weight_buf.buffer,
               np.int32(N), np.int32(D))

    # ── RMS normalization ────────────────────────────────────────

    def run_rms_norm(
        self,
        input_buf: CLBuffer,
        weight_buf: CLBuffer,
        output_buf: CLBuffer,
        rrms_buf: CLBuffer,
        M: int, N: int,
        eps: float = 1e-5,
    ) -> None:
        """Run RMS normalization forward."""
        queue = get_queue()
        kernel = self._registry.get_kernel("norm.cl", "rms_norm_f32")
        global_size = (self._compute_global_size(M),)
        local_size = (min(256, M),) if M >= 256 else None
        kernel(queue, global_size, local_size,
               input_buf.buffer, weight_buf.buffer,
               output_buf.buffer, rrms_buf.buffer,
               np.int32(M), np.int32(N), np.float32(eps))

    def run_fused_attention(
        self,
        Q: CLBuffer,
        K: CLBuffer,
        V: CLBuffer,
        Out: CLBuffer,
        B: int, H: int, M: int, N: int, D: int,
        scale: float,
        is_causal: bool = False,
    ) -> None:
        """Run the fused scaled dot-product attention kernel."""
        queue = get_queue()
        kernel_name = "flash_attention_causal_f32" if is_causal else "flash_attention_f32"
        kernel = self._registry.get_kernel("flash_attention.cl", kernel_name)
        global_size = (B * H * M * 256,)
        local_size = (256,)
        kernel(queue, global_size, local_size,
               Q.buffer, K.buffer, V.buffer, Out.buffer,
               np.int32(B), np.int32(H), np.int32(M), np.int32(N), np.int32(D),
               np.float32(scale))

    def run_rope(
        self,
        X_buf: CLBuffer,
        cos_buf: CLBuffer,
        sin_buf: CLBuffer,
        total_tokens: int,
        half_dim: int,
    ) -> None:
        """Apply in-place Rotary Positional Embeddings (RoPE)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("flash_attention.cl", "rope_f32")
        global_size = (
            self._compute_global_size(total_tokens, 16),
            self._compute_global_size(half_dim, 16),
        )
        local_size = None
        kernel(queue, global_size, local_size,
               X_buf.buffer, cos_buf.buffer, sin_buf.buffer,
               np.int32(total_tokens), np.int32(half_dim))

    def run_swiglu(
        self,
        gate_buf: CLBuffer,
        up_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run fused SwiGLU: out = SiLU(gate) * up."""
        queue = get_queue()
        kernel = self._registry.get_kernel("flash_attention.cl", "swiglu_f32")
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               gate_buf.buffer, up_buf.buffer, out_buf.buffer, np.int32(n))

    def run_sum_columns(
        self,
        input_buf: CLBuffer,
        out_buf: CLBuffer,
        rows: int,
        cols: int,
    ) -> None:
        """Sum input_buf[rows, cols] along columns to out_buf[cols]."""
        queue = get_queue()
        kernel = self._registry.get_kernel("reduction.cl", "sum_columns_f32")
        global_size = (self._compute_global_size(cols),)
        local_size = (min(256, cols),) if cols >= 256 else None
        kernel(queue, global_size, local_size,
               input_buf.buffer, out_buf.buffer,
               np.int32(rows), np.int32(cols))

    def run_reshape_grad_out(
        self,
        input_buf: CLBuffer,
        out_buf: CLBuffer,
        N: int, C: int, S: int,
    ) -> None:
        """Reshape/transpose grad_output [N, C, H_out * W_out] to [C, N * H_out * W_out]."""
        queue = get_queue()
        kernel = self._registry.get_kernel("conv.cl", "reshape_grad_out_f32")
        global_size = (
            self._compute_global_size(C, 16),
            self._compute_global_size(N * S, 16),
        )
        local_size = None
        kernel(queue, global_size, local_size,
               input_buf.buffer, out_buf.buffer,
               np.int32(N), np.int32(C), np.int32(S))

    def run_conv2d_bias_backward(
        self,
        grad_out_buf: CLBuffer,
        out_buf: CLBuffer,
        N: int, C: int, H: int, W: int,
    ) -> None:
        """Sum grad_out_buf[N, C, H, W] along axes (0, 2, 3) to out_buf[C]."""
        queue = get_queue()
        kernel = self._registry.get_kernel("reduction.cl", "sum_conv2d_bias_f32")
        global_size = (self._compute_global_size(C),)
        local_size = (min(256, C),) if C >= 256 else None
        kernel(queue, global_size, local_size,
               grad_out_buf.buffer, out_buf.buffer,
               np.int32(N), np.int32(C), np.int32(H), np.int32(W))


# ── Module-level singleton ───────────────────────────────────────────
_global_engine: OpenCLEngine | None = None


def get_engine() -> OpenCLEngine:
    """Return the global compute engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = OpenCLEngine()
    return _global_engine
