"""
ojasOpt — In-Kernel Fused Thermodynamic Optimizers for PyTorch & OjasX.
Replaces Apex / DeepSpeed FusedAdam with single-pass OpenCL GPU execution.
"""

from __future__ import annotations
import math
from typing import Iterable, Callable, Optional
import numpy as np
import torch
from torch.optim import Optimizer

import torchcl
from torchcl.api import to_opencl, is_opencl_tensor, _get_buf
from torchcl.ops.engine import get_engine
from torchcl.runtime.context import get_queue


class AdamW(Optimizer):
    """In-Kernel Fused AdamW optimizer executing in a single GPU memory pass."""
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._engine = get_engine()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        queue = get_queue()
        kernel = self._engine._registry.get_kernel("optimizer.cl", "fused_adamw_f32")

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if not is_opencl_tensor(p):
                    p_cl = to_opencl(p)
                else:
                    p_cl = p
                if not is_opencl_tensor(grad):
                    grad_cl = to_opencl(grad)
                else:
                    grad_cl = grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Allocate momentum buffers on GPU
                    state["exp_avg"] = torchcl.zeros_like(p_cl)
                    state["exp_avg_sq"] = torchcl.zeros_like(p_cl)

                state["step"] += 1
                step = state["step"]
                bc1 = 1.0 - (beta1 ** step)
                bc2 = 1.0 - (beta2 ** step)

                numel = p.numel()
                global_size = (self._engine._compute_global_size(numel),)
                local_size = (min(256, numel),) if numel >= 256 else None

                p_buf = _get_buf(p_cl)
                grad_buf = _get_buf(grad_cl)
                m_buf = _get_buf(state["exp_avg"])
                v_buf = _get_buf(state["exp_avg_sq"])

                kernel(
                    queue, global_size, local_size,
                    p_buf.buffer, grad_buf.buffer, m_buf.buffer, v_buf.buffer,
                    np.float32(lr), np.float32(beta1), np.float32(beta2),
                    np.float32(eps), np.float32(weight_decay),
                    np.float32(bc1), np.float32(bc2),
                    np.int32(numel)
                )

                # Sync back to PyTorch tensor storage if needed
                if not is_opencl_tensor(p):
                    p.copy_(torchcl.to_cpu(p_cl))

        return loss


class Lion(Optimizer):
    """In-Kernel Fused Lion (Sign Momentum) Optimizer."""
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-2,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._engine = get_engine()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        queue = get_queue()
        kernel = self._engine._registry.get_kernel("optimizer.cl", "fused_lion_f32")

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                p_cl = to_opencl(p) if not is_opencl_tensor(p) else p
                grad_cl = to_opencl(grad) if not is_opencl_tensor(grad) else grad

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torchcl.zeros_like(p_cl)

                numel = p.numel()
                global_size = (self._engine._compute_global_size(numel),)
                local_size = (min(256, numel),) if numel >= 256 else None

                p_buf = _get_buf(p_cl)
                grad_buf = _get_buf(grad_cl)
                m_buf = _get_buf(state["exp_avg"])

                kernel(
                    queue, global_size, local_size,
                    p_buf.buffer, grad_buf.buffer, m_buf.buffer,
                    np.float32(lr), np.float32(beta1), np.float32(beta2),
                    np.float32(weight_decay),
                    np.int32(numel)
                )

                if not is_opencl_tensor(p):
                    p.copy_(torchcl.to_cpu(p_cl))

        return loss


class SGD(Optimizer):
    """In-Kernel Fused SGD with Momentum & Decoupled Weight Decay."""
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._engine = get_engine()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        queue = get_queue()
        kernel = self._engine._registry.get_kernel("optimizer.cl", "fused_sgd_momentum_f32")

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                p_cl = to_opencl(p) if not is_opencl_tensor(p) else p
                grad_cl = to_opencl(grad) if not is_opencl_tensor(grad) else grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buf"] = torchcl.zeros_like(p_cl)

                numel = p.numel()
                global_size = (self._engine._compute_global_size(numel),)
                local_size = (min(256, numel),) if numel >= 256 else None

                p_buf = _get_buf(p_cl)
                grad_buf = _get_buf(grad_cl)
                m_buf = _get_buf(state["momentum_buf"])

                kernel(
                    queue, global_size, local_size,
                    p_buf.buffer, grad_buf.buffer, m_buf.buffer,
                    np.float32(lr), np.float32(momentum), np.float32(weight_decay),
                    np.int32(numel)
                )

                if not is_opencl_tensor(p):
                    p.copy_(torchcl.to_cpu(p_cl))

        return loss
