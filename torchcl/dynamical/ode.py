"""
ojasDynamical — Continuous-Time Liquid Neural ODE Solvers for OjasX.
Replaces discrete layer steps with continuous dynamical trajectory integration on OpenCL GPUs.
"""

from __future__ import annotations
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor, _get_buf, _wrap_output
from torchcl.ops.engine import get_engine
from torchcl.runtime.context import get_queue


def rk4_gpu_step(
    y: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
    k4: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Execute vector Runge-Kutta 4th order update directly on OpenCL GPU."""
    if not is_opencl_tensor(y): y = to_opencl(y)
    if not is_opencl_tensor(k1): k1 = to_opencl(k1)
    if not is_opencl_tensor(k2): k2 = to_opencl(k2)
    if not is_opencl_tensor(k3): k3 = to_opencl(k3)
    if not is_opencl_tensor(k4): k4 = to_opencl(k4)

    engine = get_engine()
    numel = y.numel()
    out_buf = engine.allocate_output(tuple(y.shape), y.dtype)

    kernel = engine._registry.get_kernel("liquid_ode.cl", "rk4_step_f32")
    queue = get_queue()

    global_size = (engine._compute_global_size(numel),)
    local_size = (min(256, numel),) if numel >= 256 else None

    kernel(
        queue, global_size, local_size,
        _get_buf(y).buffer,
        _get_buf(k1).buffer,
        _get_buf(k2).buffer,
        _get_buf(k3).buffer,
        _get_buf(k4).buffer,
        out_buf.buffer,
        np.float32(h),
        np.int32(numel)
    )

    return _wrap_output(out_buf, tuple(y.shape), y.dtype)


def odeint_rk4(
    func: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t: torch.Tensor | list[float],
    dt: float = 0.05,
) -> torch.Tensor:
    """
    Numerically integrate dy/dt = func(t, y) using GPU-accelerated RK4 solver.
    Returns: trajectory tensor of shape [len(t), *y0.shape]
    """
    if not is_opencl_tensor(y0):
        y0 = to_opencl(y0)

    t_points = t if isinstance(t, list) else t.tolist()
    trajectory = [y0]

    curr_y = y0
    for i in range(len(t_points) - 1):
        t_start = t_points[i]
        t_end = t_points[i + 1]
        t_curr = t_start

        while t_curr < t_end:
            step_h = min(dt, t_end - t_curr)
            if step_h <= 1e-7:
                break

            # 4 evaluations of vector field
            k1 = func(t_curr, curr_y)
            k2 = func(t_curr + 0.5 * step_h, curr_y + 0.5 * step_h * k1)
            k3 = func(t_curr + 0.5 * step_h, curr_y + 0.5 * step_h * k2)
            k4 = func(t_curr + step_h, curr_y + step_h * k3)

            # GPU parallel step
            curr_y = rk4_gpu_step(curr_y, k1, k2, k3, k4, step_h)
            t_curr += step_h

        trajectory.append(curr_y)

    # Stack along time dimension
    traj_cpus = [torchcl.to_cpu(s) for s in trajectory]
    return torchcl.to_opencl(torch.stack(traj_cpus, dim=0))


class LiquidNeuralODE(nn.Module):
    """Continuous-Time Liquid Neural Network ODE Layer."""
    def __init__(self, hidden_dim: int = 64, tau: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def vector_field(self, t: float, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt = -h/tau + net(h)."""
        decay = -h / self.tau
        excitation = self.net(h)
        return decay + excitation

    def forward(self, h0: torch.Tensor, t_span: list[float] | None = None) -> torch.Tensor:
        """Integrate continuous state trajectory over t_span."""
        if t_span is None:
            t_span = [0.0, 0.5, 1.0]
        return odeint_rk4(self.vector_field, h0, t_span, dt=0.1)


__all__ = ["rk4_gpu_step", "odeint_rk4", "LiquidNeuralODE"]
