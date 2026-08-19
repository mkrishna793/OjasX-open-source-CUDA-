"""
Test Suite for ojasDynamical: Continuous-Time Liquid Neural ODE Solvers.
"""

import pytest
import math
import torch
import torchcl
import torchcl.dynamical as dynamical


def test_ojas_dynamical_exponential_decay():
    # Analytical ODE: dy/dt = -0.5 * y => y(t) = y(0) * exp(-0.5 * t)
    y0 = torch.tensor([10.0, 20.0])
    t_span = [0.0, 1.0, 2.0]

    def linear_decay(t: float, y: torch.Tensor) -> torch.Tensor:
        return -0.5 * y

    traj = dynamical.odeint_rk4(linear_decay, y0, t_span, dt=0.01)
    traj_cpu = torchcl.to_cpu(traj)

    # Expected at t = 2.0: y0 * exp(-1.0)
    expected_t2 = y0 * math.exp(-1.0)
    diff = (traj_cpu[-1] - expected_t2).abs().max().item()

    assert diff < 1e-3, f"RK4 integration error too high vs analytical solution: {diff}"


def test_ojas_dynamical_liquid_neural_ode():
    torch.manual_seed(42)
    model = dynamical.LiquidNeuralODE(hidden_dim=16, tau=1.0)
    h0 = torch.randn(4, 16)

    traj = model(h0, t_span=[0.0, 0.5, 1.0])
    traj_cpu = torchcl.to_cpu(traj)

    # Shape: [3 time points, batch=4, dim=16]
    assert traj_cpu.shape == (3, 4, 16)
    assert not torch.isnan(traj_cpu).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
