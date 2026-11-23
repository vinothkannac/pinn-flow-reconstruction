"""Checks my autograd residuals on a field where I know the answer (a Taylor-Green field):
    u = -cos(kx) sin(ky),  v = sin(kx) cos(ky),  p = -(cos(2kx) + cos(2ky)) / 4
It has no divergence, and with nu = 0 the convection and pressure terms cancel exactly.
The Laplacian of u is -2 k^2 u, so with viscosity the momentum residual should be 2 nu k^2 (u, v).
I did this first because if the residual is wrong the whole PINN is wrong and I would not notice.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from pinn.config import Config
from pinn.model import FlowNet
from pinn.residuals import ns_residuals

K = 2.0 * math.pi


def taylor_green(xy):
    x, y = xy[:, 0:1], xy[:, 1:2]
    u = -torch.cos(K * x) * torch.sin(K * y)
    v = torch.sin(K * x) * torch.cos(K * y)
    p = -0.25 * (torch.cos(2 * K * x) + torch.cos(2 * K * y))
    return u, v, p


def _pts(n=200):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 2, generator=g, dtype=torch.float64)


def test_continuity_is_zero():
    c, _, _ = ns_residuals(taylor_green, _pts(), 0.01)
    assert c.abs().max() < 1e-10


def test_euler_limit_is_zero():  # nu = 0: convection + pressure gradient cancel
    _, mx, my = ns_residuals(taylor_green, _pts(), 0.0)
    assert mx.abs().max() < 1e-9 and my.abs().max() < 1e-9


def test_viscous_term_matches_analytic():
    nu = 0.037
    xy = _pts()
    _, mx, my = ns_residuals(taylor_green, xy, nu)
    u, v, _ = taylor_green(xy)
    assert torch.allclose(mx, 2 * nu * K**2 * u, atol=1e-9)
    assert torch.allclose(my, 2 * nu * K**2 * v, atol=1e-9)


def test_nu_tensor_gets_gradient():
    log_nu = torch.tensor(math.log(0.02), dtype=torch.float64, requires_grad=True)
    _, mx, my = ns_residuals(taylor_green, _pts(), torch.exp(log_nu))
    (mx**2 + my**2).mean().backward()
    assert log_nu.grad is not None and torch.isfinite(log_nu.grad)


def test_network_residual_is_differentiable_wrt_weights():
    net = FlowNet(Config(hidden_layers=2, hidden_units=8)).double()
    r = ns_residuals(net.uvp, _pts(50).float().double(), net.nu)
    sum((ri**2).mean() for ri in r).backward()
    assert all(p.grad is not None for p in net.parameters())
