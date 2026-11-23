"""Navier-Stokes residuals from autograd. Used in training and in the tests."""
import torch


def _grad(f, xy):
    return torch.autograd.grad(f, xy, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def ns_residuals(field_fn, xy, nu):
    # field_fn(xy) gives (u, v, p), each of shape (N, 1). I return three residuals:
    #   continuity  u_x + v_y
    #   x-momentum  u u_x + v u_y + p_x - nu (u_xx + u_yy)
    #   y-momentum  u v_x + v v_y + p_y - nu (v_xx + v_yy)
    # They are zero for a perfect solution. The flow is steady, so there is no time term.
    # nu is a number normally and a tensor in the inverse problem (so it gets a gradient).
    if not xy.requires_grad:
        xy = xy.detach().requires_grad_(True)
    u, v, p = field_fn(xy)

    # first derivatives (create_graph=True because I take derivatives of these again)
    gu, gv, gp = _grad(u, xy), _grad(v, xy), _grad(p, xy)
    u_x, u_y = gu[:, 0:1], gu[:, 1:2]
    v_x, v_y = gv[:, 0:1], gv[:, 1:2]
    p_x, p_y = gp[:, 0:1], gp[:, 1:2]

    # second derivatives, only the ones the viscous term needs
    u_xx = _grad(u_x, xy)[:, 0:1]
    u_yy = _grad(u_y, xy)[:, 1:2]
    v_xx = _grad(v_x, xy)[:, 0:1]
    v_yy = _grad(v_y, xy)[:, 1:2]

    continuity = u_x + v_y
    mom_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return continuity, mom_x, mom_y
