"""The network: (x, y) in, (u, v, p) out. In the inverse problem the viscosity is trained as well."""
import math

import torch
from torch import nn

from .geometry import get_geometry


class FlowNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # I scale x and y to [-1, 1] with the bounding box. The Venturi is 12 long and 1 high,
        # and without scaling the network would see very different input sizes.
        (x0, x1), (y0, y1) = get_geometry(cfg.case).bounds
        self.register_buffer("lo", torch.tensor([x0, y0], dtype=torch.float32))
        self.register_buffer("hi", torch.tensor([x1, y1], dtype=torch.float32))

        sizes = [2] + [cfg.hidden_units] * cfg.hidden_layers + [3]
        self.layers = nn.ModuleList()
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layer = nn.Linear(n_in, n_out)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

        # I train log(nu) and not nu itself, so nu always stays positive.
        # It is only trainable in the inverse problem.
        if cfg.inverse:
            self.log_nu = nn.Parameter(torch.tensor(math.log(cfg.nu_init)))
        else:
            self.register_buffer("log_nu", torch.tensor(math.log(cfg.nu)))

    @property
    def nu(self):
        return torch.exp(self.log_nu)

    def uvp(self, xy):
        # xy has shape (N, 2). Returns u, v, p, each with shape (N, 1).
        # I use tanh because the residual needs second derivatives and ReLU has none.
        h = 2.0 * (xy - self.lo) / (self.hi - self.lo) - 1.0
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        out = self.layers[-1](h)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
