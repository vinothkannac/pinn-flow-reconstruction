"""The two flow domains and where their boundary points are.

cavity : unit square, the lid (y = 1) moves with u = 1, the other walls are no-slip.
venturi: 2D channel. At x = 0, 2, 4, 5, 9, 12 the full heights are 1, 1, 0.5, 0.5, 1, 1.
         Parabolic inlet (mean velocity 1), no-slip walls, outlet pressure p = 0.
The numbers are the same as in the OpenFOAM mesh, so the PINN and OpenFOAM solve the same problem.
"""
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parents[1] / "data"


class Geometry:
    name = ""
    bounds = ((0.0, 1.0), (0.0, 1.0))      # ((xmin, xmax), (ymin, ymax))
    reference_file = ""

    def inside(self, xy):
        raise NotImplementedError

    def boundary(self, n):
        raise NotImplementedError

    def outline(self):
        raise NotImplementedError

    def reference_path(self):
        return DATA / self.name / self.reference_file

    def sample_interior(self, n, seed):
        # n random points inside the domain. I draw points in the bounding box and throw away
        # the ones outside (needed for the Venturi). The seed makes it repeatable.
        (x0, x1), (y0, y1) = self.bounds
        gen = torch.Generator().manual_seed(seed)
        points = []
        while sum(len(p) for p in points) < n:
            r = torch.rand(2 * n, 2, generator=gen, dtype=torch.float64)
            xy = torch.stack([x0 + (x1 - x0) * r[:, 0], y0 + (y1 - y0) * r[:, 1]], dim=1)
            points.append(xy[torch.as_tensor(self.inside(xy.numpy()))])
        return torch.cat(points)[:n]

    def snapshot_points(self, nx=96):
        # regular grid of points inside the domain, only used for the training videos
        (x0, x1), (y0, y1) = self.bounds
        ny = max(int(round(nx * (y1 - y0) / (x1 - x0))), 24)
        gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) / nx * (x1 - x0),
                             y0 + (np.arange(ny) + 0.5) / ny * (y1 - y0))
        xy = np.c_[gx.ravel(), gy.ravel()]
        return xy[self.inside(xy)]


class Cavity(Geometry):
    name = "cavity"
    reference_file = "openfoam_N128.npz"

    def inside(self, xy):
        return (xy[:, 0] > 0) & (xy[:, 0] < 1) & (xy[:, 1] > 0) & (xy[:, 1] < 1)

    def boundary(self, n):
        # n points per wall at (i + 0.5) / n. That way the two lid corners are never used.
        # The velocity jumps there (0 to 1) and I did not want the network to chase that.
        s = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        zero, one = torch.zeros_like(s), torch.ones_like(s)
        xy = torch.cat([torch.stack([s, one], 1),      # lid
                        torch.stack([s, zero], 1),     # bottom
                        torch.stack([zero, s], 1),     # left
                        torch.stack([one, s], 1)])     # right
        uv = torch.zeros(4 * n, 2, dtype=torch.float64)
        uv[:n, 0] = 1.0                                # u = 1 on the lid
        return {"vel": (xy, uv), "out": None}

    def outline(self):
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)


class Venturi(Geometry):
    name = "venturi"
    reference_file = "openfoam_k4.npz"
    X = np.array([0.0, 2.0, 4.0, 5.0, 9.0, 12.0])
    H = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
    bounds = ((0.0, 12.0), (-0.5, 0.5))

    def half_height(self, x):
        return 0.5 * np.interp(x, self.X, self.H)

    def inside(self, xy):
        return (xy[:, 0] > 0) & (xy[:, 0] < 12) & (np.abs(xy[:, 1]) < self.half_height(xy[:, 0]))

    def boundary(self, n):
        m = 3 * n                                      # the walls are 12 long, so I give each one 3n points
        x = (torch.arange(m, dtype=torch.float64) + 0.5) / m * 12.0
        h = torch.as_tensor(self.half_height(x.numpy()))
        y_in = (torch.arange(n, dtype=torch.float64) + 0.5) / n - 0.5
        xy = torch.cat([torch.stack([x, h], 1),                              # upper wall
                        torch.stack([x, -h], 1),                             # lower wall
                        torch.stack([torch.zeros_like(y_in), y_in], 1)])     # inlet
        uv = torch.zeros(len(xy), 2, dtype=torch.float64)
        uv[2 * m:, 0] = 1.5 * (1 - (2 * y_in) ** 2)                          # parabolic inlet profile
        outlet = torch.stack([torch.full_like(y_in, 12.0), y_in], 1)
        return {"vel": (xy, uv), "out": outlet}

    def outline(self):
        top = np.c_[self.X, self.H / 2]
        return np.vstack([top, top[::-1] * [1, -1], top[:1]])


def get_geometry(name):
    return {"cavity": Cavity, "venturi": Venturi}[name]()
