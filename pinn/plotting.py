"""Plot style and helper functions that all the figure scripts use.
I keep the colours the same everywhere: blue = PINN, orange = data-only, green = sensors, black = OpenFOAM.
For the fields: cividis for speed (readable with colour blindness), red-blue centred on 0 for pressure, magma for errors.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # no window needed, I only save files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri as mtri
from matplotlib.colors import Normalize, TwoSlopeNorm

C = dict(ref="#1c1c1b", pinn="#2a78d6", data="#eb6834", aux="#1baf7a", muted="#8a8984",
         grid="#e7e6e1", text="#0b0b0b", text2="#52514e", bg="#ffffff")
CM_SPEED, CM_ERR, CM_P, CM_SIGNED = "cividis", "magma", "RdBu_r", "RdBu_r"
LABEL = {"ref": "OpenFOAM", "pinn": "PINN", "data_only": "Data-only", "pinn_inverse": "PINN (inverse)"}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": C["bg"],
        "figure.facecolor": C["bg"], "axes.facecolor": C["bg"], "font.family": "DejaVu Sans",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": C["text2"], "text.color": C["text"], "xtick.color": C["text2"], "ytick.color": C["text2"],
        "axes.edgecolor": C["muted"], "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.7, "axes.axisbelow": True,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9,
        "image.interpolation": "nearest",
    })


def triangulation(x, y, geom):
    # triangles between the cell centres; I hide the ones that fall outside the Venturi shape
    t = mtri.Triangulation(x, y)
    cen = np.c_[x[t.triangles].mean(1), y[t.triangles].mean(1)]
    t.set_mask(~geom.inside(cen))
    return t


def draw_field(ax, tri, z, cmap, norm=None, vmin=None, vmax=None, geom=None, title=None):
    if norm is None:
        norm = Normalize(vmin if vmin is not None else np.min(z), vmax if vmax is not None else np.max(z))
    m = ax.tripcolor(tri, z, shading="gouraud", cmap=cmap, norm=norm, rasterized=True)
    if geom is not None:
        o = geom.outline()
        ax.plot(o[:, 0], o[:, 1], color=C["ref"], lw=1.1, solid_joinstyle="round")
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", pad=4)
    return m


def pressure_norm(p, center=0.0, clip=None):
    # Colour scale with white at `center`. With clip=(low, high) in percent the scale ignores
    # extreme values, which I need for the cavity because of the corners.
    lo, hi = (float(np.min(p)), float(np.max(p))) if clip is None else map(float, np.percentile(p, clip))
    if lo < center < hi:
        return TwoSlopeNorm(center, lo, hi)
    return Normalize(lo, hi)


def colorbar(fig, m, ax, label, orientation="vertical", cax=None, **kw):
    if cax is not None:
        cb = fig.colorbar(m, cax=cax)
    else:
        cb = fig.colorbar(m, ax=ax, orientation=orientation, pad=0.015, fraction=kw.pop("fraction", 0.04),
                          aspect=kw.pop("aspect", 28), **kw)
    cb.set_label(label, color=C["text2"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    return cb


def streamlines(ax, tri, u, v, geom, nx=480, density=1.3, color="white", lw=0.7, alpha=0.9, seeds=None):
    (x0, x1), (y0, y1) = geom.bounds
    ny = max(int(nx * (y1 - y0) / (x1 - x0)), 40)
    ex, ey = 0.004 * (x1 - x0), 0.01 * (y1 - y0)   # stay a bit away from the wall, the interpolation looks bad right at the edge
    gx, gy = np.linspace(x0 + ex, x1 - ex, nx), np.linspace(y0 + ey, y1 - ey, ny)
    X, Y = np.meshgrid(gx, gy)
    U = mtri.LinearTriInterpolator(tri, u)(X, Y)
    V = mtri.LinearTriInterpolator(tri, v)(X, Y)
    kw = {} if seeds is None else dict(start_points=seeds, integration_direction="forward")
    ax.streamplot(gx, gy, U, V, color=color, linewidth=lw, density=density, arrowsize=0.6 if seeds is None else 0.0001,
                  broken_streamlines=False, **kw)


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
