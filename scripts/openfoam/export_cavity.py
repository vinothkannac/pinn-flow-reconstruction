"""Export the steady cavity fields at the cell centres to data/cavity/openfoam_N<N>.npz (x, y, u, v, p).

The cell centres x and y come from the field C that `postProcess -func writeCellCentres` writes.
It has the same cell order as U and p, so I do not have to guess the order.
As an extra check I compare the centres with the uniform grid I expect.
"""
import argparse

import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly
from pinn.foam_io import read_field, time_dirs

parser = argparse.ArgumentParser()
parser.add_argument("--meshes", type=int, nargs="+", default=[32, 64, 128])
args = parser.parse_args()

for N in args.meshes:
    case = ROOT / "openfoam" / "cavity" / f"run_N{N}"
    t, folder = [(t, d) for t, d in time_dirs(case) if t > 0][-1]
    n = N * N
    C = read_field(folder / "C", n)
    U = read_field(folder / "U", n)
    p = read_field(folder / "p", n)
    x, y = C[:, 0], C[:, 1]

    g = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(g, g)
    a, b = np.lexsort((x, y)), np.lexsort((X.ravel(), Y.ravel()))
    err = max(np.abs(x[a] - X.ravel()[b]).max(), np.abs(y[a] - Y.ravel()[b]).max())
    assert err < 1e-9, f"cell centres of N={N} differ from the uniform grid by {err}"

    out = ROOT / "data" / "cavity" / f"openfoam_N{N}.npz"
    np.savez(out, x=x, y=y, u=U[:, 0], v=U[:, 1], p=p, time=t)
    print(f"N={N}: {n} cells from time {t:g} -> {out.name}")
