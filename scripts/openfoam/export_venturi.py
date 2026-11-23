"""Export the converged Venturi fields at the cell centres to data/venturi/openfoam_k<k>.npz.
Same idea as export_cavity.py."""
import argparse

import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly
from pinn.foam_io import read_field, time_dirs

parser = argparse.ArgumentParser()
parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4])
args = parser.parse_args()

# same shape as in generate_venturi_mesh.py
X_STATION = [0, 2, 4, 5, 9, 12]
HEIGHT = [1, 1, 0.5, 0.5, 1, 1]

for k in args.ks:
    case = ROOT / "openfoam" / "venturi" / f"run_k{k}"
    t, folder = [(t, d) for t, d in time_dirs(case) if t > 0][-1]
    # mesh: 20k cells per unit length along x, 20k cells across
    n = 12 * 20 * k * 20 * k
    C = read_field(folder / "C", n)
    U = read_field(folder / "U", n)
    p = read_field(folder / "p", n)
    assert np.abs(U[:, 2]).max() < 1e-10    # no flow in z, otherwise something is wrong with the 2D setup
    half = np.interp(C[:, 0], X_STATION, HEIGHT) / 2
    assert (np.abs(C[:, 1]) < half).all(), "a cell centre lies outside the geometry"

    out = ROOT / "data" / "venturi" / f"openfoam_k{k}.npz"
    np.savez(out, x=C[:, 0], y=C[:, 1], u=U[:, 0], v=U[:, 1], p=p, iteration=t)
    print(f"k={k}: {n} cells, converged after {t:g} iterations -> {out.name}")
