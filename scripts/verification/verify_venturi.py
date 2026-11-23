"""Checks of the Venturi runs. There is no benchmark for this case, so I use three checks:
1. the lowest pressure on the three meshes (does it converge?)
2. the pressure gradient in the straight inlet should be that of plane Poiseuille flow, -12 nu U / H^2
3. the mass flux through a cross-section should be 1
Writes results/venturi/verification.csv."""
import argparse
import numpy as np
import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly

NU, U_MEAN, H_IN = 0.01, 1.0, 1.0
X, H = [0, 2, 4, 5, 9, 12], [1, 1, .5, .5, 1, 1]
ap = argparse.ArgumentParser()
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4])
a = ap.parse_args()
rows = []
for k in a.ks:
    d = np.load(ROOT / "data" / "venturi" / f"openfoam_k{k}.npz")
    x, y, u, p = d["x"], d["y"], d["u"], d["p"]
    cols = np.unique(np.round(x, 9))
    def column(xc):
        # all cells with x = xc, sorted by y
        m = np.isclose(x, xc)
        o = np.argsort(y[m])
        return y[m][o], u[m][o], p[m][o]

    def flux(xc):
        yy, uu, _ = column(xc)
        h = np.interp(xc, X, H)
        return float(np.sum(uu) * h / len(uu))   # all cells in a column have the same height, so the mean times h works
    # mean pressure over each cross-section, then a straight line fit in the inlet section for the slope
    pm = np.array([column(c)[2].mean() for c in cols])  # cross-section mean pressure
    m = (cols > 0.25) & (cols < 1.75)
    dpdx = np.polyfit(cols[m], pm[m], 1)[0]
    i = p.argmin()
    rows.append({"k": k, "cells": len(x), "p_min": p.min(), "x_pmin": x[i], "y_pmin": y[i],
                 "dp_inlet_to_outlet": pm[0], "dpdx_inlet_section": dpdx,
                 "dpdx_poiseuille": -12 * NU * U_MEAN / H_IN ** 2,
                 "flux_dev_inlet": flux(cols[0]) - 1, "flux_dev_throat": flux(cols[np.abs(cols - 4.5).argmin()]) - 1,
                 "flux_dev_outlet": flux(cols[-1]) - 1})
df = pd.DataFrame(rows)
(ROOT / "results" / "venturi").mkdir(parents=True, exist_ok=True)
df.to_csv(ROOT / "results" / "venturi" / "verification.csv", index=False)
print(df.to_string(index=False))
