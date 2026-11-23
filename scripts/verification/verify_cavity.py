"""Compare the OpenFOAM centerline profiles (32, 64, 128 cells) with Ghia, Ghia and Shin (1982), Re = 100.

I interpolate OpenFOAM linearly to the Ghia coordinates. The two rows at the walls (coordinate 0 and 1)
are just the boundary conditions, so I leave them out of the max and RMS numbers.
Writes results/cavity/openfoam_vs_ghia.csv and results/figures/cavity_mesh_convergence.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly
from pinn import plotting as P
from pinn.data_utils import centerline_u, centerline_v, load_ghia, load_reference

ghia = load_ghia()
y_ghia, u_ghia = ghia["u"]
x_ghia, v_ghia = ghia["v"]

rows, profiles = [], {}
s = np.linspace(0, 1, 401)
for N in (32, 64, 128):
    ref = load_reference(ROOT / "data" / "cavity" / f"openfoam_N{N}.npz")
    du = (centerline_u(ref, y_ghia) - u_ghia)[(y_ghia > 0) & (y_ghia < 1)]
    dv = (centerline_v(ref, x_ghia) - v_ghia)[(x_ghia > 0) & (x_ghia < 1)]
    rows.append({"mesh": f"{N}x{N}", "n_points_u": len(du), "n_points_v": len(dv),
                 "u_max_abs_dev": np.abs(du).max(), "u_rms_dev": np.sqrt(np.mean(du ** 2)),
                 "v_max_abs_dev": np.abs(dv).max(), "v_rms_dev": np.sqrt(np.mean(dv ** 2))})
    profiles[N] = (centerline_u(ref, s), centerline_v(ref, s))

table = pd.DataFrame(rows)
table.to_csv(ROOT / "results" / "cavity" / "openfoam_vs_ghia.csv", index=False)
print(table.to_string(index=False))

P.apply_style()
colors = {32: P.C["muted"], 64: P.C["data"], 128: P.C["pinn"]}
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
for N, (u, v) in profiles.items():
    ax[0].plot(u, s, color=colors[N], label=f"OpenFOAM {N}x{N}")
    ax[1].plot(s, v, color=colors[N], label=f"OpenFOAM {N}x{N}")
ax[0].plot(u_ghia, y_ghia, "o", mfc="none", mec=P.C["text"], label="Ghia et al. (1982)")
ax[1].plot(x_ghia, v_ghia, "o", mfc="none", mec=P.C["text"], label="Ghia et al. (1982)")
ax[0].set(xlabel="u", ylabel="y", title="u on x = 0.5")
ax[1].set(xlabel="x", ylabel="v", title="v on y = 0.5")
ax[0].legend(fontsize=8.5)
fig.suptitle("Lid-driven cavity, Re = 100: OpenFOAM against Ghia et al.", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
P.save(fig, ROOT / "results" / "figures" / "cavity_mesh_convergence.png")
