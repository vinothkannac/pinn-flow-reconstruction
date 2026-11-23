"""Makes the summary tables and the two figures that need the full experiment grid.

Reads results/<case>/metrics.csv (or metrics_reduced.csv with --reduced) and writes
    results/<case>/summary.csv, summary.md      mean and std over seeds
    results/<case>/inverse.csv                  recovered viscosity per run
    results/figures/<case>_error_vs_noise.png
    results/figures/<case>_viscosity_recovery.png
The field plots, profiles, loss curves and videos are made by scripts/reporting/make_media.py.

    python experiments/summarize_results.py --case venturi
"""
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly

from pinn import plotting as P

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["cavity", "venturi"], required=True)
parser.add_argument("--reduced", action="store_true", help="use metrics_reduced.csv")
args = parser.parse_args()

tag = "reduced" if args.reduced else "full"
results = ROOT / "results" / args.case
table = pd.read_csv(results / ("metrics_reduced.csv" if args.reduced else "metrics.csv"))
P.apply_style()

# ---- mean and std over the seeds (this is why I want 3 seeds)
forward = table[table.model != "pinn_inverse"]
columns = [c for c in ("rel_l2_u", "rel_l2_v", "rel_l2_p", "p_min_abs_err", "p_min_loc_err", "train_time_s")
           if c in forward]
summary = forward.groupby(["model", "N", "noise_pct"])[columns].agg(["mean", "std"])
summary[("n_runs", "")] = forward.groupby(["model", "N", "noise_pct"]).size()
summary.to_csv(results / "summary.csv")
(results / "summary.md").write_text(summary.round(4).to_markdown())

inverse = table[table.model == "pinn_inverse"]
if len(inverse):
    inverse[["N", "noise_pct", "seed", "nu_est", "nu_rel_err", "rel_l2_u", "rel_l2_v", "status"]].to_csv(
        results / "inverse.csv", index=False)

# ---- relative L2 error against noise level
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
colors = {"pinn": P.C["pinn"], "data_only": P.C["data"]}
for ax, column in zip(axes, ("rel_l2_u", "rel_l2_v", "rel_l2_p")):
    for model in ("pinn", "data_only"):
        for N, style in ((50, "--"), (200, "-")):
            sub = forward[(forward.model == model) & (forward.N == N)].groupby("noise_pct")[column].agg(["mean", "std"])
            if len(sub):
                ax.errorbar(sub.index, sub["mean"], sub["std"].fillna(0), color=colors[model], ls=style,
                            marker="o", capsize=3, label=f"{P.LABEL[model]}, N = {N}")
    ax.set(xlabel="noise (% of RMS)", ylabel="relative L2 error", yscale="log", title=column)
axes[0].legend(fontsize=8)
fig.suptitle(f"{args.case}: error against noise (mean and std over seeds; {tag} runs)", x=0.01, ha="left",
             fontsize=11.5, fontweight="bold")
P.save(fig, ROOT / "results" / "figures" / f"{args.case}_error_vs_noise.png")

# ---- viscosity during training (inverse problem)
if len(inverse):
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in inverse.iterrows():
        history = results / "runs" / tag / f"{row.run_id}_history.npz"
        if history.exists():
            h = np.load(history)
            ax.plot(h["it"], h["nu"], label=f"noise {row.noise_pct:g} %, seed {int(row.seed)}")
    ax.axhline(0.01, color="k", ls=":", label="true nu = 0.01")
    ax.set(xlabel="iteration", ylabel="nu", title=f"{args.case}: viscosity recovered by the PINN (N = 200)")
    ax.legend(fontsize=8)
    P.save(fig, ROOT / "results" / "figures" / f"{args.case}_viscosity_recovery.png")
print("done")
