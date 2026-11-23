"""Runs the whole experiment grid for one case.

Grid: N in {50, 200} x noise in {0, 1, 5} % x seeds {0, 1, 2} x model in {data_only, pinn},
plus the inverse problem (N = 200, noise 1 and 5 %, 3 seeds).

Colab often disconnects, so I made it safe to start again: runs that are already in the metrics file
are skipped, and a run that was cut off continues from its last checkpoint.

Examples:
    python experiments/run_experiments.py --case venturi --device auto               # full grid
    python experiments/run_experiments.py --case cavity --quick --device cpu         # laptop smoke test
    python experiments/run_experiments.py --case venturi --only pinn --N 200 --noise 1 --seeds 0
"""
import sys
import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly

from pinn.config import Config, apply_overrides
from pinn.data_utils import load_reference, load_sensors, try_load_ghia
from pinn.geometry import get_geometry
from pinn.train import run_single, save_history

# the one run I use for the field plots and the videos (N, noise, seed)
REPRESENTATIVE = {"cavity": (50, 1.0, 0), "venturi": (200, 1.0, 0)}

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["cavity", "venturi"], required=True)
parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
parser.add_argument("--quick", action="store_true", help="few iterations; results go to the 'reduced' files")
parser.add_argument("--only", nargs="+", choices=["data_only", "pinn", "pinn_inverse"], default=None)
parser.add_argument("--N", type=int, nargs="+", default=[50, 200])
parser.add_argument("--noise", type=float, nargs="+", default=[0, 1, 5])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--config", default=None, help="JSON file with a saved Config")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
parser.add_argument("--force", action="store_true", help="run again even if the run is already in the csv")
args = parser.parse_args()

cfg = Config.from_json(args.config) if args.config else Config()
cfg = replace(apply_overrides(cfg, args.set), device=args.device, case=args.case)
if args.quick:
    cfg = apply_overrides(cfg.as_quick(), args.set)

tag = "reduced" if args.quick else "full"
results = ROOT / "results" / args.case
metrics_file = results / ("metrics_reduced.csv" if args.quick else "metrics.csv")
for folder in ("runs", "predictions", "snapshots", "checkpoints"):
    (results / folder / tag).mkdir(parents=True, exist_ok=True)

ref = load_reference(get_geometry(args.case).reference_path())
ghia = try_load_ghia() if args.case == "cavity" else None

# list of all runs as (model, N, noise, seed). The inverse problem only runs with N = 200 and 1 or 5 % noise.
jobs = []
for model in args.only or ["data_only", "pinn", "pinn_inverse"]:
    if model == "pinn_inverse":
        grid = itertools.product([200], [n for n in args.noise if n in (1, 5)], args.seeds)
    else:
        grid = itertools.product(args.N, args.noise, args.seeds)
    jobs += [(model, N, noise, seed) for N, noise, seed in grid]

finished = set()
if metrics_file.exists() and not args.force:
    finished = set(pd.read_csv(metrics_file)["run_id"])

for model, N, noise, seed in jobs:
    run_id = f"{model}_N{N}_noise{noise:g}_seed{seed}"
    if run_id in finished:
        print("skip", run_id)
        continue
    print("=== " + run_id + " ===", flush=True)

    sensors = load_sensors(args.case, N, noise, seed)
    is_representative = (N, noise, seed) == REPRESENTATIVE[args.case] and model in ("pinn", "data_only")
    run_cfg = replace(cfg, seed=seed, snapshot_every=(100 if args.quick else 250) if is_representative else 0)
    row, net, history, pred = run_single(run_cfg, model, ref, ghia, sensors,
                                         run_dir=results / "checkpoints" / tag / run_id, resume=True)
    row.update({"run_id": run_id, "N": N, "noise_pct": noise, "seed": seed})
    save_history(results / "runs" / tag / f"{run_id}_history.npz", history)

    if is_representative:
        np.savez(results / "predictions" / tag / f"{run_id}.npz",
                 x=ref["x"], y=ref["y"], u=pred[:, 0], v=pred[:, 1], p=pred[:, 2])
        snap_xy, snaps = run_single.last_snapshots
        np.savez(results / "snapshots" / tag / f"{run_id}.npz", xy=snap_xy, it=np.array(snaps["it"]),
                 uvp=np.stack(snaps["uvp"]))

    # add the row to the csv. The models have different columns (the inverse one has nu_est), so I use concat
    new_row = pd.DataFrame([row])
    table = pd.concat([pd.read_csv(metrics_file), new_row], ignore_index=True) if metrics_file.exists() else new_row
    table.to_csv(metrics_file, index=False)
    print({k: row[k] for k in ("status", "rel_l2_u", "rel_l2_v", "rel_l2_p", "train_time_s") if k in row})
