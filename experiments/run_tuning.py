"""A short sweep over the loss weights. Every setting is added to results/<case>/tuning_log.csv.

I choose the best setting with the error on a second set of noisy sensors that the network did not train on
(200 other random cells, different seed). I do not choose with the error against the full OpenFOAM field,
because in a real problem I would not have that field and it would be cheating. The errors against the
reference are logged too, but only to look at. Tuning uses seed 100 and the experiments use 0 to 2,
so the tuning does not see the same sensors as the final runs.

    python experiments/run_tuning.py --case venturi --quick     # smoke test
    python experiments/run_tuning.py --case venturi              # full length
"""
import sys
import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly

from pinn.config import Config
from pinn.data_utils import load_reference, make_sensors
from pinn.evaluate import evaluate_fields, predict
from pinn.geometry import get_geometry
from pinn.train import train

# (lambda_data, lambda_pde, lambda_bc). The first one is my default, the rest change one weight at a time.
SWEEP = [(1, 1, 10), (1, 1, 1), (1, 1, 100), (1, 0.1, 10), (1, 10, 10), (10, 1, 10)]
TUNE_SEED, VAL_SEED = 100, 1000

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["cavity", "venturi"], required=True)
parser.add_argument("--N", type=int, default=50)
parser.add_argument("--noise", type=float, default=1.0)
parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

log_file = ROOT / "results" / args.case / "tuning_log.csv"
log_file.parent.mkdir(parents=True, exist_ok=True)

ref = load_reference(get_geometry(args.case).reference_path())
sensors = make_sensors(ref, args.N, args.noise, TUNE_SEED)
val = make_sensors(ref, 200, args.noise, VAL_SEED)
base = replace(Config(), case=args.case, seed=TUNE_SEED, device=args.device)
if args.quick:
    base = base.as_quick()

done = pd.read_csv(log_file) if log_file.exists() else pd.DataFrame()
for l_data, l_pde, l_bc in SWEEP:
    if len(done):
        same = (done.lambda_data == l_data) & (done.lambda_pde == l_pde) & (done.lambda_bc == l_bc) \
            & (done.N == args.N) & (done.noise_pct == args.noise) & (done.quick == args.quick)
        if same.any():
            print("skip", (l_data, l_pde, l_bc))
            continue

    cfg = replace(base, lambda_data=l_data, lambda_pde=l_pde, lambda_bc=l_bc)
    result = train(cfg, sensors, verbose=False)
    net = result["net"]

    pred_val = predict(net, np.c_[val["x"], val["y"]])
    val_rmse = float(np.sqrt(np.mean((pred_val[:, 0] - val["u"]) ** 2 + (pred_val[:, 1] - val["v"]) ** 2) / 2))
    pred_all = predict(net, np.c_[ref["x"], ref["y"]])

    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "lambda_data": l_data, "lambda_pde": l_pde,
           "lambda_bc": l_bc, "N": args.N, "noise_pct": args.noise, "seed": TUNE_SEED, "quick": args.quick,
           "adam_iters": cfg.adam_iters, "lbfgs_iters": cfg.lbfgs_iters, "status": result["status"],
           "val_rmse_uv": val_rmse, "train_time_s": result["elapsed"], "device": result["device"]}
    if np.isfinite(pred_all).all():
        row.update(evaluate_fields(pred_all, ref))
    new_row = pd.DataFrame([row])
    table = pd.concat([pd.read_csv(log_file), new_row], ignore_index=True) if log_file.exists() else new_row
    table.to_csv(log_file, index=False)
    print(row, flush=True)

table = pd.read_csv(log_file)
table = table[(table.N == args.N) & (table.noise_pct == args.noise) & (table.quick == args.quick) & (table.status == "ok")]
if len(table):
    print("\nbest settings by held-out RMSE:")
    print(table.sort_values("val_rmse_uv").head(3).to_string(index=False))
