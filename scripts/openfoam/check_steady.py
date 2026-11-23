"""Check that an icoFoam cavity run has reached a steady state.

icoFoam is a transient solver, so I run it until nothing changes any more. My test: the biggest
change of U in any cell between the last two written times (5 time units apart) must be below
the tolerance. I also check that the Courant number in the log stayed below 0.5.
The result goes to results/cavity/steady_N<N>.json.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly
from pinn.foam_io import read_field, time_dirs

parser = argparse.ArgumentParser()
parser.add_argument("--case", required=True, help="path of the run folder")
parser.add_argument("--N", type=int, required=True, help="cells per direction")
parser.add_argument("--tol", type=float, default=1e-5)
args = parser.parse_args()

# only look at folders that have a U file (the 0 folder is the start, I skip it)
times = [(t, d) for t, d in time_dirs(args.case) if t > 0 and (d / "U").exists()]
if len(times) < 2:
    sys.exit("need at least two written time directories - did icoFoam finish?")
(t_prev, dir_prev), (t_last, dir_last) = times[-2], times[-1]

n_cells = args.N * args.N
U_prev = read_field(dir_prev / "U", n_cells)
U_last = read_field(dir_last / "U", n_cells)
p_prev = read_field(dir_prev / "p", n_cells)
p_last = read_field(dir_last / "p", n_cells)
# only the first two columns of U, the z component is always 0 in 2D
dU = float(np.max(np.abs(U_last[:, :2] - U_prev[:, :2])))
dp = float(np.max(np.abs(p_last - p_prev)))

log_text = (Path(args.case) / "log.icoFoam").read_text()
courant = [float(c) for c in re.findall(r"Courant Number mean: \S+ max: (\S+)", log_text)]
co_max = max(courant) if courant else float("nan")

result = {"N": args.N, "t_prev": t_prev, "t_last": t_last,
          "max_abs_dU_over_interval": dU, "max_abs_dp_over_interval": dp, "tol": args.tol,
          "steady": bool(dU < args.tol), "max_courant": co_max, "courant_ok": bool(co_max < 0.5)}
out_dir = ROOT / "results" / "cavity"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"steady_N{args.N}.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
sys.exit(0 if result["steady"] and result["courant_ok"] else 1)
