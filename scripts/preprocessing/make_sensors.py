"""Creates the sensor files from the finest OpenFOAM mesh: N = 50 and 200, noise 0, 1 and 5 %, seeds 0 to 2.
I save them to files so every run uses exactly the same sensors."""
import argparse

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly
from pinn.data_utils import load_reference, make_sensors, sensor_path
from pinn.geometry import get_geometry
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["cavity", "venturi"], required=True)
parser.add_argument("--N", type=int, nargs="+", default=[50, 200])
parser.add_argument("--noise", type=float, nargs="+", default=[0, 1, 5])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = parser.parse_args()

ref = load_reference(get_geometry(args.case).reference_path())
for N in args.N:
    for noise in args.noise:
        for seed in args.seeds:
            path = sensor_path(args.case, N, noise, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **make_sensors(ref, N, noise, seed))
print("sensor files written for", args.case)
