"""Loading the OpenFOAM data and the Ghia values, getting the centerlines, and making the sensor data."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GHIA_PATH = DATA / "cavity" / "ghia_re100.csv"


def load_reference(path):
    # OpenFOAM fields at the cell centres, as a dict with x, y, u, v, p
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the Allrun script of that case first.")
    d = np.load(path)
    return {k: np.asarray(d[k], dtype=float) for k in ("x", "y", "u", "v", "p")}


def grid_view(ref):
    # Puts the cavity cell centres on a grid, indexed [j (y), i (x)].
    # I sort by x and y instead of trusting the OpenFOAM cell order, and check that it really is a regular grid.
    N = int(round(np.sqrt(len(ref["x"]))))
    order = np.lexsort((ref["x"], ref["y"]))        # sort by y, then by x
    X = ref["x"][order].reshape(N, N)
    Y = ref["y"][order].reshape(N, N)
    if not (np.allclose(X, X[0][None, :]) and np.allclose(Y, Y[:, 0][:, None])):
        raise ValueError("cell centres are not a regular grid")
    fields = {k: ref[k][order].reshape(N, N) for k in ("u", "v", "p")}
    return X[0], Y[:, 0], fields


def centerline_u(ref, y_query, lid_u=1.0):
    # u along x = 0.5. With an even number of cells x = 0.5 lies between two cell columns,
    # so I interpolate between those two first and then along y. I add the wall values as end points.
    xs, ys, f = grid_view(ref)
    column = np.array([np.interp(0.5, xs, f["u"][j]) for j in range(len(ys))])
    return np.interp(y_query, np.r_[0.0, ys, 1.0], np.r_[0.0, column, lid_u])


def centerline_v(ref, x_query):
    # v along y = 0.5 (v = 0 at the side walls), same idea as above
    xs, ys, f = grid_view(ref)
    row = np.array([np.interp(0.5, ys, f["v"][:, i]) for i in range(len(xs))])
    return np.interp(x_query, np.r_[0.0, xs, 1.0], np.r_[0.0, row, 0.0])


def load_ghia(path=GHIA_PATH):
    # The CSV has the columns profile, coordinate, value. Profile 'u' is u on x = 0.5 (coordinate = y),
    # profile 'v' is v on y = 0.5 (coordinate = x). The values are from Ghia, Ghia and Shin (1982).
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. It has to be typed in from the Ghia et al. paper "
                                "(format: see data/README.md).")
    df = pd.read_csv(path)
    out = {}
    for name in ("u", "v"):
        rows = df[df["profile"].astype(str).str.strip() == name].sort_values("coordinate")
        if rows.empty:
            raise ValueError(f"no rows with profile == '{name}' in {path}")
        out[name] = (rows["coordinate"].to_numpy(float), rows["value"].to_numpy(float))
    return out


def try_load_ghia(path=GHIA_PATH):
    # the Venturi case has no Ghia data, so a missing file should not stop anything
    try:
        return load_ghia(path)
    except FileNotFoundError:
        return None


def sensor_path(case, N, noise_pct, seed):
    return DATA / case / "sensors" / f"N{N}_noise{noise_pct:g}_seed{seed}.npz"


def make_sensors(ref, N, noise_pct, seed):
    # Picks N random cell centres (no repeats) and gives u and v there, never p.
    # The noise is Gaussian, with a standard deviation of noise_pct percent of the RMS of that
    # velocity component over the whole field, so the noise is relative to the flow.
    # The positions only depend on the seed, so 1 % and 5 % noise use the same points and are easier to compare.
    idx = np.random.default_rng(seed).choice(len(ref["x"]), size=N, replace=False)
    rng = np.random.default_rng([seed, N, int(round(noise_pct * 10))])
    sensors = {"cell_index": idx, "x": ref["x"][idx], "y": ref["y"][idx]}
    for c in ("u", "v"):
        rms = float(np.sqrt(np.mean(ref[c] ** 2)))
        std = noise_pct / 100.0 * rms
        sensors[c + "_clean"] = ref[c][idx]
        sensors[c] = ref[c][idx] + std * rng.standard_normal(N)
        sensors["noise_std_" + c] = std
    return sensors


def load_sensors(case, N, noise_pct, seed):
    path = sensor_path(case, N, noise_pct, seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/preprocessing/make_sensors.py --case {case}")
    d = np.load(path)
    return {k: d[k] for k in d.files}
