"""Training. Adam first, then L-BFGS. The same code trains the PINN, the data-only baseline and the inverse problem.

Example:
    python -m pinn.train --case venturi --model pinn --N 200 --noise 1 --seed 0 --quick --out runs/test
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .config import Config, apply_overrides
from .data_utils import load_reference, load_sensors, try_load_ghia
from .evaluate import evaluate_fields, ghia_deviation, predict, venturi_pressure_metrics
from .geometry import get_geometry
from .model import FlowNet
from .residuals import ns_residuals

TERMS = ("data", "pde", "bc", "out")


# ---- setup
def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def set_seed(seed):
    # I fix all the seeds so a run can be repeated.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sample_colloc(cfg, block, device, dtype):
    # Random points inside the domain for the physics loss. The seed depends on the run seed and on
    # the block number, so the same run always gets the same points (needed to resume from a checkpoint).
    block = max(block, 0) if cfg.resample_every > 0 else 0
    pts = get_geometry(cfg.case).sample_interior(cfg.n_colloc, cfg.seed * 1_000_003 + block)
    return pts.to(device, dtype)


# ---- loss
def _mse(a):
    return (a ** 2).mean()


def compute_losses(net, cfg, data, colloc, terms=None):
    # Gives a dict with each loss term (data, pde, bc, out) and 'total', which is the sum of lambda * term.
    # Terms with lambda = 0 are skipped by default, so the data-only baseline does not compute the physics.
    lam = {"data": cfg.lambda_data, "pde": cfg.lambda_pde, "bc": cfg.lambda_bc,
           "out": cfg.lambda_out if data.get("xy_o") is not None else 0.0}
    if terms is None:
        terms = [t for t in TERMS if lam[t] > 0]
    out, total = {}, 0.0
    for t in terms:
        if t == "data":
            xy = data["xy_s"].clone()
            u, v, _ = net.uvp(xy)
            val = 0.5 * (_mse(u - data["u_s"]) + _mse(v - data["v_s"]))
        elif t == "bc":
            xy = data["xy_b"].clone()
            u, v, _ = net.uvp(xy)
            val = 0.5 * (_mse(u - data["uv_b"][:, 0:1]) + _mse(v - data["uv_b"][:, 1:2]))
        elif t == "out":   # outlet pressure p = 0 (reference pressure of geometries with an outlet)
            if data.get("xy_o") is None:
                continue
            xy = data["xy_o"].clone()
            val = _mse(net.uvp(xy)[2])
        else:
            r = ns_residuals(net.uvp, colloc, net.nu)
            val = sum(_mse(ri) for ri in r) / 3.0
        out[t] = val
        if lam[t] > 0:
            total = total + lam[t] * val
    out["total"] = total
    return out


# ---- checkpoints
def _save(path, payload):
    # I write to a temporary file first and then rename it. If Colab kills the session while saving,
    # the old checkpoint is still fine.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _empty_history():
    return {k: [] for k in ("it", "phase", "total", "data", "pde", "bc", "out", "nu", "lr", "t")}


# ---- train
def train(cfg, sensors, ckpt_dir=None, resume=False, verbose=True):
    # Returns a dict with net, history, elapsed, status and device. status is 'ok' or 'diverged'.
    device = get_device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    set_seed(cfg.seed)
    net = FlowNet(cfg).to(device=device, dtype=dtype)

    geom = get_geometry(cfg.case)
    bnd = geom.boundary(cfg.n_bc_per_wall)
    xy_b, uv_b = (t.to(device, dtype) for t in bnd["vel"])
    data = {
        "xy_s": torch.as_tensor(np.c_[sensors["x"], sensors["y"]], dtype=dtype, device=device),
        "u_s": torch.as_tensor(sensors["u"], dtype=dtype, device=device).reshape(-1, 1),
        "v_s": torch.as_tensor(sensors["v"], dtype=dtype, device=device).reshape(-1, 1),
        "xy_b": xy_b, "uv_b": uv_b,
        "xy_o": None if bnd["out"] is None else bnd["out"].to(device, dtype),
    }
    snap_xy = geom.snapshot_points() if cfg.snapshot_every > 0 else None
    snaps = {"it": [], "uvp": []}

    def snapshot(it):
        if snap_xy is not None:
            snaps["it"].append(it)
            snaps["uvp"].append(predict(net, snap_xy).astype(np.float32))

    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=cfg.adam_lr)
    if cfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.adam_iters, 1),
                                                           eta_min=cfg.adam_lr * cfg.lr_min_factor)
    elif cfg.lr_schedule == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, cfg.lr_step_size, cfg.lr_gamma)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)

    hist, start_it, lbfgs_done, elapsed0 = _empty_history(), 0, 0, 0.0
    ckpt = Path(ckpt_dir) / "last.pt" if ckpt_dir else None
    if resume and ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location=device)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["adam"])
        sched.load_state_dict(st["sched"])
        hist, start_it, lbfgs_done, elapsed0 = st["history"], st["it"], st["lbfgs_done"], st["elapsed"]
        if verbose:
            print(f"[resume] it={start_it} lbfgs_done={lbfgs_done} from {ckpt}")

    t0 = time.perf_counter()
    clock = lambda: elapsed0 + time.perf_counter() - t0
    status = "ok"

    def save(it, lb):
        if ckpt is None:
            return
        _save(ckpt, {"model": net.state_dict(), "adam": opt.state_dict(), "sched": sched.state_dict(),
                     "it": it, "lbfgs_done": lb, "history": hist, "elapsed": clock(), "cfg": asdict(cfg)})

    def log(it, phase, L, colloc, lr):
        vals = {k: float(v.detach()) for k, v in L.items() if k != "total"}
        missing = [t for t in TERMS if t not in vals and (t != "out" or data["xy_o"] is not None)]
        if missing:  # terms with lambda = 0 are still logged, e.g. so I can see the PDE residual of the data-only network
            with torch.enable_grad():
                extra = compute_losses(net, cfg, data, colloc, terms=missing)
            vals.update({k: float(extra[k].detach()) for k in missing})
        hist["it"].append(it); hist["phase"].append(phase); hist["total"].append(float(L["total"].detach()))
        for t in TERMS:
            hist[t].append(vals.get(t, float("nan")))
        hist["nu"].append(float(net.nu) if cfg.inverse else float("nan"))
        hist["lr"].append(lr); hist["t"].append(clock())
        if verbose:
            print(f"it {it:6d} {'adam ' if phase == 0 else 'lbfgs'} total {hist['total'][-1]:.3e} "
                  f"data {vals['data']:.3e} pde {vals['pde']:.3e} bc {vals['bc']:.3e}"
                  + (f" out {vals['out']:.3e}" if 'out' in vals else "")
                  + (f" nu {hist['nu'][-1]:.5f}" if cfg.inverse else ""), flush=True)

    # ---- Adam
    block, colloc = -1, None
    for it in range(start_it, cfg.adam_iters):
        b = it // cfg.resample_every if cfg.resample_every > 0 else 0
        if b != block:
            block, colloc = b, sample_colloc(cfg, b, device, dtype)
        opt.zero_grad()
        L = compute_losses(net, cfg, data, colloc)
        L["total"].backward()
        opt.step()
        sched.step()
        if it % cfg.log_every == 0 or it == cfg.adam_iters - 1:
            log(it, 0, L, colloc, opt.param_groups[0]["lr"])
        if cfg.snapshot_every > 0 and it % cfg.snapshot_every == 0:
            snapshot(it)
        if not math.isfinite(L["total"].item()):
            status = "diverged"
            break
        if (it + 1) % cfg.ckpt_every == 0:
            save(it + 1, 0)

    # ---- L-BFGS
    # I keep the collocation points fixed here. L-BFGS needs a loss that does not change between steps.
    # I run it in chunks so I can save a checkpoint in between.
    if status == "ok" and cfg.lbfgs_iters > 0:
        colloc = sample_colloc(cfg, cfg.adam_iters // max(cfg.resample_every, 1) + 1, device, dtype)
        chunk = max(cfg.ckpt_every, 1)
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=chunk, max_eval=int(chunk * 1.25),
                               history_size=cfg.lbfgs_history, tolerance_grad=1e-12,
                               tolerance_change=1e-14, line_search_fn="strong_wolfe")
        evals = {"n": 0, "last": None}

        def closure():
            lb.zero_grad()
            L = compute_losses(net, cfg, data, colloc)
            L["total"].backward()
            evals["n"] += 1
            evals["last"] = L
            if evals["n"] % cfg.log_every == 0:
                log(cfg.adam_iters + lbfgs_done + evals["n"], 1, L, colloc, 1.0)  # x-axis: function evaluations
            return L["total"]

        while lbfgs_done < cfg.lbfgs_iters:
            n_run = min(chunk, cfg.lbfgs_iters - lbfgs_done)
            lb.param_groups[0]["max_iter"] = n_run
            lb.param_groups[0]["max_eval"] = int(n_run * 1.25)
            before = lb.state[params[0]].get("n_iter", 0)
            lb.step(closure)
            ran = lb.state[params[0]].get("n_iter", 0) - before
            lbfgs_done += max(ran, 1)
            if not math.isfinite(evals["last"]["total"].item()):
                status = "diverged"
                break
            save(cfg.adam_iters, lbfgs_done)
            snapshot(cfg.adam_iters + lbfgs_done)
            if ran < n_run:  # converged / no progress
                break
        if evals["last"] is not None:
            log(cfg.adam_iters + lbfgs_done + evals["n"], 1, evals["last"], colloc, 1.0)

    elapsed = clock()
    save(cfg.adam_iters, cfg.lbfgs_iters)
    snapshot(cfg.adam_iters + lbfgs_done)   # final state
    return {"net": net, "history": hist, "elapsed": elapsed, "status": status, "device": str(device),
            "snapshots": snaps, "snapshot_xy": snap_xy}


# ---- one full run
def run_single(cfg, kind, ref, ghia, sensors, run_dir=None, resume=False, verbose=True):
    # Trains one model and computes its errors. Returns (metrics row, net, history, prediction).
    cfg = cfg.for_kind(kind)
    res = train(cfg, sensors, ckpt_dir=run_dir, resume=resume, verbose=verbose)
    net = res["net"]
    pred = predict(net, np.c_[ref["x"], ref["y"]])
    row = {"model": kind, "status": res["status"], "device": res["device"], "dtype": cfg.dtype, "quick": cfg.quick, "train_time_s": res["elapsed"],
           "adam_iters": cfg.adam_iters, "lbfgs_iters": cfg.lbfgs_iters,
           "hidden_layers": cfg.hidden_layers, "hidden_units": cfg.hidden_units, "n_colloc": cfg.n_colloc,
           "case": cfg.case,
           "lambda_data": cfg.lambda_data, "lambda_pde": cfg.lambda_pde, "lambda_bc": cfg.lambda_bc}
    if np.isfinite(pred).all():
        row.update(evaluate_fields(pred, ref))
        row.update(ghia_deviation(net, ghia))
    else:
        row["status"] = "nan_prediction"
    if cfg.case == "venturi" and cfg.lambda_pde > 0 and np.isfinite(pred).all():
        row.update(venturi_pressure_metrics(pred, ref))
    for k in ("total", "data", "pde", "bc", "out"):
        row["final_loss_" + k] = res["history"][k][-1] if res["history"][k] else float("nan")
    if cfg.inverse:
        nu = float(net.nu)
        row["nu_est"] = nu
        row["nu_rel_err"] = abs(nu - cfg.nu) / cfg.nu
    # I store the video snapshots on the function so run_experiments.py can get them without changing the return values
    run_single.last_snapshots = (res["snapshot_xy"], res["snapshots"])
    return row, net, res["history"], pred


def save_history(path, history):
    np.savez(path, **{k: np.asarray(v, dtype=float) for k, v in history.items()})


# ---- command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["pinn", "data_only", "pinn_inverse"], default="pinn")
    ap.add_argument("--case", choices=["cavity", "venturi"], default="cavity")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--noise", type=float, default=1.0, help="noise in percent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--quick", action="store_true", help="few iterations, for a CPU smoke test")
    ap.add_argument("--config", default=None, help="JSON file written by Config.to_json")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default="results/single")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cfg = Config.from_json(a.config) if a.config else Config()
    cfg = apply_overrides(cfg, a.set)
    cfg = replace(cfg, seed=a.seed, device=a.device, case=a.case)
    if a.quick:
        cfg = cfg.as_quick()
        cfg = apply_overrides(cfg, a.set)  # explicit --set wins over quick defaults

    ref = load_reference(get_geometry(a.case).reference_path())
    ghia = try_load_ghia() if a.case == "cavity" else None
    sensors = load_sensors(a.case, a.N, a.noise, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    row, net, hist, _ = run_single(cfg, a.model, ref, ghia, sensors, run_dir=out / "ckpt", resume=a.resume)
    row.update({"N": a.N, "noise_pct": a.noise, "seed": a.seed})
    save_history(out / "history.npz", hist)
    (out / "metrics.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
