"""Makes the figures and videos for one case. Every number in them is read from data/ or results/,
I do not type any value into a plot by hand.

    python scripts/reporting/make_media.py --case venturi --what reference verification sensors     # needs only OpenFOAM data
    python scripts/reporting/make_media.py --case venturi --what comparison walls loss video        # needs the trained runs
    python scripts/reporting/make_media.py --case cavity --what all

Output: results/figures/<case>_*.png and results/videos/<case>_*.gif / .mp4
"""
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.tri import LinearTriInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so "from pinn..." works when I run this file directly

from pinn import plotting as P
from pinn.data_utils import centerline_u, centerline_v, grid_view, load_reference, load_sensors, try_load_ghia
from pinn.evaluate import shift_pressure
from pinn.geometry import get_geometry

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["cavity", "venturi"], required=True)
ap.add_argument("--what", nargs="+", default=["all"])
ap.add_argument("--sub", default="reduced", help="which trained runs to use: reduced or full")
a = ap.parse_args()
# --what all makes everything, otherwise only the parts I list
want = lambda k: "all" in a.what or k in a.what

P.apply_style()
G = get_geometry(a.case)
REF = load_reference(G.reference_path())
RES = ROOT / "results" / a.case
FIG, VID = ROOT / "results" / "figures", ROOT / "results" / "videos"
TRI = P.triangulation(REF["x"], REF["y"], G)
SPEED = np.hypot(REF["u"], REF["v"])
# the run I show in the comparison figures and videos, same as in run_experiments.py
REP = {"cavity": (50, 1.0, 0), "venturi": (200, 1.0, 0)}[a.case]
tag = lambda m: f"{m}_N{REP[0]}_noise{REP[1]:g}_seed{REP[2]}"
FILE_NAME = {"speed": "velocity", "p": "pressure"}
CASE_LABEL = {"cavity": "Lid-driven cavity, Re = 100", "venturi": "2D Venturi passage, Re = 100"}[a.case]


def load_pred(model):
    f = RES / "predictions" / a.sub / f"{tag(model)}.npz"
    return np.load(f) if f.exists() else None


def fig_size(ncols=1, nrows=1, panel_w=None):
    (x0, x1), (y0, y1) = G.bounds
    asp = (y1 - y0) / (x1 - x0)
    w = panel_w or (10.5 if a.case == "venturi" else 3.4)
    return (w * ncols + (1.4 if a.case == "venturi" else 0.9), (w * asp + 0.55) * nrows + 0.9)


# --------------------------------------------------------------- reference overview
def reference():
    p, i = REF["p"], int(np.argmin(REF["p"]))
    if a.case == "venturi":
        fig, axs = plt.subplots(3, 2, figsize=(11, 6.3), height_ratios=[1, 1, 1.9], width_ratios=[1, 0.018],
                                constrained_layout=True)
        ax, cax = axs[:, 0], axs[:, 1]
        cax[2].set_axis_off()
        m = P.draw_field(ax[0], TRI, SPEED, P.CM_SPEED, vmin=0, geom=G, title="Velocity magnitude and streamlines  (flow left → right)")
        seeds = np.c_[np.full(15, 0.08), np.linspace(-0.46, 0.46, 15)]
        P.streamlines(ax[0], TRI, REF["u"], REF["v"], G, lw=0.6, seeds=seeds)
        P.colorbar(fig, m, ax[0], "|U|  (U_in = 1)", cax=cax[0])
        m = P.draw_field(ax[1], TRI, p, P.CM_P, norm=P.pressure_norm(p), geom=G,
                         title="Kinematic pressure p/ρ  (outlet p = 0)  —  never given to the PINN as data")
        ax[1].plot(REF["x"][i], REF["y"][i], "o", ms=7, mfc="none", mec=P.C["ref"], mew=1.6)
        ax[1].annotate(f"p_min = {p[i]:.3f}\nx = {REF['x'][i]:.2f}", (REF["x"][i], REF["y"][i]), (REF["x"][i] + .8, .05),
                       fontsize=9, arrowprops=dict(arrowstyle="-", color=P.C["ref"], lw=.8), va="center")
        P.colorbar(fig, m, ax[1], "p/ρ", cax=cax[1])
        xs = np.unique(np.round(REF["x"], 9))
        pw = [p[(np.isclose(REF["x"], x)) & (REF["y"] == REF["y"][np.isclose(REF["x"], x)].min())][0] for x in xs]
        ax[2].plot(xs, pw, color=P.C["ref"])
        ax[2].plot(REF["x"][i], p[i], "o", color=P.C["pinn"], zorder=3)
        ax[2].axhline(0, color=P.C["muted"], lw=.8, ls=":")
        ax[2].set(xlabel="x", ylabel="wall pressure p/ρ", xlim=(0, 12), title="Pressure along the lower wall")
        ax[2].annotate("lowest wall pressure,\nright after the throat", (REF["x"][i], p[i]), (7.2, p[i] + 1.2),
                       fontsize=9, arrowprops=dict(arrowstyle="-", color=P.C["muted"], lw=.8))
        fig.suptitle(f"OpenFOAM reference — {CASE_LABEL} (steady laminar simpleFoam, {len(REF['x']):,} cells)",
                     x=0.01, ha="left", fontsize=12, fontweight="bold")
    else:
        fig, ax = plt.subplots(1, 3, figsize=(11.2, 4.1), constrained_layout=True)
        m = P.draw_field(ax[0], TRI, SPEED, P.CM_SPEED, vmin=0, geom=G, title="Velocity magnitude + streamlines")
        P.streamlines(ax[0], TRI, REF["u"], REF["v"], G, density=0.85, lw=0.6)
        P.colorbar(fig, m, ax[0], "|U|", orientation="horizontal", fraction=.06)
        pp = shift_pressure(p)
        m = P.draw_field(ax[1], TRI, pp, P.CM_P, norm=P.pressure_norm(pp, clip=(2, 98)), geom=G,
                         title="Pressure (mean-shifted)")
        P.colorbar(fig, m, ax[1], "p − mean(p)   (colour scale: 2nd–98th percentile)", orientation="horizontal", fraction=.06)
        # vorticity on the regular cavity grid, with finite differences
        xs, ys, g = grid_view(REF)
        w = np.gradient(g["v"], xs, axis=1) - np.gradient(g["u"], ys, axis=0)
        mm = ax[2].imshow(w, origin="lower", extent=(0, 1, 0, 1), cmap=P.CM_SIGNED, norm=P.pressure_norm(w, clip=(2, 98)), interpolation="bilinear")
        ax[2].plot(*G.outline().T, color=P.C["ref"], lw=1.1)
        ax[2].set_title("Vorticity  ω = ∂v/∂x − ∂u/∂y")
        ax[2].set_axis_off()
        P.colorbar(fig, mm, ax[2], "ω   (colour scale: 2nd–98th percentile)", orientation="horizontal", fraction=.06)
        fig.suptitle(f"OpenFOAM reference — {CASE_LABEL} (128×128 cells)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    P.save(fig, FIG / f"{a.case}_reference.png")


# ------------------------------------------------------------------- verification
def verification():
    if a.case != "venturi":
        return  # the cavity verification figure (against Ghia) is made by scripts/verification/verify_cavity.py
    ver = pd.read_csv(ROOT / "results" / "venturi" / "verification.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    cols = [P.C["muted"], P.C["data"], P.C["pinn"]]
    yy = np.linspace(-.5, .5, 200)
    ax[0].plot(1.5 * (1 - (2 * yy) ** 2), yy, color=P.C["ref"], label="plane Poiseuille (analytic)")
    for k, c in zip(ver.k, cols):
        d = load_reference(ROOT / "data" / "venturi" / f"openfoam_k{k}.npz")
        xc = np.unique(np.round(d["x"], 9))
        xc = xc[np.abs(xc - 1.0).argmin()]
        m = np.isclose(d["x"], xc)
        o = np.argsort(d["y"][m])
        ax[0].plot(d["u"][m][o][::max(1, int(2 ** (2 - np.log2(k))) )], d["y"][m][o][::max(1, int(2 ** (2 - np.log2(k))))], "o", ms=3.2, color=c,
                   label=f"OpenFOAM, {int(ver.cells[ver.k == k].iloc[0]):,} cells")
    ax[0].set(xlabel="u at x = 1 (inlet section)", ylabel="y", title="(a) Developed inlet profile")
    ax[0].legend(fontsize=7.5, loc="center left")
    d = REF
    xs = np.unique(np.round(d["x"], 9))
    pm = np.array([d["p"][np.isclose(d["x"], x)].mean() for x in xs])
    ax[1].plot(xs, pm, color=P.C["ref"], label="cross-section mean p")
    x1 = np.array([0.25, 1.75])
    p1 = np.interp(1.0, xs, pm)
    slope = ver.dpdx_poiseuille.iloc[0]
    ax[1].plot(x1, p1 + slope * (x1 - 1.0), color=P.C["data"], lw=2.4, alpha=.9, label=f"analytic slope {slope:.3f}")
    ax[1].set(xlabel="x", ylabel="p/ρ", title="(b) Pressure drop along the passage")
    ax[1].legend(fontsize=8)
    ax[2].semilogx(ver.cells, ver.p_min, "o-", color=P.C["pinn"])
    for c, v in zip(ver.cells, ver.p_min):
        ax[2].annotate(f"{v:.4f}", (c, v), (0, 8), textcoords="offset points", ha="center", fontsize=8.5)
    ax[2].set(xlabel="cells", ylabel="p_min", title="(c) Minimum pressure vs mesh")
    ax[2].margins(x=.25, y=.4)
    chg = abs(ver.p_min.iloc[-1] - ver.p_min.iloc[0]) / abs(ver.p_min.iloc[-1]) * 100
    ax[2].text(0.04, 0.05, f"axis zoomed: coarsest → finest\nchanges p_min by {chg:.2f} %", transform=ax[2].transAxes,
               fontsize=8.5, color=P.C["text2"])
    fig.suptitle("Venturi verification: analytic Poiseuille gradient and mesh convergence", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    P.save(fig, FIG / "venturi_verification.png")


# ------------------------------------------------------------------------ sensors
def sensors():
    wide = a.case == "venturi"
    if wide:
        fig = plt.figure(figsize=(10, 6.6), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.8])
        axf = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
        axp = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    else:
        fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
        axf, axp = list(axs[:2]), list(axs[2:])
    for axi, N in zip(axf, (50, 200)):
        s = load_sensors(a.case, N, 0, 0)
        P.draw_field(axi, TRI, SPEED, P.CM_SPEED, vmin=0, geom=G, title=f"N = {N} sensors")
        axi.plot(s["x"], s["y"], "o", ms=4.2 if wide else 3.4, mfc=P.C["aux"], mec="white", mew=.6)
    for axi, nz in zip(axp, (1, 5)):
        s = load_sensors(a.case, 200, nz, 0)
        axi.plot(s["u_clean"], s["u"], ".", ms=4, color=P.C["pinn"], alpha=.7)
        lim = [min(s["u_clean"].min(), s["u"].min()), max(s["u_clean"].max(), s["u"].max())]
        axi.plot(lim, lim, color=P.C["muted"], lw=.9)
        axi.set(xlabel="true u", ylabel="measured u", title=f"{nz} % noise (N = 200)")
        axi.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"{CASE_LABEL}: synthetic sensor data — random cell centres, Gaussian noise, only (u, v) given, never p",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    P.save(fig, FIG / f"{a.case}_sensors.png")


# --------------------------------------------------------------------- comparison
def _field(q, d):
    if q == "speed": return np.hypot(d["u"], d["v"])
    if q == "p": return shift_pressure(d["p"])
    return d[q]


def comparison():
    preds = {m: load_pred(m) for m in ("pinn", "data_only")}
    if any(v is None for v in preds.values()):
        print("comparison: trained representative runs not found, skipping")
        return
    quants = ["u", "v", "p"] if a.case == "cavity" else ["speed", "p"]
    names = {"u": "u", "v": "v", "p": "p (mean-shifted)", "speed": "|U|"}
    if a.case == "venturi":
        for q in quants:
            _stack_fig(q, names, preds)
    else:
        _grid_fig(quants, names, preds)


def _vel_err(q, m, d):  # absolute error of whatever is plotted (for speed I use the vector error)
    if q == "speed":
        return np.hypot(d["u"] - REF["u"], d["v"] - REF["v"])
    return np.abs(_field(q, d) - _field(q, REF))


def _stack_fig(q, names, preds):
    ref = _field(q, REF)
    norm = P.pressure_norm(ref) if q == "p" else Normalize(0, ref.max())
    cmap = P.CM_P if q == "p" else P.CM_SPEED
    errs = {m: _vel_err(q, m, preds[m]) for m in preds}
    emax = max(e.max() for e in errs.values()) * 0.8
    fig, ax = plt.subplots(5, 1, figsize=(11, 8.2), constrained_layout=True)
    rows = [("OpenFOAM", ref, cmap, norm), ("PINN", _field(q, preds["pinn"]), cmap, norm),
            ("Data-only network", _field(q, preds["data_only"]), cmap, norm),
            (f"|error| PINN  (max {errs['pinn'].max():.3g}, own colour scale)", errs["pinn"], P.CM_ERR, Normalize(0, errs["pinn"].max())),
            (f"|error| Data-only  (max {errs['data_only'].max():.3g}, own colour scale)", errs["data_only"], P.CM_ERR, Normalize(0, errs["data_only"].max()))]
    for k, (axi, (t, z, cm, nm)) in enumerate(zip(ax, rows)):
        m = P.draw_field(axi, TRI, z, cm, norm=nm, geom=G, title=t)
        if k in (0, 3, 4):
            P.colorbar(fig, m, axi, names[q] if k == 0 else "abs. error", fraction=.025)
    n, nz, seed = REP
    fig.suptitle(f"{CASE_LABEL}: {names[q]} — N = {n} sensors, {nz:g} % noise, seed {seed}"
                 + ("  (reduced CPU protocol)" if a.sub == "reduced" else ""), x=0.01, ha="left", fontsize=12, fontweight="bold")
    P.save(fig, FIG / f"{a.case}_{FILE_NAME[q]}_comparison.png")


def _grid_fig(quants, names, preds):
    fig, ax = plt.subplots(5, len(quants), figsize=(3.5 * len(quants) + .8, 3.45 * 5 / 1.25), constrained_layout=True)
    for j, q in enumerate(quants):
        ref = _field(q, REF)
        norm = P.pressure_norm(ref, clip=(2, 98)) if q == "p" else (P.pressure_norm(ref) if q != "speed" else Normalize(0, ref.max()))
        errs = {m: _vel_err(q, m, preds[m]) for m in preds}
        emax = max(e.max() for e in errs.values()) * 0.8
        rows = [("OpenFOAM", ref, P.CM_SIGNED, norm), ("PINN", _field(q, preds["pinn"]), P.CM_SIGNED, norm),
                ("Data-only", _field(q, preds["data_only"]), P.CM_SIGNED, norm),
                (f"|error| PINN (max {errs['pinn'].max():.2g})", errs["pinn"], P.CM_ERR, Normalize(0, errs["pinn"].max())),
                (f"|error| Data-only (max {errs['data_only'].max():.2g})", errs["data_only"], P.CM_ERR, Normalize(0, errs["data_only"].max()))]
        for i, (t, z, cm, nm) in enumerate(rows):
            m = P.draw_field(ax[i, j], TRI, z, cm, norm=nm, geom=G, title=f"{t}: {names[q]}")
            if i in (0, 3, 4):
                P.colorbar(fig, m, ax[i, j], "", orientation="horizontal", fraction=.05, aspect=22)
    n, nz, seed = REP
    fig.suptitle(f"{CASE_LABEL}: N = {n} sensors, {nz:g} % noise, seed {seed}"
                 + ("  (reduced CPU protocol)" if a.sub == "reduced" else ""), x=0.01, ha="left", fontsize=12, fontweight="bold")
    P.save(fig, FIG / f"{a.case}_comparison.png")


# --------------------------------------------------------------- profiles / walls
def walls():
    preds = {m: load_pred(m) for m in ("pinn", "data_only")}
    if any(v is None for v in preds.values()):
        return
    if a.case == "venturi":
        xs = np.unique(np.round(REF["x"], 9))
        sel = lambda d: np.array([np.where(np.isclose(REF["x"], x) & (REF["y"] == REF["y"][np.isclose(REF["x"], x)].min()))[0][0] for x in xs])
        idx = sel(REF)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True, width_ratios=[1.5, 1])
        ax[0].plot(xs, REF["p"][idx], color=P.C["ref"], label="OpenFOAM")
        ax[0].plot(xs, preds["pinn"]["p"][idx], color=P.C["pinn"], ls="--", label="PINN")
        ax[0].axhline(0, color=P.C["muted"], lw=.8, ls=":")
        ip = int(np.argmin(REF["p"]))
        ax[0].plot(REF["x"][ip], REF["p"][ip], "o", mfc="none", mec=P.C["ref"], ms=8)
        ax[0].set(xlabel="x", ylabel="p/ρ", title="Lower-wall pressure (outlet-anchored)", xlim=(0, 12))
        ax[0].legend()
        zoom = (xs > 3) & (xs < 8)
        ax[1].plot(xs[zoom], REF["p"][idx][zoom], color=P.C["ref"])
        ax[1].plot(xs[zoom], preds["pinn"]["p"][idx][zoom], color=P.C["pinn"], ls="--")
        ax[1].set(xlabel="x", title="Zoom: throat exit")
        fig.text(.01, -.04, "Data-only network omitted: it is never trained on pressure, so its absolute p is undetermined.",
                 fontsize=8.5, color=P.C["text2"])
        P.save(fig, FIG / "venturi_wall_pressure.png")
    else:
        ghia = try_load_ghia()
        s = np.linspace(0, 1, 401)
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
        ax[0].plot(centerline_u(REF, s), s, color=P.C["ref"], label="OpenFOAM 128²")
        ax[1].plot(s, centerline_v(REF, s), color=P.C["ref"], label="OpenFOAM 128²")
        for m, c, ls in (("pinn", P.C["pinn"], "--"), ("data_only", P.C["data"], "-.")):
            d = dict(preds[m])
            ax[0].plot(centerline_u(d, s), s, color=c, ls=ls, label=P.LABEL[m])
            ax[1].plot(s, centerline_v(d, s), color=c, ls=ls, label=P.LABEL[m])
        if ghia:
            ax[0].plot(ghia["u"][1], ghia["u"][0], "o", mfc="none", mec=P.C["text2"], ms=5, label="Ghia et al. (1982)")
            ax[1].plot(ghia["v"][0], ghia["v"][1], "o", mfc="none", mec=P.C["text2"], ms=5, label="Ghia et al. (1982)")
        ax[0].set(xlabel="u", ylabel="y", title="u on x = 0.5")
        ax[1].set(xlabel="x", ylabel="v", title="v on y = 0.5")
        ax[0].legend(fontsize=8)
        P.save(fig, FIG / "cavity_centerlines.png")


# ------------------------------------------------------------------------- losses
def loss():
    fs = {m: RES / "runs" / a.sub / f"{tag(m)}_history.npz" for m in ("pinn", "data_only")}
    if not all(f.exists() for f in fs.values()):
        return
    H = {m: np.load(f) for m, f in fs.items()}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True, sharey=True)
    terms = [("data", "sensors (data)", P.C["aux"]), ("pde", "Navier–Stokes residual", P.C["pinn"]),
             ("bc", "boundary conditions", P.C["data"])] + ([("out", "outlet p = 0", P.C["muted"])] if a.case == "venturi" else [])
    for axi, m in zip(ax, ("pinn", "data_only")):
        h = H[m]
        for k, lab, c in terms:
            y = h[k]
            if np.isfinite(y).any():
                axi.semilogy(h["it"], y, color=c, lw=1.6, label=lab)
        axi.set(xlabel="iteration (right of dotted line: L-BFGS evaluations)", title=P.LABEL[m] if m == "pinn" else "Data-only (residual shown as diagnostic, not trained)")
        ph = np.where(h["phase"] == 1)[0]
        if len(ph): axi.axvline(h["it"][ph[0]], color=P.C["muted"], ls=":", lw=1)
        axi.margins(x=.02)
    ax[0].set_ylabel("loss term (unweighted)")
    hd, lb = ax[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="outside lower center", ncols=len(lb), fontsize=9)
    P.save(fig, FIG / f"{a.case}_loss_curves.png")


# -------------------------------------------------------------------------- video
def video(quantity):
    snaps = {}
    for m in ("pinn", "data_only"):
        f = RES / "snapshots" / a.sub / f"{tag(m)}.npz"
        if not f.exists():
            print("video: snapshots missing", f)
            return
        snaps[m] = np.load(f)
    hist = {m: np.load(RES / "runs" / a.sub / f"{tag(m)}_history.npz") for m in snaps}
    xy = snaps["pinn"]["xy"]
    ST = P.mtri.Triangulation(xy[:, 0], xy[:, 1])
    cen = np.c_[xy[ST.triangles, 0].mean(1), xy[ST.triangles, 1].mean(1)]
    ST.set_mask(~G.inside(cen))
    nfr = min(len(snaps["pinn"]["it"]), len(snaps["data_only"]["it"]))
    f_of = lambda uvp: np.hypot(uvp[:, 0], uvp[:, 1]) if quantity == "speed" else (uvp[:, 2] - uvp[:, 2].mean())
    # the OpenFOAM field on the same points as the snapshots, so the three panels match
    rf = REF["p"] - REF["p"].mean() if quantity == "p" else SPEED
    ref_s = LinearTriInterpolator(TRI, rf)(xy[:, 0], xy[:, 1]).filled(np.nan)
    ok = np.isfinite(ref_s)
    cmap = P.CM_P if quantity == "p" else P.CM_SPEED
    norm = P.pressure_norm(rf) if quantity == "p" else Normalize(0, rf.max())
    venturi = a.case == "venturi"
    if venturi:
        fig = plt.figure(figsize=(10, 8.4), constrained_layout=True)
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.5])
        axf = [fig.add_subplot(gs[i]) for i in range(3)]
        axl = fig.add_subplot(gs[3])
    else:
        fig = plt.figure(figsize=(11, 6.4), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])
        axf = [fig.add_subplot(gs[0, i]) for i in range(3)]
        axl = fig.add_subplot(gs[1, :])
    titles = ["OpenFOAM (reference)", "PINN", "Data-only network"]
    meshes = []
    for i, (axi, t) in enumerate(zip(axf, titles)):
        z = ref_s if i == 0 else f_of(snaps["pinn" if i == 1 else "data_only"]["uvp"][0])
        meshes.append(P.draw_field(axi, ST, np.nan_to_num(z), cmap, norm=norm, geom=G, title=t))
    P.colorbar(fig, meshes[0], axf, {"p": "p − mean(p)", "speed": "|U|"}[quantity], fraction=.03 if venturi else .02)
    for m, c in (("pinn", P.C["pinn"]), ("data_only", P.C["data"])):
        h = hist[m]
        axl.semilogy(h["it"], h["data"], color=c, label=f"{P.LABEL[m]} — sensor misfit")
    axl.semilogy(hist["pinn"]["it"], hist["pinn"]["pde"], color=P.C["pinn"], ls=":", label="PINN — Navier–Stokes residual")
    axl.set(xlabel="training iteration", ylabel="loss term")
    axl.legend(fontsize=8, ncols=3, loc="upper right")
    vl = axl.axvline(0, color=P.C["ref"], lw=1.2)
    ttl = fig.suptitle("", x=0.01, ha="left", fontsize=11, fontweight="bold")
    sub = " (reduced CPU protocol)" if a.sub == "reduced" else ""

    def update(k):
        for j, m in ((1, "pinn"), (2, "data_only")):
            meshes[j].set_array(np.nan_to_num(f_of(snaps[m]["uvp"][k])))
        it = int(snaps["pinn"]["it"][k])
        vl.set_xdata([it, it])
        ttl.set_text(f"{CASE_LABEL}: learning the {'pressure' if quantity == 'p' else 'velocity'} field "
                     f"from {REP[0]} noisy sensors\niteration {it}{sub}")
        return meshes + [vl]

    ani = FuncAnimation(fig, update, frames=nfr, interval=90, blit=False)
    VID.mkdir(parents=True, exist_ok=True)
    base = VID / f"{a.case}_training_{FILE_NAME[quantity]}"
    ani.save(str(base) + ".gif", writer=PillowWriter(fps=10), dpi=80)
    print("wrote", str(base) + ".gif")
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        from matplotlib.animation import FFMpegWriter
        ani.save(str(base) + ".mp4", writer=FFMpegWriter(fps=10, bitrate=2400), dpi=110)
        print("wrote", str(base) + ".mp4")
    except Exception as e:  # the mp4 needs ffmpeg, the gif is enough if it is missing
        print("mp4 skipped:", e)
    plt.close(fig)


for key, fn in [("reference", reference), ("verification", verification), ("sensors", sensors),
                ("comparison", comparison), ("walls", walls), ("loss", loss)]:
    if want(key):
        fn()
if want("video"):
    for q in ("p", "speed"):
        video(q)
