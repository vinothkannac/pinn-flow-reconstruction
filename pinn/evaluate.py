"""Error numbers: against OpenFOAM at all cell centres, and against Ghia on the centerlines."""
import numpy as np
import torch


def predict(net, xy, batch=20000):
    # (N, 2) numpy in, (N, 3) numpy out with columns u, v, p. The p still has an arbitrary constant.
    # I go through it in batches so a big mesh does not use too much memory.
    net.eval()
    p0 = next(net.parameters())
    outs = []
    for i in range(0, len(xy), batch):
        t = torch.as_tensor(xy[i:i + batch], dtype=p0.dtype, device=p0.device)
        with torch.no_grad():
            u, v, p = net.uvp(t)
        outs.append(torch.cat([u, v, p], 1).detach().cpu().numpy())
    net.train()
    return np.concatenate(outs, 0).astype(float)


def shift_pressure(p):
    # Pressure only matters up to a constant, so I subtract the mean. All cells have the same size,
    # so a plain mean is fine. I do this to both the PINN and the OpenFOAM pressure before computing an error.
    return p - p.mean()


def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def evaluate_fields(pred_uvp, ref):
    return {
        "rel_l2_u": rel_l2(pred_uvp[:, 0], ref["u"]),
        "rel_l2_v": rel_l2(pred_uvp[:, 1], ref["v"]),
        "rel_l2_p": rel_l2(shift_pressure(pred_uvp[:, 2]), shift_pressure(ref["p"])),
    }


def ghia_deviation(net, ghia):
    # Compare the network with the Ghia values by evaluating it at the Ghia coordinates
    # (u on x = 0.5, v on y = 0.5). Gives NaN if there is no Ghia file.
    keys = ["ghia_u_max", "ghia_u_rms", "ghia_v_max", "ghia_v_rms"]
    if ghia is None:
        return {k: float("nan") for k in keys}
    out = {}
    yq, uq = ghia["u"]
    xq, vq = ghia["v"]
    pu = predict(net, np.c_[np.full_like(yq, 0.5), yq])[:, 0]
    pv = predict(net, np.c_[xq, np.full_like(xq, 0.5)])[:, 1]
    for name, d in (("u", pu - uq), ("v", pv - vq)):
        out[f"ghia_{name}_max"] = float(np.max(np.abs(d)))
        out[f"ghia_{name}_rms"] = float(np.sqrt(np.mean(d ** 2)))
    return out


def venturi_pressure_metrics(pred_uvp, ref):
    # Compares the lowest pressure and where it is, since that is where cavitation would start.
    # Here I do NOT shift the pressure: p = 0 at the outlet in both the PINN and OpenFOAM.
    # The pressure is p/rho. Only makes sense for the PINN, the data-only network knows nothing about pressure.
    ip, ir = int(np.argmin(pred_uvp[:, 2])), int(np.argmin(ref["p"]))
    return {"p_min_pred": float(pred_uvp[ip, 2]), "p_min_ref": float(ref["p"][ir]),
            "p_min_abs_err": float(abs(pred_uvp[ip, 2] - ref["p"][ir])),
            "p_min_loc_err": float(np.hypot(ref["x"][ip] - ref["x"][ir], ref["y"][ip] - ref["y"][ir]))}
