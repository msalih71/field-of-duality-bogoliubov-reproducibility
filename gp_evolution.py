"""Time-dependent Gross-Pitaevskii evolution of the stationary backgrounds.

    i d_t psi = [ -(1/2) d_xx + V(x) + g(x) |psi|^2 ] psi ,   hbar = m = 1, mu = 0,

with V from the inverse construction, so that psi0 = sqrt(n0) exp(i theta0) is
static.  Sixth-order finite differences in x, classical RK4 in t.  Outside a
central window the field is relaxed towards psi0 (sponge) and the end points
are held at psi0, so that outgoing waves leave the box.

A small localised perturbation is added and the relative density deviation
delta(t) = max_window | |psi|^2/n0 - 1 |  (insensitive to a global phase) is recorded.  On a stable background delta does not grow; on the two-horizon
control it must grow at the rate predicted by the independent complex-frequency
search (src/stability_scan.py), which cross-validates both calculations.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fod_scattering import Background

np.seterr(over="ignore")


def run(bg, Lx=90.0, dx=0.08, dt=0.003, T=300.0, eps=1e-6, sponge=20.0, x0=0.0,
        sample=1.0):
    x = np.arange(-Lx, Lx + dx / 2, dx)
    n0 = bg.n(x)
    # phase: theta0' = j / n0, integrated with Simpson on a refined grid
    xf = np.linspace(-Lx, Lx, 8 * (x.size - 1) + 1)
    f = bg.j / bg.n(xf)
    h = xf[1] - xf[0]
    cum = np.concatenate([[0.0], np.cumsum((f[:-2:2] + 4 * f[1:-1:2] + f[2::2]) * h / 3)])
    theta0 = cum[::4]                                   # Simpson panels of width 2h
    theta0 = theta0 - np.interp(0.0, x, theta0)
    psi0 = np.sqrt(n0) * np.exp(1j * theta0)
    V = bg.potential(x, mu=0.0)
    g = bg.c2(x) / n0
    gam = np.zeros_like(x)
    edge = np.abs(x) > Lx - sponge
    s = (np.abs(x[edge]) - (Lx - sponge)) / sponge
    gam[edge] = 1.0 * s * s * (3 - 2 * s)
    win = np.abs(x) < Lx - sponge - 5.0

    def rhs(p):
        lap = np.zeros_like(p)
        lap[3:-3] = (2 * p[6:] - 27 * p[5:-1] + 270 * p[4:-2] - 490 * p[3:-3]
                     + 270 * p[2:-4] - 27 * p[1:-5] + 2 * p[:-6]) / (180 * dx * dx)
        out = -1j * (-0.5 * lap + (V + g * np.abs(p) ** 2) * p) - gam * (p - psi0)
        out[:3] = 0.0
        out[-3:] = 0.0
        return out

    resid_analytic_V = float(np.max(np.abs(rhs(psi0))[win]))
    # Make psi0 stationary for the *discrete* operator up to current non-conservation:
    # absorb the O(dx^4) real part of the truncation error into the potential.
    lap0 = np.zeros_like(psi0)
    lap0[3:-3] = (2 * psi0[6:] - 27 * psi0[5:-1] + 270 * psi0[4:-2] - 490 * psi0[3:-3]
                  + 270 * psi0[2:-4] - 27 * psi0[1:-5] + 2 * psi0[:-6]) / (180 * dx * dx)
    V = V.copy()
    V[3:-3] = np.real(0.5 * lap0[3:-3] / psi0[3:-3]) - g[3:-3] * n0[3:-3]
    resid0 = float(np.max(np.abs(rhs(psi0))[win]))      # remaining (imaginary) residual
    psi = psi0 * (1.0 + eps * np.exp(-((x - x0) ** 2) / 4.0))
    nsteps = int(round(T / dt)); every = int(round(sample / dt))
    ts, ds = [0.0], [float(np.max(np.abs(np.abs(psi[win]) ** 2 / n0[win] - 1)))]
    for k in range(1, nsteps + 1):
        k1 = rhs(psi); k2 = rhs(psi + 0.5 * dt * k1)
        k3 = rhs(psi + 0.5 * dt * k2); k4 = rhs(psi + dt * k3)
        psi = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if k % every == 0:
            ts.append(k * dt); ds.append(float(np.max(np.abs(np.abs(psi[win]) ** 2 / n0[win] - 1))))
            if not np.isfinite(ds[-1]) or ds[-1] > 0.3:
                break
    return np.array(ts), np.array(ds), (resid_analytic_V, resid0)


def growth_rate(ts, ds, lo, hi):
    """Least-squares slope of log(delta) where lo < delta < hi (linear regime)."""
    m = (ds > lo) & (ds < hi)
    idx = np.where(m)[0]
    if idx.size < 20:
        return None
    # use the last contiguous run
    cut = np.where(np.diff(idx) > 1)[0]
    idx = idx[cut[-1] + 1:] if cut.size else idx
    if idx.size < 20:
        return None
    p = np.polyfit(ts[idx], np.log(ds[idx]), 1)
    return float(p[0]), float(ts[idx][0]), float(ts[idx][-1])


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgA = dict(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0, g=1.0)
    cfgB = dict(kind="const_c", nL=1.0, nR=0.4, j=0.6, width=1.0, c2L=1.0)
    pred = json.load(open(os.path.join(ROOT, "results", "stability.json")))
    out, series = {}, {}
    cases = [("A_constant_g", cfgA, 400.0), ("B_constant_c", cfgB, 400.0)]
    for d in (8.0, 14.0):
        c = dict(cfgA); c["pair_half_length"] = d
        cases.append((f"control_pair_half_length_{d:g}", c, 600.0))
    for name, cfg, T in cases:
        t0 = time.time()
        ts, ds, r0 = run(Background(**cfg), T=T)
        rec = dict(T_reached=float(ts[-1]), delta_initial=float(ds[0]),
                   delta_max=float(ds.max()), delta_final=float(ds[-1]),
                   delta_last_quarter_max=float(ds[3 * len(ds) // 4:].max()),
                   residual_with_analytic_V=r0[0], residual_after_discrete_correction=r0[1], seconds=time.time() - t0)
        if name.startswith("control"):
            gr = growth_rate(ts, ds, 1e-4, 3e-2)
            modes = pred[name]["unstable_modes"]
            top = max(modes, key=lambda z: z["omega"][1])
            rec["predicted_fastest_growth_rate"] = top["omega"][1]
            rec["predicted_fastest_re_omega"] = top["omega"][0]
            if gr:
                rec["fitted_growth_rate"], rec["fit_t0"], rec["fit_t1"] = gr
                rec["ratio_fitted_to_predicted"] = gr[0] / top["omega"][1]
        out[name] = rec; series[name + "_t"] = ts; series[name + "_d"] = ds
        print(name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()})
    json.dump(out, open(os.path.join(ROOT, "results", "gp_evolution.json"), "w"), indent=1)
    np.savez(os.path.join(ROOT, "results", "gp_evolution_series.npz"), **series)
