"""Independent re-computation of the Evans determinant and of the zero count.

Nothing is imported from fod_scattering.py except the definition of the background
profile (class Background); nothing is imported from evans_winding.py or
stability_scan.py.  The computation differs from src/evans_winding.py in:

  * variables    Nambu amplitudes of delta psi with the condensate phase gauged out,
                     delta psi = e^{i theta0} ( Ut e^{-iwt} + conj(Vt) e^{i conj(w) t} ),
                 not the relative field (u, vartheta);
  * equations    the textbook BdG operator with the external potential V(x) of the
                 inverse construction (hbar = m = 1, mu = 0):
                     Ut'' = (2(h - w) + v^2 - i v') Ut - 2 i v Ut' + 2 g n Vt
                     Vt'' = (2(h + w) + v^2 + i v') Vt + 2 i v Vt' + 2 g n Ut ,
                 h = V + 2 g n;
  * asymptotics  the decaying subspace is taken from the eigenvalues of the 4x4
                 asymptotic matrix (numpy.linalg.eigvals), not from the roots of the
                 dispersion polynomial, and the eigenvectors from the kernel of the
                 2x2 quadratic pencil of the Nambu system;
  * transport    the 6-component compound system for 2-forms, integrated with the
                 adaptive DOP853 scheme of scipy.integrate.solve_ivp (no fixed step,
                 no re-normalisation, no antisymmetrisation step); the asymptotic
                 growth rate is removed analytically;
  * contour      its own sampling and refinement, accepted only if two phase
                 tolerances give the same count.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(__file__))
from fod_scattering import Background          # background profile only

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _compound_tensor():
    """T[p, q, i, k]: the matrix of W -> A W + W A^T on 2-forms is C = T . A."""
    T = np.zeros((6, 6, 4, 4))
    for q, (a, b) in enumerate(PAIRS):
        E = np.zeros((4, 4)); E[a, b], E[b, a] = 1.0, -1.0
        for i in range(4):
            for k in range(4):
                A = np.zeros((4, 4)); A[i, k] = 1.0
                R = A @ E + E @ A.T
                for p, (c, d) in enumerate(PAIRS):
                    T[p, q, i, k] = R[c, d]
    return T


T = _compound_tensor()
A1 = np.zeros((4, 4)); A1[1, 0], A1[3, 2] = -2.0, 2.0
C1 = np.einsum("pqik,ik->pq", T, A1)


def A0(bg, x):
    n, v, dv, gn = float(bg.n(x)), float(bg.v(x)), float(bg.dv(x)), float(bg.c2(x))
    h = float(bg.potential(x, mu=0.0)) + 2.0 * gn
    A = np.zeros((4, 4), dtype=complex)
    A[0, 1] = 1.0
    A[1] = [2 * h + v * v - 1j * dv, -2j * v, 2 * gn, 0.0]
    A[2, 3] = 1.0
    A[3] = [2 * gn, 0.0, 2 * h + v * v + 1j * dv, 2j * v]
    return A


def start(bg, x0, om, side):
    """Analytic 2-form of the decaying subspace at x0, and the sum of its two rates."""
    v, gn = float(bg.v(x0)), float(bg.c2(x0))
    M0 = A0(bg, x0)
    W = np.zeros((om.size, 6), dtype=complex); s = np.zeros(om.size, dtype=complex)
    for j, w in enumerate(om):
        mu = np.linalg.eigvals(M0 + w * A1)
        keep = mu[mu.real > 0] if side == "L" else mu[mu.real < 0]
        if keep.size != 2:
            W[j] = np.nan; continue
        vec = []
        for m_ in keep:
            a, b = 2 * gn, m_ * m_ + 2j * v * m_ - 2 * gn + 2 * w      # kernel of the pencil
            vec.append(np.array([a, m_ * a, b, m_ * b]))
        y1, y2 = vec
        W[j] = [(y1[c] * y2[d] - y1[d] * y2[c]) / (keep[0] - keep[1]) for c, d in PAIRS]
        s[j] = keep.sum()
    return W, s


def D(bg, om, X0, rtol=1e-10, atol=1e-13):
    """Evans determinant at the complex frequencies om (1-d array)."""
    om = np.asarray(om, dtype=complex)
    out = []
    for side, x0 in (("L", -X0), ("R", X0)):
        W0, s = start(bg, x0, om, side)
        bad = np.isnan(W0[:, 0]); W0[bad] = 0.0

        def rhs(x, y):
            Y = y.reshape(-1, 6)
            C0 = np.einsum("pqik,ik->pq", T, A0(bg, x))
            return (Y @ C0.T + om[:, None] * (Y @ C1.T) - s[:, None] * Y).ravel()

        sol = solve_ivp(rhs, (x0, 0.0), W0.ravel(), method="DOP853", rtol=rtol, atol=atol)
        assert sol.success
        Wf = sol.y[:, -1].reshape(-1, 6); Wf[bad] = np.nan
        out.append(Wf)
    L, R = out
    return L[:, 0] * R[:, 5] - L[:, 1] * R[:, 4] + L[:, 2] * R[:, 3] + L[:, 3] * R[:, 2] - L[:, 4] * R[:, 1] + L[:, 5] * R[:, 0]


def cutoff(bg, X=40.0):
    """Largest cutoff frequency over the profile (maximum of v k - sqrt(c^2 k^2 + k^4/4))."""
    best = 0.0
    for x in np.linspace(-X, X, 801):
        v, c2 = float(bg.v(x)), float(bg.c2(x))
        if v * v > c2:
            r = minimize_scalar(lambda k: -(v * k - np.sqrt(c2 * k * k + k ** 4 / 4)),
                                bounds=(1e-6, 2 * np.sqrt(v * v - c2)), method="bounded",
                                options=dict(xatol=1e-12))
            best = max(best, -r.fun)
    return best


def count_zeros(bg, X0, re_max, im_lo, im_hi, special, tol, ds=0.02, max_rounds=60):
    """Number of zeros of D inside [-re_max, re_max] x [im_lo, im_hi] (argument principle)."""
    corners = [complex(-re_max, im_lo), complex(re_max, im_lo), complex(re_max, im_hi), complex(-re_max, im_hi)]
    sp_ = np.asarray(special, dtype=complex)
    z = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        Ls, t = abs(b - a), 0.0
        while t < Ls:
            w = a + (b - a) * t / Ls
            z.append(w)
            t += min(ds, 0.1 * float(np.abs(w - sp_).min()))
    z = np.array(z); d = D(bg, z, X0)
    for rounds in range(max_rounds):
        step = np.angle(np.roll(d, -1) / d)
        bad = np.where(np.abs(step) > tol)[0]
        if bad.size == 0:
            break
        zm = 0.5 * (z[bad] + np.roll(z, -1)[bad]); dm = D(bg, zm, X0)
        z = np.insert(z, bad + 1, zm); d = np.insert(d, bad + 1, dm)
    step = np.angle(np.roll(d, -1) / d)
    return dict(count=float(step.sum() / (2 * np.pi)), points=int(z.size), converged=bool(np.abs(step).max() <= tol),
                max_phase_step=float(np.abs(step).max()), n_nan=int(np.isnan(d).sum()))


def refine_zero(bg, X0, w0, steps=30):
    """Secant iteration on the analytic determinant, started next to w0."""
    a, b = complex(w0), complex(w0) * (1 + 1e-3) + 1e-5
    fa, fb = D(bg, [a], X0)[0], D(bg, [b], X0)[0]
    for _ in range(steps):
        if fb == fa:
            break
        c = b - fb * (b - a) / (fb - fa)
        a, fa, b = b, fb, c
        fb = D(bg, [b], X0)[0]
        if abs(b - a) < 1e-12:
            break
    return b


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgA = dict(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0, g=1.0)
    cfgB = dict(kind="const_c", nL=1.0, nR=0.4, j=0.6, width=1.0, c2L=1.0)
    out = {}
    scan = json.load(open(os.path.join(ROOT, "results", "stability.json")))
    ref = json.load(open(os.path.join(ROOT, "results", "evans_winding.json")))
    # ---- two-horizon controls: counts, and positions of the zeros
    for d in (8.0, 14.0):
        cfg = dict(cfgA); cfg["pair_half_length"] = d
        bg = Background(**cfg); X0 = d + 14.0; wc = cutoff(bg); spc = (0.0, wc, -wc)
        key = f"control_pair_half_length_{d:g}"; t0 = time.time(); rec = dict(counts=[])
        for lo in (0.02, 0.005, 0.002, 1e-3, 0.0006):
            c1 = count_zeros(bg, X0, 1.0, lo, 0.3, spc, tol=0.3)
            c2 = count_zeros(bg, X0, 1.0, lo, 0.3, spc, tol=0.15)
            rec["counts"].append(dict(im_lo=lo, count=c1["count"], count_tighter_tolerance=c2["count"],
                                      converged=c1["converged"] and c2["converged"], points=c2["points"]))
        zs = []
        for z in scan[key]["unstable_modes"]:
            w0 = complex(*z["omega"])
            if w0.imag > 5e-4:
                w = refine_zero(bg, X0, w0)
                zs.append(dict(scan=[w0.real, w0.imag], independent=[w.real, w.imag], abs_difference=abs(w - w0)))
        rec.update(zeros=zs, seconds=time.time() - t0)
        out[key] = rec
        print(key, [(c["im_lo"], round(c["count"], 6), round(c["count_tighter_tolerance"], 6), c["converged"]) for c in rec["counts"]],
              "max |dw| = %.2e" % max(z["abs_difference"] for z in zs), "t=%.0fs" % rec["seconds"])
    # ---- single-horizon backgrounds over the a-priori region
    for name, cfg in (("A_constant_g", cfgA), ("B_constant_c", cfgB)):
        bg = Background(**cfg); xs = np.linspace(-40, 40, 8001)
        vmax, bmax = float(bg.v(xs).max()), float(bg.c2(xs).max())
        box_re, box_im = 1.01 * (4 * vmax ** 2 + bmax), 1.05 * bmax
        wc = cutoff(bg); spc = (0.0, wc, -wc); t0 = time.time(); rec = dict(rectangle_re=box_re, rectangle_im_hi=box_im, cutoff=wc, counts=[])
        for lo in (1e-3, 1e-4, 1e-5, 1e-6):
            c1 = count_zeros(bg, 14.0, box_re, lo, box_im, spc, tol=0.3)
            c2 = count_zeros(bg, 14.0, box_re, lo, box_im, spc, tol=0.15)
            rec["counts"].append(dict(im_lo=lo, count=c1["count"], count_tighter_tolerance=c2["count"],
                                      converged=c1["converged"] and c2["converged"], points=c2["points"],
                                      n_nan=c1["n_nan"] + c2["n_nan"]))
        c3 = count_zeros(bg, 18.0, box_re, 1e-3, box_im, spc, tol=0.3)      # different matching distance
        rec["count_other_matching_distance"] = c3["count"]
        rec["reference_counts_evans_winding"] = [ref[name]["winding"]] + [c["winding"] for c in ref[name]["counts_vs_lower_edge"]]
        rec["seconds"] = time.time() - t0
        out[name] = rec
        print(name, [(c["im_lo"], round(c["count"], 6), round(c["count_tighter_tolerance"], 6), c["converged"], c["n_nan"]) for c in rec["counts"]],
              "X0=18:", round(c3["count"], 6), "t=%.0fs" % rec["seconds"])
    json.dump(out, open(os.path.join(ROOT, "results", "evans_independent.json"), "w"), indent=1)
