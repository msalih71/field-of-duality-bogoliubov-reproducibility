"""Zero counting for the stability problem, and the real-axis solvability determinant.

(1) Analytic Evans function.  For Im(w) > 0 let k1, k2 be the two roots of the
    asymptotic dispersion relation whose plane waves decay away from the
    scatterer on a given side, and v(k) = (k^2, i k^3, -2i(w - v0 k), 2k(w - v0 k))
    the corresponding (un-normalised, analytic) solution vectors.  The 2-form
        W0 = [ v(k1) ^ v(k2) ] / (k1 - k2)
    is a symmetric function of the pair {k1, k2} and hence analytic in w.  It is
    transported to x = 0 with  W' = A W + W A^T  (the induced equation on
    2-forms), which is stable because the decaying subspace is the dominant one
    in the direction of integration.  Rescaling by positive real factors on the
    way does not change the argument.  The pairing
        D(w) = W_L ^ W_R
    is then (a positive multiple of) an analytic function of w in the upper
    half-plane whose zeros are exactly the unstable eigenfrequencies, so the
    number of zeros inside a closed contour equals the winding number of D.

(2) For real w the same construction with "outgoing or decaying" in place of
    "decaying" gives the determinant whose non-vanishing is equivalent to unique
    solvability of the scattering boundary-value problem.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fod_scattering import Background, A_relative, asymptotic_modes, omega_max

np.seterr(over="ignore", invalid="ignore")
WD = np.zeros((4, 4), dtype=complex); WD[1, 2] = -2j; WD[3, 0] = 2j      # dA/dw


def roots_batch(om, v, c2):
    n = om.size
    C = np.zeros((n, 4, 4), dtype=complex)
    # k^4 + 4(c2 - v^2) k^2 + 8 w v k - 4 w^2 = 0   (monic form of the quartic)
    C[:, 1, 0] = C[:, 2, 1] = C[:, 3, 2] = 1.0
    C[:, 0, 3] = 4 * om ** 2
    C[:, 1, 3] = -8 * om * v
    C[:, 2, 3] = -4 * (c2 - v * v)
    return np.linalg.eigvals(C)


def vec(k, om, v):
    return np.stack([k * k, 1j * k ** 3, -2j * (om - v * k), 2 * k * (om - v * k)], axis=-1)


def initial_wedge(om, v, c2, side):
    ks = roots_batch(om, v, c2)
    sel = (ks.imag < 0) if side == "L" else (ks.imag > 0)
    ok = sel.sum(axis=1) == 2
    idx = np.argsort(~sel, axis=1)[:, :2]               # the two selected roots first
    k1 = np.take_along_axis(ks, idx[:, :1], 1)[:, 0]
    k2 = np.take_along_axis(ks, idx[:, 1:2], 1)[:, 0]
    a, b = vec(k1, om, v), vec(k2, om, v)
    W = (a[:, :, None] * b[:, None, :] - b[:, :, None] * a[:, None, :]) / (k1 - k2)[:, None, None]
    return W, ok


class Evans:
    def __init__(self, bg, X0, h=0.02, renorm=10):
        self.bg, self.X0, self.h, self.renorm = bg, X0, h, renorm
        self.n = int(round(X0 / h))
        self.AL = A_relative(bg, np.linspace(-X0, 0.0, 2 * self.n + 1), 0.0)
        self.AR = A_relative(bg, np.linspace(X0, 0.0, 2 * self.n + 1), 0.0)

    def _transport(self, W, A, h, om):
        w = om[:, None, None] * WD[None]
        f = lambda Aw, W_: Aw @ W_ + W_ @ np.swapaxes(Aw, 1, 2)
        for i in range(self.n):
            a0, am, a1 = A[2 * i][None] + w, A[2 * i + 1][None] + w, A[2 * i + 2][None] + w
            k1 = f(a0, W); k2 = f(am, W + 0.5 * h * k1)
            k3 = f(am, W + 0.5 * h * k2); k4 = f(a1, W + h * k3)
            W = W + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # The equation W' = A W + W A^T also propagates symmetric tensors, whose
            # fastest mode grows twice as fast as the wanted 2-form; project them out.
            W = 0.5 * (W - np.swapaxes(W, 1, 2))
            if (i + 1) % self.renorm == 0:
                W = W / np.linalg.norm(W, axis=(1, 2))[:, None, None]     # positive factor
        return W / np.linalg.norm(W, axis=(1, 2))[:, None, None]

    def D(self, om):
        om = np.atleast_1d(np.asarray(om, dtype=complex))
        _, vL, c2L = self.bg.side("L"); _, vR, c2R = self.bg.side("R")
        WL, okL = initial_wedge(om, vL, c2L, "L")
        WR, okR = initial_wedge(om, vR, c2R, "R")
        WL = WL / np.linalg.norm(WL, axis=(1, 2))[:, None, None]
        WR = WR / np.linalg.norm(WR, axis=(1, 2))[:, None, None]
        # NOTE: the initial normalisation is a positive factor as well
        L = self._transport(WL, self.AL, self.h, om)
        R = self._transport(WR, self.AR, -self.h, om)
        D = (L[:, 0, 1] * R[:, 2, 3] - L[:, 0, 2] * R[:, 1, 3] + L[:, 0, 3] * R[:, 1, 2]
             + L[:, 1, 2] * R[:, 0, 3] - L[:, 1, 3] * R[:, 0, 2] + L[:, 2, 3] * R[:, 0, 1])
        D[~(okL & okR)] = np.nan
        return D


def winding(ev, re_max, im_lo, im_hi, ds=0.01, tol=0.5, max_iter=40, chunk=4000, special=(0.0,)):
    """Winding number of D around the rectangle [-re_max, re_max] x [im_lo, im_hi].

    D vanishes at w = 0 (neutral mode) and has branch points at the real cutoff
    frequencies, so its phase varies on the scale of the distance to those points.
    The initial mesh is therefore graded (spacing <= 0.2 x distance to the nearest
    special point), and the result is accepted only after a verification pass in
    which every interval is bisected and the two half-steps add up to the full step
    (a phase step that had been aliased by a multiple of 2 pi fails this test).
    """
    c = [complex(-re_max, im_lo), complex(re_max, im_lo), complex(re_max, im_hi), complex(-re_max, im_hi)]
    sp_ = np.asarray(special, dtype=complex)
    pts = []
    for a, b in zip(c, c[1:] + c[:1]):
        L, t, seg = abs(b - a), 0.0, []
        while t < L:
            w = a + (b - a) * t / L
            seg.append(w)
            t += min(ds, 0.2 * float(np.abs(w - sp_).min()))
        pts.append(np.array(seg))
    z = np.concatenate(pts)
    evalD = lambda zz: np.concatenate([ev.D(zz[k:k + chunk]) for k in range(0, zz.size, chunk)])
    D = evalD(z)
    verified = False
    for it in range(max_iter):
        dphi = np.angle(np.roll(D, -1) / D)
        bad = np.where(np.abs(dphi) > tol)[0]
        if bad.size == 0:
            # verification pass: bisect every interval
            mid = 0.5 * (z + np.roll(z, -1)); Dm = evalD(mid)
            h1, h2 = np.angle(Dm / D), np.angle(np.roll(D, -1) / Dm)
            bad = np.where((np.abs(h1 + h2 - dphi) > 1e-6) | (np.abs(h1) > tol) | (np.abs(h2) > tol))[0]
            if bad.size == 0:
                verified = True
                break
            z = np.insert(z, bad + 1, mid[bad]); D = np.insert(D, bad + 1, Dm[bad])
            continue
        mid = 0.5 * (z[bad] + np.roll(z, -1)[bad])
        Dm = evalD(mid)
        z = np.insert(z, bad + 1, mid); D = np.insert(D, bad + 1, Dm)
    dphi = np.angle(np.roll(D, -1) / D)
    return dict(winding=float(dphi.sum() / (2 * np.pi)), points=int(z.size), verified=bool(verified),
                max_phase_step=float(np.abs(dphi).max()), refinements=it,
                min_abs_D=float(np.nanmin(np.abs(D))), n_nan=int(np.isnan(D).sum()),
                rectangle=[-re_max, re_max, im_lo, im_hi])


def real_axis_determinant(bg, omegas, X0=14.0, h=0.02):
    """|det[Q_L, Q_R]| with Q spanning the outgoing-or-decaying solutions (real w)."""
    n = int(round(X0 / h))
    AL = A_relative(bg, np.linspace(-X0, 0.0, 2 * n + 1), 0.0)
    AR = A_relative(bg, np.linspace(X0, 0.0, 2 * n + 1), 0.0)
    nL, vL, c2L = bg.side("L"); nR, vR, c2R = bg.side("R")
    out = []
    for w in omegas:
        mL = asymptotic_modes(w, nL, vL, c2L, "L"); mR = asymptotic_modes(w, nR, vR, c2R, "R")
        YL = np.array([m.vec for m in mL if m.kind in ("out", "decay")]).T
        YR = np.array([m.vec for m in mR if m.kind in ("out", "decay")]).T
        if YL.shape[1] != 2 or YR.shape[1] != 2:
            out.append(np.nan); continue
        Q = []
        for Y, A, hh in ((YL, AL, h), (YR, AR, -h)):
            Y = np.linalg.qr(Y)[0]
            Aw = A + w * WD[None]
            for i in range(n):
                a0, am, a1 = Aw[2 * i], Aw[2 * i + 1], Aw[2 * i + 2]
                k1 = a0 @ Y; k2 = am @ (Y + 0.5 * hh * k1)
                k3 = am @ (Y + 0.5 * hh * k2); k4 = a1 @ (Y + hh * k3)
                Y = Y + (hh / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                if (i + 1) % 10 == 0:
                    Y = np.linalg.qr(Y)[0]
            Q.append(np.linalg.qr(Y)[0])
        out.append(abs(np.linalg.det(np.column_stack(Q))))
    return np.array(out)


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgA = dict(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0, g=1.0)
    cfgB = dict(kind="const_c", nL=1.0, nR=0.4, j=0.6, width=1.0, c2L=1.0)
    out = {}
    # ---- controls first: the count must equal the number of modes found by the scan
    pred = json.load(open(os.path.join(ROOT, "results", "stability.json")))
    for d in (8.0, 14.0):
        cfg = dict(cfgA); cfg["pair_half_length"] = d
        t0 = time.time()
        wmc = omega_max(0.6 / 0.4, 0.4); spc = (0.0, wmc, -wmc)
        r = winding(Evans(Background(**cfg), d + 14.0), 1.0, 1e-3, 0.3, special=spc)
        r["counts_vs_lower_edge"] = []
        for lo in (0.02, 0.005, 0.002, 0.0006):
            rr = winding(Evans(Background(**cfg), d + 14.0), 1.0, lo, 0.3, special=spc)
            ex = sum((2 if abs(z["omega"][0]) > 1e-6 else 1) for z in pred[f"control_pair_half_length_{d:g}"]["unstable_modes"] if z["omega"][1] > lo)
            r["counts_vs_lower_edge"].append(dict(im_lo=lo, winding=rr["winding"], expected_from_scan=ex, verified=rr["verified"], max_phase_step=rr["max_phase_step"]))
        modes = pred[f"control_pair_half_length_{d:g}"]["unstable_modes"]
        exp = sum((2 if abs(z["omega"][0]) > 1e-6 else 1) for z in modes if z["omega"][1] > 1e-3)
        r.update(expected_from_scan=exp, seconds=time.time() - t0)
        out[f"control_pair_half_length_{d:g}"] = r
        print("control", d, r)
    # ---- single-horizon backgrounds: a-priori box |w| <= 4 v_max^2 + sup(g n0), Im w <= sup(g n0)
    for name, cfg in (("A_constant_g", cfgA), ("B_constant_c", cfgB)):
        bg = Background(**cfg)
        xs = np.linspace(-40, 40, 8001)
        vmax, bmax = float(bg.v(xs).max()), float(bg.c2(xs).max())
        box_re, box_im = 4 * vmax ** 2 + bmax, bmax
        t0 = time.time()
        wm = omega_max(*bg.side("R")[1:]); spc = (0.0, wm, -wm)
        r = winding(Evans(bg, 14.0, h=0.01), 1.01 * box_re, 1e-3, 1.05 * box_im, special=spc)
        r.update(v_max=vmax, sup_g_n0=bmax, apriori_bound_abs_omega=box_re,
                 apriori_bound_im_omega=box_im, seconds=time.time() - t0)
        # resolution check of the same count
        r2 = winding(Evans(bg, 14.0, h=0.02), 1.01 * box_re, 1e-3, 1.05 * box_im, special=spc)
        r["winding_coarser_step"] = r2["winding"]
        # the same count with the lower edge of the contour moved towards the real axis
        r["counts_vs_lower_edge"] = []
        for lo in (1e-4, 1e-5, 1e-6):
            rr = winding(Evans(bg, 14.0, h=0.01), 1.01 * box_re, lo, 1.05 * box_im, special=spc)
            r["counts_vs_lower_edge"].append(dict(im_lo=lo, winding=rr["winding"], points=rr["points"], verified=rr["verified"],
                                                  max_phase_step=rr["max_phase_step"], min_abs_D=rr["min_abs_D"]))
        # real-axis solvability determinant
        wm = omega_max(*bg.side("R")[1:])
        om = np.concatenate([np.linspace(0.02, 0.98, 97) * wm, np.linspace(1.02, 4.0, 150) * wm])
        dr = real_axis_determinant(bg, om)
        r["real_axis"] = dict(n_frequencies=int(om.size), omega_range=[float(om[0]), float(om[-1])],
                              min_d=float(np.nanmin(dr)), argmin_omega_over_max=float(om[np.nanargmin(dr)] / wm),
                              min_d_over_omega=float(np.nanmin(dr / om)), n_nan=int(np.isnan(dr).sum()))
        out[name] = r
        print(name, {k: v for k, v in r.items() if k != "real_axis"}); print("   real axis", r["real_axis"])
    json.dump(out, open(os.path.join(ROOT, "results", "evans_winding.json"), "w"), indent=1)
