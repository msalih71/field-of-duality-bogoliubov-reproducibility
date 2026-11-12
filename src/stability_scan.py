"""Complex-frequency search for dynamical instabilities (test 9).

For Im(w) > 0 a dynamically unstable mode is a solution of the stationary
relative-field equations that decays at both ends.  On each homogeneous side
the four roots k of the dispersion relation split into two whose plane waves
decay away from the scatterer and two that grow.  Let E_L(w) be the
two-dimensional space of solutions decaying towards -infinity and E_R(w) the
space decaying towards +infinity, both transported to x = 0.  An unstable
eigenfrequency exists exactly where the two spaces intersect, i.e. where

    d(w) = | det [ Q_L(w) , Q_R(w) ] |  = 0 ,

with Q_L, Q_R orthonormal bases (0 <= d <= 1; an Evans-function-type test).
Transport is done with RK4 in the direction in which the wanted subspace
grows, with repeated QR re-orthonormalisation, so it is numerically stable.

The detector is validated on a black-hole/white-hole pair, where lasing
instabilities are known to exist (Finazzi & Parentani 2010).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from fod_scattering import Background, A_relative, omega_max

np.seterr(over="ignore")


def decaying_basis(omega, v, c2, side):
    ks = np.roots([0.25, 0.0, c2 - v * v, 2.0 * omega * v, -omega * omega])
    keep = [k for k in ks if (k.imag < 0 if side == "L" else k.imag > 0)]
    cols = []
    for k in keep:
        u, th = k * k, -2j * (omega - v * k)
        cols.append(np.array([u, 1j * k * u, th, 1j * k * th], dtype=complex))
    return np.array(cols).T if cols else np.zeros((4, 0), dtype=complex)


class Transporter:
    """Pre-tabulates A(x) at w = 0 on the RK4 nodes of both half-lines."""

    def __init__(self, bg, X0, h=0.02, qr_every=10):
        self.bg, self.X0, self.h, self.qr_every = bg, X0, h, qr_every
        n = int(round(X0 / h))
        self.n = n
        self.xL = np.linspace(-X0, 0.0, 2 * n + 1)      # nodes and midpoints
        self.xR = np.linspace(X0, 0.0, 2 * n + 1)
        self.AL = A_relative(bg, self.xL, 0.0)
        self.AR = A_relative(bg, self.xR, 0.0)
        self.W = np.zeros((4, 4), dtype=complex)        # dA/dw
        self.W[1, 2] = -2j
        self.W[3, 0] = 2j

    def _transport(self, Y, A, h, omega):
        Aw = A + omega * self.W[None, :, :]
        for i in range(self.n):
            a0, am, a1 = Aw[2 * i], Aw[2 * i + 1], Aw[2 * i + 2]
            k1 = a0 @ Y
            k2 = am @ (Y + 0.5 * h * k1)
            k3 = am @ (Y + 0.5 * h * k2)
            k4 = a1 @ (Y + h * k3)
            Y = Y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if (i + 1) % self.qr_every == 0:
                Y, _ = np.linalg.qr(Y)
        Y, _ = np.linalg.qr(Y)
        return Y

    def d(self, omega):
        _, vL, c2L = self.bg.side("L")
        _, vR, c2R = self.bg.side("R")
        YL = decaying_basis(omega, vL, c2L, "L")
        YR = decaying_basis(omega, vR, c2R, "R")
        if YL.shape[1] != 2 or YR.shape[1] != 2:
            return np.nan
        QL = self._transport(np.linalg.qr(YL)[0], self.AL, self.h, omega)
        QR = self._transport(np.linalg.qr(YR)[0], self.AR, -self.h, omega)
        return float(abs(np.linalg.det(np.column_stack([QL, QR]))))


def analyse(name, bg, X0, re_grid, im_grid, h=0.02):
    t = time.time()
    T = Transporter(bg, X0, h)
    G = np.array([[T.d(wr + 1j * wi) for wr in re_grid] for wi in im_grid])
    zeros = []
    for a in range(len(im_grid)):
        for b in range(len(re_grid)):
            g = G[a, b]
            nb = G[max(a - 1, 0):a + 2, max(b - 1, 0):b + 2]
            if np.isfinite(g) and g <= np.nanmin(nb) and g < 0.2:
                f = lambda z: T.d(z[0] + 1j * abs(z[1]))
                r = minimize(f, [re_grid[b], im_grid[a]], method="Nelder-Mead",
                             options=dict(xatol=1e-9, fatol=1e-12, maxiter=600))
                w = complex(r.x[0], abs(r.x[1]))
                if r.fun < 1e-6 and w.imag > 1e-6 and not any(
                        abs(w - complex(*z["omega"])) < 1e-4 for z in zeros):
                    # confirm with a finer step
                    T2 = Transporter(bg, X0, h / 2)
                    zeros.append(dict(omega=[w.real, w.imag], d=float(r.fun),
                                      d_half_step=T2.d(w)))
    return dict(name=name, X0=X0, h=h,
                grid_re=[float(re_grid[0]), float(re_grid[-1]), len(re_grid)],
                grid_im=[float(im_grid[0]), float(im_grid[-1]), len(im_grid)],
                min_d=float(np.nanmin(G)), median_d=float(np.nanmedian(G)),
                n_nan=int(np.isnan(G).sum()), unstable_modes=zeros,
                seconds=time.time() - t), G


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgA = dict(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0, g=1.0)
    cfgB = dict(kind="const_c", nL=1.0, nR=0.4, j=0.6, width=1.0, c2L=1.0)
    out, grids = {}, {}
    for name, cfg in (("A_constant_g", cfgA), ("B_constant_c", cfgB)):
        bg = Background(**cfg)
        wm = omega_max(*bg.side("R")[1:])
        re = np.linspace(0.0, 2.0 * wm, 49)
        im = np.geomspace(0.003, 1.5, 24)
        out[name], grids[name] = analyse(name, bg, 14.0, re, im)
        out[name]["omega_max"] = wm
    for d in (8.0, 14.0):                       # positive control: two horizons
        cfg = dict(cfgA); cfg["pair_half_length"] = d
        bg = Background(**cfg)
        wm = omega_max(0.6 / 0.4, 0.4)
        re = np.linspace(0.0, 1.2 * wm, 73)
        im = np.geomspace(0.001, 0.3, 26)
        key = f"control_pair_half_length_{d:g}"
        out[key], grids[key] = analyse(key, bg, d + 14.0, re, im)
        out[key]["omega_max"] = wm
    json.dump(out, open(os.path.join(ROOT, "results", "stability.json"), "w"), indent=1)
    np.savez(os.path.join(ROOT, "results", "stability_grids.npz"), **grids)
    for k, v in out.items():
        print(k, "min d = %.3e" % v["min_d"], "| unstable:",
              [(round(z["omega"][0], 5), round(z["omega"][1], 5), "%.1e" % z["d_half_step"])
               for z in v["unstable_modes"]], "| t=%.0fs" % v["seconds"], "nan", v["n_nan"])
