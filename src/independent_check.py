"""Independent cross-check of the scattering matrix.

Differs from src/fod_scattering.py in three ways:
  * variables: standard Nambu amplitudes (U, V) of delta psi, not the relative field;
  * equations: the textbook BdG operator built from psi0 (with its phase) and the
    external potential V(x) of the inverse construction -- neither appears in the
    relative-field equations;
  * discretisation: fourth-order finite differences of the second-order equations with
    one-sided boundary stencils, not first-order-system collocation.

Stationary BdG equations (hbar = m = 1, mu = 0), delta psi = U e^{-iwt} + conj(V) e^{iwt}:
     w U = [ -(1/2) d_xx + V_ext + 2 g n0 ] U + g psi0^2 V
    -w V = [ -(1/2) d_xx + V_ext + 2 g n0 ] V + g conj(psi0)^2 U
Only the asymptotic plane-wave modes are shared with the main code; they are mapped
to (U, U', V, V') by U = psi0 (u + i th), V = conj(psi0) (u - i th).
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, os.path.dirname(__file__))
from fod_scattering import Background, asymptotic_modes, omega_max, scattering_matrix

np.seterr(over="ignore")


def nambu_scattering(bg, omega, L=30.0, N=24000):
    # background phase on a fine grid (theta0' = v0)
    xf = np.linspace(-L, L, 400001)
    th0 = cumulative_trapezoid(bg.v(xf), xf, initial=0.0)
    th0 -= np.interp(0.0, xf, th0)
    theta = lambda x: np.interp(x, xf, th0)
    psi0 = lambda x: np.sqrt(bg.n(x)) * np.exp(1j * theta(x))
    Vext = lambda x: bg.potential(x, mu=0.0)
    gfun = lambda x: bg.c2(x) / bg.n(x)

    def to_nambu(x, vec):
        u, du, th, dth = vec
        p = psi0(x); dp = p * (bg.dn(x) / (2 * bg.n(x)) + 1j * bg.v(x))
        f, df = u + 1j * th, du + 1j * dth
        h, dh = u - 1j * th, du - 1j * dth
        return np.array([p * f, dp * f + p * df, np.conj(p) * h, np.conj(dp) * h + np.conj(p) * dh])

    nL, vL, c2L = bg.side("L"); nR, vR, c2R = bg.side("R")
    mL = asymptotic_modes(omega, nL, vL, c2L, "L"); mR = asymptotic_modes(omega, nR, vR, c2R, "R")
    EL = np.array([to_nambu(-L, mo.vec) for mo in mL]).T
    ER = np.array([to_nambu(L, mo.vec) for mo in mR]).T
    PL, PR = np.linalg.inv(EL), np.linalg.inv(ER)
    key = lambda side, mo: (-mo.norm, 0 if side == "L" else 1, -abs(mo.k))
    ins = sorted([("L", i) for i, mo in enumerate(mL) if mo.kind == "in"]
                 + [("R", i) for i, mo in enumerate(mR) if mo.kind == "in"],
                 key=lambda t: key(t[0], (mL if t[0] == "L" else mR)[t[1]]))
    outs = sorted([("L", i) for i, mo in enumerate(mL) if mo.kind == "out"]
                  + [("R", i) for i, mo in enumerate(mR) if mo.kind == "out"],
                  key=lambda t: key(t[0], (mL if t[0] == "L" else mR)[t[1]]))
    con = [("L", i) for i, mo in enumerate(mL) if mo.kind in ("in", "grow")] \
        + [("R", i) for i, mo in enumerate(mR) if mo.kind in ("in", "grow")]
    assert len(con) == 4
    # ---- second-order finite-difference discretisation of the Nambu equations
    import scipy.sparse as sps, scipy.sparse.linalg as spla
    x = np.linspace(-L, L, N + 1); h = x[1] - x[0]
    p = psi0(x); g = gfun(x); hh = Vext(x) + 2 * g * bg.n(x)
    rows, cols, vals = [], [], []
    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)
    iu = lambda i: 2 * i
    iv = lambda i: 2 * i + 1
    c5 = np.array([-1, 16, -30, 16, -1]) / (12 * h * h)          # centred, 4th order
    c6 = np.array([10, -15, -4, 14, -6, 1]) / (12 * h * h)       # at the first interior point
    r = 0
    for i in range(1, N):
        if 2 <= i <= N - 2:
            idx, co = range(i - 2, i + 3), c5
        elif i == 1:
            idx, co = range(0, 6), c6
        else:
            idx, co = range(N, N - 6, -1), c6
        # -(1/2) U'' + (hh - w) U + g p^2 V = 0
        for k, ck in zip(idx, co):
            add(r, iu(k), -0.5 * ck)
        add(r, iu(i), hh[i] - omega); add(r, iv(i), g[i] * p[i] ** 2); r += 1
        # -(1/2) V'' + (hh + w) V + g conj(p)^2 U = 0
        for k, ck in zip(idx, co):
            add(r, iv(k), -0.5 * ck)
        add(r, iv(i), hh[i] + omega); add(r, iu(i), g[i] * np.conj(p[i]) ** 2); r += 1
    d1 = np.array([-25, 48, -36, 16, -3]) / (12 * h)             # one-sided first derivative
    bc_row0 = r
    for s_, i_ in con:
        P = PL[i_] if s_ == "L" else PR[i_]                       # acts on (U, U', V, V')
        if s_ == "L":
            pts, dco = range(0, 5), d1
            b = 0
        else:
            pts, dco = range(N, N - 5, -1), -d1
            b = N
        add(r, iu(b), P[0]); add(r, iv(b), P[2])
        for k, ck in zip(pts, dco):
            add(r, iu(k), P[1] * ck); add(r, iv(k), P[3] * ck)
        r += 1
    K = sps.coo_matrix((vals, (rows, cols)), shape=(2 * N + 2, 2 * N + 2)).tocsc()
    lu = spla.splu(K)
    S = np.zeros((len(outs), len(ins)), dtype=complex)
    info = dict(N=N, L=L)
    def boundary_state(sol, left):
        U, V = sol[0::2], sol[1::2]
        if left:
            return np.array([U[0], d1 @ U[0:5], V[0], d1 @ V[0:5]])
        return np.array([U[N], -d1 @ U[N:N - 5:-1], V[N], -d1 @ V[N:N - 5:-1]])
    for a_, src in enumerate(ins):
        rhs_ = np.zeros(2 * N + 2, dtype=complex)
        rhs_[bc_row0 + con.index(src)] = 1.0
        sol = lu.solve(rhs_)
        aL, aR = PL @ boundary_state(sol, True), PR @ boundary_state(sol, False)
        S[:, a_] = [aL[i] if s == "L" else aR[i] for s, i in outs]
    return S, info


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for name, cfg in (("A_constant_g", dict(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0)),
                      ("B_constant_c", dict(kind="const_c", nL=1.0, nR=0.4, j=0.6, width=1.0, c2L=1.0))):
        bg = Background(**cfg); wm = omega_max(*bg.side("R")[1:])
        for f in (0.1, 0.3, 0.7, 1.3):
            S, info = nambu_scattering(bg, f * wm)
            ref = scattering_matrix(bg, f * wm, L=40.0, N=32000)
            A, B = np.abs(S) ** 2, np.abs(ref["S"]) ** 2
            eta_i, eta_o = ref["eta_in"], ref["eta_out"]
            rec = dict(background=name, omega_over_max=f, shape=list(S.shape),
                       max_abs_diff_S2=float(np.max(np.abs(A - B))),
                       max_rel_diff_S2=float(np.max(np.abs(A - B) / B)),
                       pseudo_unitarity_residual_nambu=float(np.linalg.norm(S.conj().T @ eta_o @ S - eta_i, 2)),
                       grid=info)
            out.append(rec)
            print(name, f, rec["shape"], "abs %.2e rel %.2e resid %.2e" % (
                rec["max_abs_diff_S2"], rec["max_rel_diff_S2"], rec["pseudo_unitarity_residual_nambu"]),
                info)
    json.dump(out, open(os.path.join(ROOT, "results", "independent_check.json"), "w"), indent=1)
