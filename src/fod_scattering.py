"""Stationary BdG scattering on a 1D transonic Gross-Pitaevskii background.

Units: hbar = m = 1.  The flow is towards +x; the upstream (left) side is
subsonic and the downstream (right) side may be supersonic.

Relative-field formulation (manuscript Eqs. 48-49), time dependence e^{-i w t}:

    -i w u  + v u'  = -(1/2) D th
    -i w th + v th' = -2 c^2 u + (1/2) D u ,     D f = f'' + (n'/n) f'

with c^2(x) = g(x) n0(x) and v(x) = j / n0(x).

Gradient formulation (manuscript Eqs. 52-53) with p = th', q = u' (ell = 1):

    -i w q + (v q)' = -(1/2) d_x[ p' + (n'/n) p ]
    -i w p + (v p)' = -2 [ c^2 q + (c^2)' u ] + (1/2) d_x[ q' + (n'/n) q ]

which is closed in (p, q) only if (c^2)' = 0; otherwise the auxiliary field u
with u' = q must be carried along.

Nothing in this module inserts a target spectrum: the S-matrix is obtained
from a linear boundary-value solve and is then *tested* for pseudo-unitarity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------- background
@dataclass
class Background:
    """Smooth stationary background defined by tanh transitions.

    n0(x)  = nL + (nR - nL) * s(x),   s(x) = (1 + tanh(x / width)) / 2
    c2(x)  : 'const_g'  -> c2 = g * n0(x)                (g constant)
             'const_c'  -> c2 = c2L everywhere           (g(x) = c2L / n0(x))
             'g_step'   -> c2 = c2L + (c2R - c2L) * s(x) (independent profile)
    v(x)   = j / n0(x)
    The external potential that makes this an exact stationary GP solution is
    V(x) = mu - j^2 / (2 n0^2) - c2(x) + (1/2) (sqrt n0)'' / sqrt n0  (Eq. 46).
    """
    nL: float = 1.0
    nR: float = 0.4
    j: float = 0.6
    width: float = 1.0
    kind: str = "const_g"
    g: float = 1.0
    c2L: float = 1.0
    c2R: float = 0.4
    pair_half_length: float = 0.0   # > 0: sub-super-sub profile (two horizons)

    # profile and its derivatives (analytic)
    def _s1(self, x):
        return 0.5 * (1.0 + np.tanh(x / self.width))

    def _ds1(self, x):
        return 0.5 / self.width / np.cosh(x / self.width) ** 2

    def _d2s1(self, x):
        return -np.tanh(x / self.width) / self.width ** 2 / np.cosh(x / self.width) ** 2

    def s(self, x):
        d = self.pair_half_length
        return self._s1(x) if d == 0 else self._s1(x + d) - self._s1(x - d)

    def ds(self, x):
        d = self.pair_half_length
        return self._ds1(x) if d == 0 else self._ds1(x + d) - self._ds1(x - d)

    def d2s(self, x):
        d = self.pair_half_length
        return self._d2s1(x) if d == 0 else self._d2s1(x + d) - self._d2s1(x - d)

    def d3s(self, x):
        t = np.tanh(x / self.width)
        sech2 = 1.0 / np.cosh(x / self.width) ** 2
        return (-sech2 + 3.0 * t * t * sech2) * sech2 / self.width ** 3 \
            if False else (2.0 * t * t - sech2) * sech2 / self.width ** 3

    def n(self, x):
        return self.nL + (self.nR - self.nL) * self.s(x)

    def dn(self, x):
        return (self.nR - self.nL) * self.ds(x)

    def d2n(self, x):
        return (self.nR - self.nL) * self.d2s(x)

    def v(self, x):
        return self.j / self.n(x)

    def dv(self, x):
        return -self.j * self.dn(x) / self.n(x) ** 2

    def lam(self, x):                       # n'/n
        return self.dn(x) / self.n(x)

    def dlam(self, x):                      # (n'/n)'
        return self.d2n(x) / self.n(x) - self.lam(x) ** 2

    def c2(self, x):
        if self.kind == "const_g":
            return self.g * self.n(x)
        if self.kind == "const_c":
            return self.c2L + 0.0 * x
        if self.kind == "g_step":
            return self.c2L + (self.c2R - self.c2L) * self.s(x)
        raise ValueError(self.kind)

    def dc2(self, x):
        if self.kind == "const_g":
            return self.g * self.dn(x)
        if self.kind == "const_c":
            return 0.0 * x
        if self.kind == "g_step":
            return (self.c2R - self.c2L) * self.ds(x)
        raise ValueError(self.kind)

    def side(self, which):
        """Asymptotic (n, v, c2) on 'L' or 'R'."""
        n = self.nL if (which == "L" or self.pair_half_length > 0) else self.nR
        if self.kind == "const_g":
            c2 = self.g * n
        elif self.kind == "const_c":
            c2 = self.c2L
        else:
            c2 = self.c2L if which == "L" else self.c2R
        return n, self.j / n, c2

    def mach(self, x):
        return self.v(x) / np.sqrt(self.c2(x))

    def horizon(self):
        """Location and surface gravity kappa = |d(v - c)/dx| at v = c."""
        from scipy.optimize import brentq
        f = lambda x: self.v(x) - np.sqrt(self.c2(x))
        L = 30.0 * self.width
        if f(-L) * f(L) > 0:
            return None, None
        xh = brentq(f, -L, L, xtol=1e-14)
        kappa = abs(self.dv(xh) - self.dc2(xh) / (2.0 * np.sqrt(self.c2(xh))))
        return xh, kappa

    def potential(self, x, mu=0.0):
        """External potential of the inverse construction, Eq. (46)."""
        n = self.n(x)
        sq = np.sqrt(n)
        d2sq = self.d2n(x) / (2 * sq) - self.dn(x) ** 2 / (4 * n * sq)
        return mu - self.j ** 2 / (2 * n ** 2) - self.c2(x) + 0.5 * d2sq / sq


# ------------------------------------------------------------ asymptotic modes
@dataclass
class Mode:
    k: complex
    kind: str            # 'in', 'out', 'decay', 'grow'
    norm: int            # Bogoliubov norm sign (+1/-1), 0 for evanescent
    vg: float            # group velocity (nan for evanescent)
    vec: np.ndarray      # (u, u', th, th') at the boundary, flux-normalised


def flux_matrix(n, j):
    """Hermitian form M with conserved flux J = y1^dagger M y2, y=(u,u',th,th')."""
    M = np.zeros((4, 4), dtype=complex)
    M[0, 2] += 1j * j
    M[2, 0] += -1j * j
    M[2, 3] += -0.5j * n
    M[3, 2] += 0.5j * n
    M[0, 1] += -0.5j * n
    M[1, 0] += 0.5j * n
    return M


def omega_max(v, c2, nk=200001):
    """Maximum of the negative-norm branch  w = v k - sqrt(c2 k^2 + k^4/4), k>0."""
    if v * v <= c2:
        return 0.0
    kc = 2.0 * np.sqrt(v * v - c2)
    k = np.linspace(0.0, kc, nk)
    w = v * k - np.sqrt(c2 * k * k + 0.25 * k ** 4)
    i = int(np.argmax(w))
    # parabolic refinement
    if 0 < i < nk - 1:
        y0, y1, y2 = w[i - 1], w[i], w[i + 1]
        return float(y1 - 0.125 * (y0 - y2) ** 2 / (y0 - 2 * y1 + y2))
    return float(w[i])


def asymptotic_modes(omega, n, v, c2, which, tol=1e-9):
    """Roots of (w - v k)^2 = c2 k^2 + k^4/4 on a uniform side, classified.

    which = 'L' (domain extends to -infinity) or 'R' (+infinity).
    """
    coeffs = [0.25, 0.0, c2 - v * v, 2.0 * omega * v, -omega * omega]
    roots = np.roots(coeffs)
    j = n * v
    M = flux_matrix(n, j)
    modes = []
    for k in roots:
        real = abs(k.imag) < tol * max(1.0, abs(k))
        if real:
            k = complex(k.real, 0.0)
        u = k * k
        th = -2j * (omega - v * k)
        vec = np.array([u, 1j * k * u, th, 1j * k * th], dtype=complex)
        if real:
            kr = k.real
            vg = (2 * v * (omega - v * kr) + 2 * c2 * kr + kr ** 3) / (2 * (omega - v * kr))
            J = (vec.conj() @ M @ vec).real
            vec = vec / np.sqrt(abs(J))
            norm = int(np.sign(omega - v * kr))
            # consistency: sign(J) must equal norm * sign(vg)
            assert int(np.sign(J)) == norm * int(np.sign(vg)), "flux/norm mismatch"
            incoming = (vg > 0) if which == "L" else (vg < 0)
            modes.append(Mode(k, "in" if incoming else "out", norm, float(vg), vec))
        else:
            # e^{ikx}: Im k > 0 decays towards +infinity and grows towards -infinity
            decays = (k.imag < 0) if which == "L" else (k.imag > 0)
            vec = vec / np.linalg.norm(vec)
            modes.append(Mode(k, "decay" if decays else "grow", 0, float("nan"), vec))
    return modes


# ------------------------------------------------------------- ODE generators
def A_relative(bg, x, omega):
    """y' = A y for y = (u, u', th, th'); returns array (len(x), 4, 4)."""
    x = np.atleast_1d(x)
    lam, v, c2 = bg.lam(x), bg.v(x), bg.c2(x)
    A = np.zeros((x.size, 4, 4), dtype=complex)
    A[:, 0, 1] = 1.0
    A[:, 2, 3] = 1.0
    # u'' = -lam u' + 2(-i w th + v th' + 2 c2 u)
    A[:, 1, 0] = 4.0 * c2
    A[:, 1, 1] = -lam
    A[:, 1, 2] = -2j * omega
    A[:, 1, 3] = 2.0 * v
    # th'' = -lam th' - 2(-i w u + v u')
    A[:, 3, 0] = 2j * omega
    A[:, 3, 1] = -2.0 * v
    A[:, 3, 3] = -lam
    return A


def A_gradient(bg, x, omega, augmented, omega_closed=False):
    """Gradient system for Y = (p, p', q, q') [+ u if augmented].

    omega_closed=True eliminates u at fixed nonzero frequency through the
    continuity equation,  u = [ v q + (1/2)(p' + (n'/n) p) ] / (i w),
    which closes the stationary system in (p, q) at the price of a 1/w
    coefficient (see the proof supplement, Theorem 3(c)).
    """
    x = np.atleast_1d(x)
    lam, dlam, v, dv = bg.lam(x), bg.dlam(x), bg.v(x), bg.dv(x)
    c2, dc2 = bg.c2(x), bg.dc2(x)
    d = 5 if augmented else 4
    A = np.zeros((x.size, d, d), dtype=complex)
    A[:, 0, 1] = 1.0
    A[:, 2, 3] = 1.0
    # p'' = -lam p' - dlam p - 2(-i w q + v q' + v' q)
    A[:, 1, 0] = -dlam
    A[:, 1, 1] = -lam
    A[:, 1, 2] = 2j * omega - 2.0 * dv
    A[:, 1, 3] = -2.0 * v
    # q'' = -lam q' - dlam q + 2(-i w p + v p' + v' p) + 4 (c2 q + c2' u)
    A[:, 3, 0] = -2j * omega + 2.0 * dv
    A[:, 3, 1] = 2.0 * v
    A[:, 3, 2] = -dlam + 4.0 * c2
    A[:, 3, 3] = -lam
    if augmented:
        A[:, 3, 4] = 4.0 * dc2
        A[:, 4, 2] = 1.0                      # u' = q
    elif omega_closed:
        f = 4.0 * dc2 / (1j * omega)
        A[:, 3, 2] += f * v
        A[:, 3, 1] += f * 0.5
        A[:, 3, 0] += f * 0.5 * lam
    return A


# ------------------------------------------------------------------ BVP solve
def _assemble(Afun, xs, d):
    """Fourth-order Lobatto IIIA (Hermite-Simpson) collocation, linear system."""
    N = xs.size - 1
    h = np.diff(xs)
    xm = 0.5 * (xs[:-1] + xs[1:])
    A0 = Afun(xs)
    Am = Afun(xm)
    I = np.eye(d)[None, :, :]
    hh = h[:, None, None]
    An, Ap = A0[:-1], A0[1:]
    Lb = -I - hh / 6.0 * (An + 2.0 * Am + 0.5 * hh * (Am @ An))
    Rb = I - hh / 6.0 * (Ap + 2.0 * Am - 0.5 * hh * (Am @ Ap))
    rows = np.repeat(np.arange(N * d).reshape(N, d), d, axis=1).reshape(N, d, d)
    colsL = np.tile(np.arange(N * d).reshape(N, 1, d), (1, d, 1))
    colsR = colsL + d
    data = np.concatenate([Lb.ravel(), Rb.ravel()])
    r = np.concatenate([rows.ravel(), rows.ravel()])
    c = np.concatenate([colsL.ravel(), colsR.ravel()])
    return sps.coo_matrix((data, (r, c)), shape=(N * d, (N + 1) * d)).tocsr()


def _mode_basis(modes, transform=None, extra=None):
    """Columns = mode vectors (optionally mapped to another representation)."""
    cols = []
    for mo in modes:
        vec = mo.vec if transform is None else transform(mo)
        cols.append(vec)
    E = np.array(cols).T
    if extra is not None:
        E = np.column_stack([E, extra])
    return E


def scattering_matrix(bg, omega, L=40.0, N=4000, formulation="relative"):
    """Return dict with S (out <- in), eta_in, eta_out, channel info, diagnostics.

    formulation: 'relative'  -> (u, u', th, th')
                 'gradient'  -> (p, p', q, q')      (valid only if (c^2)' = 0)
                 'gradient_aug' -> (p, p', q, q', u) (always valid)
    """
    xs = np.linspace(-L, L, N + 1)
    nLs, vLs, c2Ls = bg.side("L")
    nRs, vRs, c2Rs = bg.side("R")
    mL = asymptotic_modes(omega, nLs, vLs, c2Ls, "L")
    mR = asymptotic_modes(omega, nRs, vRs, c2Rs, "R")

    if formulation == "relative":
        d = 4
        Afun = lambda x: A_relative(bg, x, omega)
        EL, ER = _mode_basis(mL), _mode_basis(mR)
    elif formulation in ("gradient", "gradient_omega"):
        d = 4
        closed = formulation == "gradient_omega"
        Afun = lambda x: A_gradient(bg, x, omega, augmented=False, omega_closed=closed)
        tr = lambda mo: 1j * mo.k * np.array([mo.vec[2], mo.vec[3], mo.vec[0], mo.vec[1]])
        EL, ER = _mode_basis(mL, tr), _mode_basis(mR, tr)
    elif formulation == "gradient_aug":
        d = 5
        Afun = lambda x: A_gradient(bg, x, omega, augmented=True)
        tr = lambda mo: np.append(
            1j * mo.k * np.array([mo.vec[2], mo.vec[3], mo.vec[0], mo.vec[1]]), mo.vec[0])
        const = np.array([0, 0, 0, 0, 1.0], dtype=complex)   # spurious k = 0 mode
        EL, ER = _mode_basis(mL, tr, const), _mode_basis(mR, tr, const)
    else:
        raise ValueError(formulation)

    PL, PR = np.linalg.inv(EL), np.linalg.inv(ER)   # rows = amplitude projectors

    # channel bookkeeping
    inL = [i for i, mo in enumerate(mL) if mo.kind == "in"]
    inR = [i for i, mo in enumerate(mR) if mo.kind == "in"]
    outL = [i for i, mo in enumerate(mL) if mo.kind == "out"]
    outR = [i for i, mo in enumerate(mR) if mo.kind == "out"]
    growL = [i for i, mo in enumerate(mL) if mo.kind == "grow"]
    growR = [i for i, mo in enumerate(mR) if mo.kind == "grow"]
    # order: positive-norm channels first (left before right), negative last
    key = lambda side, idx, ms: (-ms[idx].norm, 0 if side == "L" else 1, -abs(ms[idx].k))
    ins = sorted([("L", i) for i in inL] + [("R", i) for i in inR],
                 key=lambda t: key(t[0], t[1], mL if t[0] == "L" else mR))
    outs = sorted([("L", i) for i in outL] + [("R", i) for i in outR],
                  key=lambda t: key(t[0], t[1], mL if t[0] == "L" else mR))

    Msys = _assemble(Afun, xs, d)
    nun = (N + 1) * d
    # boundary rows: prescribed amplitudes of in / grow (/ constant) modes
    bc_rows = []      # (side, projector row, tag)
    for i in inL:
        bc_rows.append(("L", PL[i], ("in", "L", i)))
    for i in growL:
        bc_rows.append(("L", PL[i], ("zero",)))
    for i in inR:
        bc_rows.append(("R", PR[i], ("in", "R", i)))
    for i in growR:
        bc_rows.append(("R", PR[i], ("zero",)))
    if formulation == "gradient_aug":
        bc_rows.append(("L", PL[4], ("zero",)))
    assert len(bc_rows) == d, f"boundary-condition count {len(bc_rows)} != {d}"

    B = sps.lil_matrix((d, nun), dtype=complex)
    for r, (side, row, _) in enumerate(bc_rows):
        if side == "L":
            B[r, 0:d] = row
        else:
            B[r, nun - d:nun] = row
    K = sps.vstack([Msys, B.tocsr()]).tocsc()
    lu = spla.splu(K)

    nin, nout = len(ins), len(outs)
    S = np.zeros((nout, nin), dtype=complex)
    flux_dev = 0.0
    sols = []
    for a, (side_in, idx_in) in enumerate(ins):
        rhs = np.zeros(nun, dtype=complex)
        for r, (_, _, tag) in enumerate(bc_rows):
            if tag[0] == "in" and tag[1] == side_in and tag[2] == idx_in:
                rhs[N * d + r] = 1.0
        y = lu.solve(rhs).reshape(N + 1, d)
        sols.append(y)
        aL, aR = PL @ y[0], PR @ y[-1]
        for b, (side_out, idx_out) in enumerate(outs):
            S[b, a] = aL[idx_out] if side_out == "L" else aR[idx_out]
        if formulation == "relative":
            # pointwise conservation of the flux J(x) along the numerical solution
            nx, jx = bg.n(xs), bg.j
            u, du, th, dth = y.T
            J = (1j * (jx * (u.conj() * th - th.conj() * u)
                       - 0.5 * nx * (th.conj() * dth - dth.conj() * th
                                     + u.conj() * du - du.conj() * u))).real
            flux_dev = max(flux_dev, float(np.max(np.abs(J - J[0]))))

    eta_in = np.diag([float((mL if s == "L" else mR)[i].norm) for s, i in ins])
    eta_out = np.diag([float((mL if s == "L" else mR)[i].norm) for s, i in outs])
    resid = np.linalg.norm(S.conj().T @ eta_out @ S - eta_in, 2)
    describe = lambda s, i: dict(side=s, k=complex((mL if s == "L" else mR)[i].k),
                                 norm=(mL if s == "L" else mR)[i].norm,
                                 vg=(mL if s == "L" else mR)[i].vg)
    return dict(S=S, eta_in=eta_in, eta_out=eta_out, residual=float(resid),
                ins=[describe(s, i) for s, i in ins],
                outs=[describe(s, i) for s, i in outs],
                flux_dev=flux_dev, xs=xs, sols=sols, modesL=mL, modesR=mR)


def hawking_beta2(res):
    """|beta|^2 = |S[Hawking out (left, +), negative-norm in (right)]|^2."""
    io = [b for b, o in enumerate(res["outs"]) if o["side"] == "L"]
    ii = [a for a, i in enumerate(res["ins"]) if i["norm"] < 0]
    if not io or not ii:
        return 0.0
    return float(abs(res["S"][io[0], ii[0]]) ** 2)


# ----------------------------------------------------------- step-like horizon
def step_scattering(omega, n, v, c2L, c2R):
    """Exact algebraic solution for a discontinuous c^2 with uniform n and v.

    With n' = 0 the state y = (u, u', th, th') is continuous across the jump
    (the coefficient c^2 has a finite discontinuity), so the S-matrix follows
    from a 4x4 linear matching problem -- no differential equation is solved.
    """
    mL = asymptotic_modes(omega, n, v, c2L, "L")
    mR = asymptotic_modes(omega, n, v, c2R, "R")
    key = lambda side, mo: (-mo.norm, 0 if side == "L" else 1, -abs(mo.k))
    ins = sorted([("L", mo) for mo in mL if mo.kind == "in"]
                 + [("R", mo) for mo in mR if mo.kind == "in"],
                 key=lambda t: key(*t))
    outs = sorted([("L", mo) for mo in mL if mo.kind == "out"]
                  + [("R", mo) for mo in mR if mo.kind == "out"],
                  key=lambda t: key(*t))
    decL = [mo for mo in mL if mo.kind == "decay"]
    decR = [mo for mo in mR if mo.kind == "decay"]
    # unknowns: out amplitudes, then evanescent amplitudes (left, right)
    cols = [(+1 if s == "L" else -1) * mo.vec for s, mo in outs] \
        + [mo.vec for mo in decL] + [-mo.vec for mo in decR]
    Mx = np.array(cols).T
    assert Mx.shape == (4, 4)
    S = np.zeros((len(outs), len(ins)), dtype=complex)
    for a, (s, mo) in enumerate(ins):
        rhs = (-1 if s == "L" else +1) * mo.vec
        sol = np.linalg.solve(Mx, rhs)
        S[:, a] = sol[:len(outs)]
    eta_in = np.diag([float(mo.norm) for _, mo in ins])
    eta_out = np.diag([float(mo.norm) for _, mo in outs])
    resid = np.linalg.norm(S.conj().T @ eta_out @ S - eta_in, 2)
    return dict(S=S, eta_in=eta_in, eta_out=eta_out, residual=float(resid),
                ins=[dict(side=s, k=mo.k, norm=mo.norm, vg=mo.vg) for s, mo in ins],
                outs=[dict(side=s, k=mo.k, norm=mo.norm, vg=mo.vg) for s, mo in outs])
