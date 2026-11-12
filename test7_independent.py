"""Independent reproduction of the scattering calculation behind acceptance test 7.

Test 7 compares |S_ba|^2 from the frequency-closed gradient system with |S_ba|^2 from the
relative-field system on background A (constant g) over 60 frequencies, and requires the
largest absolute difference to be below 1e-7 at N = 8000.  It fails (1.15e-7).

This script recomputes both scattering matrices WITHOUT importing fod_scattering.py:

  * background   n0 = 0.7 - 0.3 tanh x, g = 1, j = 0.6, with its own analytic derivatives;
  * equations    both first-order systems are coded here from Eqs. (lin) of the paper;
  * modes        eigen-decomposition of the asymptotic matrix (not the dispersion polynomial);
  * normalisation  the conserved flux J is evaluated numerically on each eigenvector
                 (for the gradient system after mapping back to (u, u', theta, theta'));
  * direction    group-velocity sign from sign(J) sign(w - v k)   (Theorem "pseudo-unitarity", (ii));
  * solver       shooting from both ends to x = 0 with the adaptive DOP853 integrator and a
                 4x4 matching problem -- no collocation, no grid.

fod_scattering.py is imported only afterwards, to compare its collocation results with this
reference.  Output: results/test7_independent.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from scipy.integrate import solve_ivp

NL, NR, JCUR, G = 1.0, 0.4, 0.6, 1.0


def profile(x):
    t = np.tanh(x); s2 = 1.0 - t * t
    n = 0.5 * (NL + NR) + 0.5 * (NR - NL) * t
    dn = 0.5 * (NR - NL) * s2
    d2n = -(NR - NL) * t * s2
    lam = dn / n
    return dict(n=n, lam=lam, dlam=d2n / n - lam * lam, v=JCUR / n, dv=-JCUR * dn / n ** 2, c2=G * n, dc2=G * dn)


def M_relative(x, w):
    """y = (u, u', th, th').  -i w u + v u' = -(th'' + lam th')/2 ;  -i w th + v th' = -2 c2 u + (u'' + lam u')/2."""
    b = profile(x); M = np.zeros((4, 4), dtype=complex)
    M[0, 1] = 1.0; M[2, 3] = 1.0
    M[1] = [4 * b["c2"], -b["lam"], -2j * w, 2 * b["v"]]
    M[3] = [2j * w, -2 * b["v"], 0.0, -b["lam"]]
    return M


def M_gradient(x, w):
    """y = (p, p', q, q'), p = th', q = u', with u = [v q + (p' + lam p)/2] / (i w)."""
    b = profile(x); M = np.zeros((4, 4), dtype=complex)
    M[0, 1] = 1.0; M[2, 3] = 1.0
    # p'' = -(lam p)' + 2 i w q - 2 (v q)'
    M[1] = [-b["dlam"], -b["lam"], 2j * w - 2 * b["dv"], -2 * b["v"]]
    # q'' = -(lam q)' - 2 i w p + 2 (v p)' + 4 c2 q + 4 c2' u
    f = 4 * b["dc2"] / (1j * w)
    M[3] = [-2j * w + 2 * b["dv"] + f * 0.5 * b["lam"], 2 * b["v"] + f * 0.5, -b["dlam"] + 4 * b["c2"] + f * b["v"], -b["lam"]]
    return M


def to_relative(x, w, y):
    """(p, p', q, q') -> (u, u', th, th') at fixed frequency."""
    b = profile(x); p, dp, q, dq = y
    u = (b["v"] * q + 0.5 * (dp + b["lam"] * p)) / (1j * w)
    th = (b["v"] * p + 2 * b["c2"] * u - 0.5 * (dq + b["lam"] * q)) / (1j * w)
    return np.array([u, q, th, p])


def flux(x, y1, y2):
    b = profile(x); u1, du1, t1, dt1 = y1; u2, du2, t2, dt2 = y2
    return 1j * (JCUR * (np.conj(u1) * t2 - np.conj(t1) * u2)
                 - 0.5 * b["n"] * (np.conj(t1) * dt2 - np.conj(dt1) * t2 + np.conj(u1) * du2 - np.conj(du1) * u2))


def modes(Mfun, rel, x0, w, side):
    """Asymptotic modes at x0 that are allowed in a bounded solution, with their type."""
    b = profile(x0)
    mu, vec = np.linalg.eig(Mfun(x0, w))
    out = []
    for m_, y in zip(mu, vec.T):
        if abs(m_.real) < 1e-7:                                   # propagating
            yr = rel(x0, w, y)
            J = flux(x0, yr, yr).real
            k = m_.imag; sig = np.sign(w - b["v"] * k); vg = np.sign(J) * sig
            kind = ("in" if vg > 0 else "out") if side == "L" else ("in" if vg < 0 else "out")
            out.append(dict(kind=kind, norm=sig, k=k, y=y / np.sqrt(abs(J))))
        elif (m_.real > 0) == (side == "L"):                      # decays away from the scatterer
            out.append(dict(kind="ev", norm=0.0, k=m_.imag, y=y / np.linalg.norm(y)))
    return out


def smatrix(Mfun, rel, w, X0=18.0, rtol=1e-12, atol=1e-14):
    sides = {}
    for side, x0 in (("L", -X0), ("R", X0)):
        ms = modes(Mfun, rel, x0, w, side)
        Y0 = np.array([m_["y"] for m_ in ms]).T                    # 4 x (number of allowed modes)
        nm = Y0.shape[1]
        rhs = lambda x, y: (Mfun(x, w) @ y.reshape(4, nm)).ravel()
        sol = solve_ivp(rhs, (x0, 0.0), Y0.ravel(), method="DOP853", rtol=rtol, atol=atol)
        assert sol.success
        sides[side] = (ms, sol.y[:, -1].reshape(4, nm))
    key = lambda t: (-t[1]["norm"], 0 if t[0] == "L" else 1, -abs(t[1]["k"]))
    allm = [(s, m_, sides[s][1][:, i]) for s in ("L", "R") for i, m_ in enumerate(sides[s][0])]
    ins = sorted([t for t in allm if t[1]["kind"] == "in"], key=key)
    outs = sorted([t for t in allm if t[1]["kind"] == "out"], key=key)
    evs = [t for t in allm if t[1]["kind"] == "ev"]
    unk = outs + evs
    assert len(unk) == 4, (len(ins), len(outs), len(evs))
    A = np.array([(1 if s == "L" else -1) * col for s, _, col in unk]).T
    S = np.zeros((len(outs), len(ins)), dtype=complex)
    for a, (s, _, col) in enumerate(ins):
        S[:, a] = np.linalg.solve(A, (-1 if s == "L" else 1) * col)[:len(outs)]
    eta_i = np.diag([t[1]["norm"] for t in ins]); eta_o = np.diag([t[1]["norm"] for t in outs])
    return S, float(np.linalg.norm(S.conj().T @ eta_o @ S - eta_i, 2))


ident = lambda x, w, y: y

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    # ---------------- reference, computed without the main code
    # cutoff of the downstream side by direct maximisation
    from scipy.optimize import minimize_scalar
    vR, c2R = JCUR / NR, G * NR
    wmax = -minimize_scalar(lambda k: -(vR * k - np.sqrt(c2R * k * k + k ** 4 / 4)), bounds=(1e-6, 2 * np.sqrt(vR * vR - c2R)),
                            method="bounded", options=dict(xatol=1e-13)).fun
    fr = np.linspace(0.02, 0.98, 60); ws = fr * wmax
    ref_rel, ref_grad, res_rel, res_grad = [], [], [], []
    for w in ws:
        S1, r1 = smatrix(M_relative, ident, w); S2, r2 = smatrix(M_gradient, to_relative, w)
        ref_rel.append(np.abs(S1) ** 2); ref_grad.append(np.abs(S2) ** 2); res_rel.append(r1); res_grad.append(r2)
    ref_rel, ref_grad = np.array(ref_rel), np.array(ref_grad)
    # accuracy of the reference itself: other tolerance, other matching distance
    chk = []
    for i in (0, 1, 15, 30, 59):
        a = np.abs(smatrix(M_relative, ident, ws[i], X0=22.0, rtol=1e-13, atol=1e-16)[0]) ** 2
        b = np.abs(smatrix(M_gradient, to_relative, ws[i], X0=22.0, rtol=1e-13, atol=1e-16)[0]) ** 2
        chk.append(max(float(np.max(np.abs(a - ref_rel[i]))), float(np.max(np.abs(b - ref_grad[i])))))
    out = dict(omega_max=float(wmax), n_frequencies=60,
               reference=dict(gradient_vs_relative_max_abs=float(np.max(np.abs(ref_grad - ref_rel))),
                              gradient_vs_relative_max_rel=float(np.max(np.abs(ref_grad - ref_rel) / ref_rel)),
                              pseudo_unitarity_residual_relative=float(max(res_rel)),
                              pseudo_unitarity_residual_gradient=float(max(res_grad)),
                              change_with_tolerance_and_matching_distance=float(max(chk)),
                              largest_S2=float(ref_rel.max()), largest_S2_at_omega_over_max=float(fr[np.argmax(ref_rel.max(axis=(1, 2)))])))
    # closure at lower frequencies than the sweep (is 1/w a problem?)
    low = []
    for f in (1e-2, 1e-3, 1e-4):
        a = np.abs(smatrix(M_relative, ident, f * wmax)[0]) ** 2; b = np.abs(smatrix(M_gradient, to_relative, f * wmax)[0]) ** 2
        a2 = np.abs(smatrix(M_relative, ident, f * wmax, X0=20.0, rtol=1e-13, atol=1e-16)[0]) ** 2
        b2 = np.abs(smatrix(M_gradient, to_relative, f * wmax, X0=20.0, rtol=1e-13, atol=1e-16)[0]) ** 2
        rl = lambda p, q: float(np.max(np.abs(p - q) / q))
        low.append(dict(omega_over_max=f, largest_S2=float(a.max()),
                        gradient_vs_relative_rel_default_tolerance=rl(b, a), gradient_vs_relative_rel_tight_tolerance=rl(b2, a2),
                        relative_change_with_tolerance=rl(a, a2), gradient_change_with_tolerance=rl(b, b2)))
    out["reference_low_frequency"] = low
    print("reference done in %.0f s:" % (time.time() - t0), out["reference"]); print(low)

    # ---------------- the main code against the reference
    sys.path.insert(0, os.path.dirname(__file__))
    from fod_scattering import Background, scattering_matrix, omega_max
    bgA = Background(kind="const_g", nL=NL, nR=NR, j=JCUR, width=1.0, g=G)
    xs = np.linspace(-6, 6, 241); pr = profile(xs)
    out["background_definitions_agree"] = float(max(np.max(np.abs(pr[k] - getattr(bgA, k)(xs))) for k in ("n", "lam", "dlam", "v", "dv", "c2", "dc2")))
    out["omega_max_agrees"] = float(abs(wmax - omega_max(*bgA.side("R")[1:])))
    rows = []
    for N in (4000, 8000, 16000, 32000):
        cr = np.array([np.abs(scattering_matrix(bgA, w, L=40.0, N=N)["S"]) ** 2 for w in ws])
        cg = np.array([np.abs(scattering_matrix(bgA, w, L=40.0, N=N, formulation="gradient_omega")["S"]) ** 2 for w in ws])
        d = np.abs(cg - cr); i, b_, a_ = np.unravel_index(np.argmax(d), d.shape)
        er, eg = np.abs(cr - ref_rel), np.abs(cg - ref_rel)
        rows.append(dict(N=N, test7_quantity_max_abs_diff=float(d.max()), attained_at_omega_over_max=float(fr[i]), attained_at_element=[int(b_), int(a_)],
                         S2_of_that_element=float(ref_rel[i, b_, a_]), max_rel_diff=float(np.max(d / cr)),
                         error_relative_formulation_abs=float(er.max()), error_gradient_formulation_abs=float(eg.max()),
                         error_relative_formulation_rel=float(np.max(er / ref_rel)), error_gradient_formulation_rel=float(np.max(eg / ref_rel)),
                         diff_per_frequency=[float(x) for x in d.max(axis=(1, 2))],
                         rel_diff_per_frequency=[float(x) for x in (d / cr).max(axis=(1, 2))]))
        print(N, {k: v for k, v in rows[-1].items() if "per_frequency" not in k})
    out["collocation_vs_reference"] = rows
    q = lambda key: [float(np.log2(rows[i][key] / rows[i + 1][key])) for i in range(len(rows) - 1)]
    out["observed_orders"] = dict(test7_quantity=q("test7_quantity_max_abs_diff"), error_relative=q("error_relative_formulation_abs"),
                                  error_gradient=q("error_gradient_formulation_abs"))
    out["frequencies_over_max"] = [float(x) for x in fr]
    out["seconds"] = time.time() - t0
    print(out["observed_orders"], "bg", out["background_definitions_agree"])
    json.dump(out, open(os.path.join(ROOT, "results", "test7_independent.json"), "w"), indent=1)
