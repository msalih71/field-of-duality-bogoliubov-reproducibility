"""Verification suite and production runs.  Writes results/*.json and figures/*.

Run from the package root:   python3 src/run_all.py
Tolerances below were fixed before the production runs and are not tuned.
"""
import json
import os
import sys
import time

import numpy as np
np.seterr(over="ignore")   # cosh overflow far from a narrow transition is harmless (-> 0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from fod_scattering import (Background, scattering_matrix, step_scattering,
                            hawking_beta2, omega_max, asymptotic_modes)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CONFIG = dict(
    L=40.0, N=8000,
    bgA=dict(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0, g=1.0),
    bgB=dict(kind="const_c", nL=1.0, nR=0.4, j=0.6, width=1.0, c2L=1.0),
    bgD=dict(kind="const_g", nL=1.0, nR=0.4, j=0.2, width=1.0, g=1.0),
    step=dict(n=1.0, v=0.75, c2L=1.0, c2R=0.25),
    tol_pseudo_unitarity=1e-8,
    tol_representation=1e-7,
    tol_domain=1e-7,
)
bgA, bgB, bgD = (Background(**CONFIG[k]) for k in ("bgA", "bgB", "bgD"))
L, N = CONFIG["L"], CONFIG["N"]
out = dict(config=CONFIG, tests={})
T0 = time.time()

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "serif"})
BLUE, ORANGE, RED, GRAY = "#1f4e79", "#d9773f", "#c0392b", "#666666"


def S2(res):
    return np.abs(res["S"]) ** 2


# ------------------------------------------------------------------ test 1
# Homogeneous limit: uniform subsonic flow -> pure transmission, and the
# asymptotic roots satisfy the Paper-1 dispersion relation.
bgU = Background(kind="const_g", nL=1.0, nR=1.0, j=0.6, width=1.0)
dev, disp = 0.0, 0.0
for w in np.linspace(0.05, 2.0, 12):
    r = scattering_matrix(bgU, w, L=L, N=2000)
    target = np.array([[0.0, 1.0], [1.0, 0.0]])       # transmission only
    dev = max(dev, float(np.max(np.abs(S2(r) - target))))
    for mo in asymptotic_modes(w, 1.0, 0.6, 1.0, "L"):
        k = mo.k
        disp = max(disp, abs((w - 0.6 * k) ** 2 - (k ** 2 + 0.25 * k ** 4)))
out["tests"]["1_homogeneous_limit"] = dict(
    max_dev_from_pure_transmission=dev, max_dispersion_residual=float(disp),
    passed=bool(dev < 1e-10 and disp < 1e-10))

# ------------------------------------------------------------------ test 2
# Everywhere-subsonic inhomogeneous flow: 2x2 Euclidean-unitary S, no
# negative-norm channel.
dev, shapes = 0.0, set()
for w in np.linspace(0.05, 1.5, 12):
    r = scattering_matrix(bgD, w, L=L, N=N)
    shapes.add(r["S"].shape)
    dev = max(dev, float(np.linalg.norm(r["S"].conj().T @ r["S"] - np.eye(2), 2)))
    assert all(c["norm"] > 0 for c in r["ins"] + r["outs"])
out["tests"]["2_subsonic_null"] = dict(
    mach_L=float(bgD.mach(-L)), mach_R=float(bgD.mach(L)),
    S_shapes=sorted(map(list, shapes)), max_unitarity_dev=dev,
    passed=bool(dev < CONFIG["tol_pseudo_unitarity"] and shapes == {(2, 2)}))

# ------------------------------------------------------------------ test 3
# Smooth-to-step limit against the exact algebraic matching solution.
st = CONFIG["step"]
wm_step = omega_max(st["v"], st["c2R"])
w_step = np.array([0.1, 0.3, 0.6]) * wm_step
widths = [0.4, 0.2, 0.1, 0.05, 0.025]
rows = []
for wd in widths:
    bgS = Background(kind="g_step", nL=st["n"], nR=st["n"], j=st["n"] * st["v"],
                     width=wd, c2L=st["c2L"], c2R=st["c2R"])
    d = 0.0
    for w in w_step:
        rs = step_scattering(w, st["n"], st["v"], st["c2L"], st["c2R"])
        r = scattering_matrix(bgS, w, L=20.0, N=40000)
        d = max(d, float(np.max(np.abs(S2(r) - np.abs(rs["S"]) ** 2))))
    rows.append(dict(width=wd, max_abs_dev_S2=d))
step_resid = max(step_scattering(w, st["n"], st["v"], st["c2L"], st["c2R"])["residual"]
                 for w in np.linspace(0.02, 0.98, 40) * wm_step)
devs = [r["max_abs_dev_S2"] for r in rows]
out["tests"]["3_step_limit"] = dict(
    omega_max=wm_step, rows=rows, step_pseudo_unitarity_residual=step_resid,
    monotone_decrease=bool(all(devs[i + 1] < devs[i] for i in range(len(devs) - 1))),
    passed=bool(all(devs[i + 1] < devs[i] for i in range(len(devs) - 1))
                and step_resid < 1e-10))

# ------------------------------------------------------------------ test 4
# Grid convergence on background A (reference: N = 32000).
wmA = omega_max(*bgA.side("R")[1:])
wmB = omega_max(*bgB.side("R")[1:])
w_conv = 0.3 * wmA
ref = scattering_matrix(bgA, w_conv, L=L, N=32000)
conv = []
for n_ in (500, 1000, 2000, 4000, 8000):
    r = scattering_matrix(bgA, w_conv, L=L, N=n_)
    conv.append(dict(N=n_, h=2 * L / n_, residual=r["residual"],
                     err_S2=float(np.max(np.abs(S2(r) - S2(ref))))))
orders = [float(np.log2(conv[i]["err_S2"] / conv[i + 1]["err_S2"]))
          for i in range(len(conv) - 1)]
out["tests"]["4_grid_convergence"] = dict(
    omega=w_conv, rows=conv, observed_orders=orders, nominal_order=4,
    reference_residual=ref["residual"],
    passed=bool(min(orders[:3]) > 3.5))

# ------------------------------------------------------------------ test 5
# Domain-size independence on background A.
dom = []
base = scattering_matrix(bgA, w_conv, L=40.0, N=16000)
for L_ in (20.0, 30.0, 60.0):
    r = scattering_matrix(bgA, w_conv, L=L_, N=int(16000 * L_ / 40.0))
    dom.append(dict(L=L_, max_abs_change_S2=float(np.max(np.abs(S2(r) - S2(base))))))
out["tests"]["5_domain_size"] = dict(
    rows=dom, passed=bool(max(d["max_abs_change_S2"] for d in dom) < CONFIG["tol_domain"]))

# ----------------------------------------------------------- production sweep
def sweep(bg, wmax, formulation="relative", nw=60, lo=0.02, hi=0.98):
    ws = np.linspace(lo, hi, nw) * wmax
    S2s, res, flux, b2 = [], [], [], []
    for w in ws:
        r = scattering_matrix(bg, w, L=L, N=N, formulation=formulation)
        S2s.append(S2(r)); res.append(r["residual"]); flux.append(r["flux_dev"])
        b2.append(hawking_beta2(r))
    return ws, np.array(S2s), np.array(res), np.array(flux), np.array(b2)


wsA, S2A, resA, fluxA, b2A = sweep(bgA, wmA)
wsB, S2B, resB, fluxB, b2B = sweep(bgB, wmB)
# above omega_max: 2x2
above = []
for bg, wm in ((bgA, wmA), (bgB, wmB)):
    for f in (1.05, 1.5, 3.0):
        r = scattering_matrix(bg, f * wm, L=L, N=N)
        above.append(dict(kind=bg.kind, omega_over_max=f, shape=list(r["S"].shape),
                          residual=r["residual"]))

# ------------------------------------------------------------------ test 6
out["tests"]["6_pseudo_unitarity"] = dict(
    max_residual_A=float(resA.max()), max_residual_B=float(resB.max()),
    max_flux_nonconservation_A=float(fluxA.max()),
    max_flux_nonconservation_B=float(fluxB.max()),
    above_omega_max=above,
    passed=bool(max(resA.max(), resB.max()) < CONFIG["tol_pseudo_unitarity"]
                and all(a["shape"] == [2, 2] for a in above)))

# ------------------------------------------------------------------ test 7
# Agreement of the two representations; and what happens if the (g n0)' u
# term is dropped on background A.
_, S2Bg, resBg, _, _ = sweep(bgB, wmB, "gradient")
_, S2Ba, resBa, _, _ = sweep(bgB, wmB, "gradient_aug")
_, S2Aa, resAa, _, _ = sweep(bgA, wmA, "gradient_aug")
_, S2Ag, resAg, _, _ = sweep(bgA, wmA, "gradient")          # deliberately wrong
_, S2Aw, resAw, _, _ = sweep(bgA, wmA, "gradient_omega")    # u eliminated at fixed w
dA_omega = float(np.max(np.abs(S2Aw - S2A)))
dB_closed = float(np.max(np.abs(S2Bg - S2B)))
dB_aug = float(np.max(np.abs(S2Ba - S2B)))
dA_aug = float(np.max(np.abs(S2Aa - S2A)))
dA_trunc = float(np.max(np.abs(S2Ag - S2A)))
out["tests"]["7_representations"] = dict(
    B_closed_gradient_vs_relative=dB_closed,
    B_augmented_gradient_vs_relative=dB_aug,
    A_augmented_gradient_vs_relative=dA_aug,
    A_frequency_closed_gradient_vs_relative=dA_omega,
    residual_A_frequency_closed=float(resAw.max()),
    A_truncated_gradient_vs_relative=dA_trunc,
    A_truncated_max_pseudo_unitarity_residual=float(resAg.max()),
    residual_B_closed=float(resBg.max()), residual_A_aug=float(resAa.max()),
    passed=bool(max(dB_closed, dB_aug, dA_aug, dA_omega) < CONFIG["tol_representation"]))

# cost / accuracy comparison at equal grid (criterion 2 of the roadmap)
refB = scattering_matrix(bgB, 0.3 * wmB, L=L, N=32000)
cmp_rows = []
for n_ in (1000, 2000, 4000):
    row = dict(N=n_)
    for form in ("relative", "gradient"):
        t = time.perf_counter()
        for _ in range(5):
            r = scattering_matrix(bgB, 0.3 * wmB, L=L, N=n_, formulation=form)
        row[form] = dict(err_S2=float(np.max(np.abs(S2(r) - S2(refB)))),
                         residual=r["residual"],
                         seconds=(time.perf_counter() - t) / 5)
    cmp_rows.append(row)
out["representation_cost_accuracy_B"] = cmp_rows

# convergence of the representation difference itself (must vanish as h^4)
rep_conv = []
for n_ in (2000, 4000, 8000, 16000):
    ra = scattering_matrix(bgB, 0.3 * wmB, L=L, N=n_, formulation="relative")
    rb = scattering_matrix(bgB, 0.3 * wmB, L=L, N=n_, formulation="gradient")
    rc = scattering_matrix(bgA, 0.3 * wmA, L=L, N=n_, formulation="relative")
    rd = scattering_matrix(bgA, 0.3 * wmA, L=L, N=n_, formulation="gradient_aug")
    re_ = scattering_matrix(bgA, 0.3 * wmA, L=L, N=n_, formulation="gradient_omega")
    rep_conv.append(dict(N=n_, B_closed_diff=float(np.max(np.abs(S2(ra) - S2(rb)))),
                         A_frequency_closed_diff=float(np.max(np.abs(S2(rc) - S2(re_)))),
                         A_augmented_diff=float(np.max(np.abs(S2(rc) - S2(rd))))))
out["representation_difference_convergence"] = dict(
    rows=rep_conv,
    orders_B=[float(np.log2(rep_conv[i]["B_closed_diff"] / rep_conv[i + 1]["B_closed_diff"]))
              for i in range(3)],
    orders_A=[float(np.log2(rep_conv[i]["A_augmented_diff"] / rep_conv[i + 1]["A_augmented_diff"]))
              for i in range(3)])

# lowest sweep frequency, frequency-closed gradient system, two grids
lowf = []
for n_ in (8000, 16000):
    ra = scattering_matrix(bgA, 0.02 * wmA, L=L, N=n_)
    rb = scattering_matrix(bgA, 0.02 * wmA, L=L, N=n_, formulation="gradient_omega")
    lowf.append(dict(N=n_, max_abs_diff_S2=float(np.max(np.abs(S2(ra) - S2(rb)))),
                     max_rel_diff_S2=float(np.max(np.abs(S2(ra) - S2(rb)) / S2(ra)))))
out["frequency_closed_lowest_frequency"] = lowf

# ------------------------------------------------------------------ test 0
# Stationary GP residual of the inverse-designed backgrounds, evaluated with
# independent finite differences of sqrt(n0) (not the analytic derivatives).
gp_rows = []
for name, bg in (("A", bgA), ("B", bgB), ("D", bgD)):
    for npts in (2001, 4001, 8001):
        xg = np.linspace(-20, 20, npts); hg = xg[1] - xg[0]
        sq = np.sqrt(bg.n(xg))
        d2 = (-sq[4:] + 16 * sq[3:-1] - 30 * sq[2:-2] + 16 * sq[1:-3] - sq[:-4]) / (12 * hg ** 2)
        xi = xg[2:-2]
        mu_local = 0.5 * bg.v(xi) ** 2 + bg.potential(xi) + bg.c2(xi) - 0.5 * d2 / sq[2:-2]
        gp_rows.append(dict(background=name, points=npts,
                            max_abs_gp_residual=float(np.max(np.abs(mu_local))),
                            current_spread=float(np.ptp(bg.n(xi) * bg.v(xi)))))
out["tests"]["0_background_gp_residual"] = dict(
    rows=gp_rows,
    passed=bool(all(r["max_abs_gp_residual"] < 1e-8 for r in gp_rows if r["points"] == 8001)))

# ------------------------------------------------------------------ test 8
# Sensitivity to the horizon width; low-frequency comparison with kappa/2pi.
sens = []
for kind, base_cfg in (("const_g", CONFIG["bgA"]), ("const_c", CONFIG["bgB"])):
    for wd in (0.5, 1.0, 2.0, 4.0):
        cfg = dict(base_cfg); cfg["width"] = wd
        bg = Background(**cfg)
        xh, kappa = bg.horizon()
        TH = kappa / (2 * np.pi)
        wm = omega_max(*bg.side("R")[1:])
        Lw = max(L, 12.0 * wd)
        ws = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0]) * TH
        vals, resm = [], 0.0
        for w in ws:
            r = scattering_matrix(bg, w, L=Lw, N=int(N * Lw / L))
            vals.append(w * hawking_beta2(r)); resm = max(resm, r["residual"])
        planck = ws / np.expm1(ws / TH)
        sens.append(dict(kind=kind, width=wd, x_h=xh, kappa=kappa, T_H=TH,
                         omega_max=wm, omega_max_over_T_H=wm / TH,
                         omega_over_TH=(ws / TH).tolist(),
                         omega_beta2=vals, planck_omega_n=planck.tolist(),
                         lowfreq_ratio_to_TH=vals[0] / TH,
                         max_rel_dev_from_planck=float(
                             np.max(np.abs(np.array(vals) / planck - 1))),
                         max_residual=resm))
out["tests"]["8_sensitivity"] = dict(rows=sens, passed=True,
                                     note="descriptive; no acceptance threshold")

out["tests"]["9_dynamical_stability"] = dict(
    passed=None, note="Run separately: src/stability_scan.py -> results/stability.json "
                      "(complex-frequency search). No GP time evolution has been run.")

out["omega_max"] = dict(A=wmA, B=wmB)
out["horizon"] = dict(A=dict(zip(("x_h", "kappa"), map(float, bgA.horizon()))),
                      B=dict(zip(("x_h", "kappa"), map(float, bgB.horizon()))))
out["runtime_seconds"] = time.time() - T0
out["versions"] = dict(numpy=np.__version__, python=sys.version.split()[0],
                       matplotlib=matplotlib.__version__)
import scipy
out["versions"]["scipy"] = scipy.__version__

with open(os.path.join(RES, "verification.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
np.savez(os.path.join(RES, "sweeps.npz"), wsA=wsA, S2A=S2A, resA=resA, b2A=b2A,
         wsB=wsB, S2B=S2B, resB=resB, b2B=b2B, S2Bg=S2Bg, S2Aa=S2Aa, S2Ag=S2Ag,
         resAg=resAg)

# -------------------------------------------------------------------- figures
x = np.linspace(-8, 8, 801)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, bg, lab in ((axs[0], bgA, "(a) constant $g$"),
                    (axs[1], bgB, "(b) constant $c$")):
    ax.plot(x, bg.n(x), color=GRAY, ls=":", label=r"$n_0/n_{\rm L}$")
    ax.plot(x, bg.v(x), color=ORANGE, label=r"$v_0$")
    ax.plot(x, np.sqrt(bg.c2(x)), color=BLUE, label=r"$c$")
    xh, _ = bg.horizon()
    ax.axvline(xh, color=RED, lw=0.8, ls="--")
    ax.set_xlabel(r"$x$  (units of $\hbar/mc_{\rm L}$)")
    ax.set_title(lab, fontsize=9, loc="left")
axs[0].set_ylabel("background")
axs[0].legend(frameon=False, loc="center left")
fig.savefig(os.path.join(FIG, "fig1_backgrounds.pdf")); fig.savefig(os.path.join(FIG, "fig1_backgrounds.png")); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
for ax, which, lab in ((axs[0], "L", "(a) subsonic side"), (axs[1], "R", "(b) supersonic side")):
    n_, v_, c2_ = bgA.side(which)
    k = np.linspace(-4, 4, 1601)
    wB = np.sqrt(c2_ * k ** 2 + 0.25 * k ** 4)
    ax.plot(k, v_ * k + wB, color=BLUE, label="positive norm")
    ax.plot(k, v_ * k - wB, color=RED, label="negative norm")
    w0 = 0.5 * wmA
    ax.axhline(w0, color=GRAY, lw=0.8, ls="--")
    for mo in asymptotic_modes(w0, n_, v_, c2_, which):
        if mo.norm != 0:
            ax.plot(mo.k.real, w0, marker=">" if mo.vg > 0 else "<", ms=6,
                    color=BLUE if mo.norm > 0 else RED,
                    mfc="white" if mo.kind == "out" else None)
    ax.set_ylim(-0.5, 2.0); ax.set_xlabel(r"$k$"); ax.set_title(lab, fontsize=9, loc="left")
axs[0].set_ylabel(r"$\omega$"); axs[0].legend(frameon=False, loc="upper left")
axs[1].axhline(wmA, color=RED, lw=0.6, ls=":"); axs[1].text(-3.9, wmA + 0.04, r"$\omega_{\max}$", color=RED)
fig.savefig(os.path.join(FIG, "fig2_dispersion_channels.pdf")); fig.savefig(os.path.join(FIG, "fig2_dispersion_channels.png")); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
in_lab = [r"$u^{\rm in}$", r"$d_1^{\rm in}$", r"$d_2^{\rm in}$ (neg.)"]
out_lab = ["Hawking", r"$d_1^{\rm out}$", r"partner (neg.)"]
for ax, ws, S2s, wm, lab in ((axs[0], wsA, S2A, wmA, "(a) constant $g$"),
                             (axs[1], wsB, S2B, wmB, "(b) constant $c$")):
    cols = [BLUE, ORANGE, RED]; ls = ["-", "--", ":"]
    for b in range(3):
        for a in range(3):
            ax.semilogy(ws / wm, S2s[:, b, a], color=cols[a], ls=ls[b], lw=1.1,
                        label=f"{out_lab[b]} $\\leftarrow$ {in_lab[a]}")
    ax.set_xlabel(r"$\omega/\omega_{\max}$"); ax.set_title(lab, fontsize=9, loc="left")
    ax.set_ylim(1e-6, 30)
axs[0].set_ylabel(r"$|S_{ba}|^2$")
axs[1].legend(frameon=False, fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.savefig(os.path.join(FIG, "fig3_S_elements.pdf")); fig.savefig(os.path.join(FIG, "fig3_S_elements.png")); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
axs[0].semilogy(wsA / wmA, resA, color=BLUE, label="constant $g$")
axs[0].semilogy(wsB / wmB, resB, color=ORANGE, label="constant $c$")
axs[0].axhline(CONFIG["tol_pseudo_unitarity"], color=GRAY, ls="--", lw=0.8)
axs[0].set_xlabel(r"$\omega/\omega_{\max}$")
axs[0].set_ylabel(r"$\|S^\dagger\eta S-\eta\|_2$"); axs[0].legend(frameon=False)
axs[0].set_title("(a)", fontsize=9, loc="left")
hs = np.array([c["h"] for c in conv])
axs[1].loglog(hs, [c["err_S2"] for c in conv], "o-", color=BLUE, label=r"error in $|S|^2$")
axs[1].loglog(hs, [c["residual"] for c in conv], "s-", color=ORANGE, label="pseudo-unitarity residual")
axs[1].loglog(hs, conv[0]["err_S2"] * (hs / hs[0]) ** 4, color=GRAY, ls="--", label=r"$h^4$")
axs[1].set_xlabel(r"grid spacing $h$"); axs[1].legend(frameon=False, fontsize=7)
axs[1].set_title("(b)", fontsize=9, loc="left")
fig.savefig(os.path.join(FIG, "fig4_residual_convergence.pdf")); fig.savefig(os.path.join(FIG, "fig4_residual_convergence.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.6))
tiny = 1e-18
ax.semilogy(wsB / wmB, np.max(np.abs(S2Bg - S2B), axis=(1, 2)) + tiny, color=ORANGE,
            label="const. $c$: closed $(p,q)$")
ax.semilogy(wsA / wmA, np.max(np.abs(S2Aa - S2A), axis=(1, 2)) + tiny, color=BLUE,
            label="const. $g$: $(p,q,u)$ augmented")
ax.semilogy(wsA / wmA, np.max(np.abs(S2Aw - S2A), axis=(1, 2)) + tiny, color=GRAY, ls="-.",
            label=r"const. $g$: $(p,q)$, $u$ eliminated at fixed $\omega$")
ax.semilogy(wsA / wmA, np.max(np.abs(S2Ag - S2A), axis=(1, 2)) + tiny, color=RED, ls="--",
            label=r"const. $g$: $(p,q)$ with $(gn_0)'u$ dropped")
ax.set_xlabel(r"$\omega/\omega_{\max}$")
ax.set_ylabel(r"$\max_{ba}\,|\,|S_{ba}|^2_{\rm grad}-|S_{ba}|^2_{\rm rel}|$")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
fig.savefig(os.path.join(FIG, "fig5_representations.pdf")); fig.savefig(os.path.join(FIG, "fig5_representations.png")); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
xx = np.linspace(0.01, 1.05, 200)
for ax, kind, lab in ((axs[0], "const_g", "(a) constant $g$"), (axs[1], "const_c", "(b) constant $c$")):
    ax.plot(xx, xx / np.expm1(xx), color="k", lw=1.0, label="Planck at $T_H$")
    for s_, col in zip([s for s in sens if s["kind"] == kind], [RED, ORANGE, BLUE, GRAY]):
        ax.plot(s_["omega_over_TH"], np.array(s_["omega_beta2"]) / s_["T_H"], "o-", ms=3,
                color=col, lw=0.8, label=f"width {s_['width']:g}, $\\omega_{{\\max}}/T_H$={s_['omega_max_over_T_H']:.1f}")
    ax.set_xlabel(r"$\omega/T_H$"); ax.set_title(lab, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=6.5)
axs[0].set_ylabel(r"$\omega|\beta_\omega|^2/T_H$")
fig.savefig(os.path.join(FIG, "fig6_width_sensitivity.pdf")); fig.savefig(os.path.join(FIG, "fig6_width_sensitivity.png")); plt.close(fig)

for name, t in out["tests"].items():
    print(f"{name:28s} passed={t['passed']}")
print("runtime", round(out["runtime_seconds"], 1), "s")
