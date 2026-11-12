"""Three supplementary checks whose result files are read by make_tables_extra.py.

(1) results/step_published_check.json
    Exact step-horizon solution against the low-frequency closed forms of Recati,
    Pavloff & Carusotto, Phys. Rev. A 80, 043603 (2009), Eqs. (21), (28), (48), for
    v0/c_u = 0.75, v0/c_d = 1.5; the cutoff from their Eqs. (4)-(5); and the two
    wave-packet frequencies of their Fig. 4, which bracket the cutoff.
(2) results/parameter_scan_worst_case_grids.json
    Pseudo-unitarity residual of the worst case of the parameter scan at three grids.
(3) results/representation_sweep_two_grids.json
    Gradient formulations against the relative-field formulation over the full
    frequency sweep at two grids.
Needs results/parameter_scan.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fod_scattering import Background, scattering_matrix, step_scattering, omega_max
np.seterr(over="ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save = lambda name, obj: json.dump(obj, open(os.path.join(ROOT, "results", name), "w"), indent=1)

# ------------------------------------------------------------------ (1)
cu, v0, cd = 1.0, 0.75, 0.5                     # units: m = hbar = c_u = 1, so mu_u = 1, xi_u = 1
hyd = (v0 ** 2 / cu ** 2 - cd ** 2 / cu ** 2) ** 1.5
pub = dict(
    Eq21_omega_Sd1d1_sq=(cu / cd) * (cu - v0) * (cu - cd) / ((cu + v0) * (cu + cd)) * hyd / 2,
    Eq28_omega_Sud2_sq=(cu - v0) / (cu + v0) * cu ** 2 / (cu ** 2 - cd ** 2) * hyd * 2,
    Eq48_abs_Suu_Sd1u=np.sqrt(cu / cd) * (cu - v0) / (cu + v0) * (cd + v0) / (cu + v0),
    Eq48_abs_Suu_Sd2u=np.sqrt(cu / cd) * (cu - v0) / (cu + v0) * abs(cd - v0) / (cu + v0))
rows = []
for w in (1e-5, 1e-4, 1e-3, 1e-2):
    S = step_scattering(w, 1.0, v0, cu ** 2, cd ** 2)["S"]        # channel order: u, d1, d2
    num = dict(Eq21_omega_Sd1d1_sq=w * abs(S[1, 1]) ** 2, Eq28_omega_Sud2_sq=w * abs(S[0, 2]) ** 2,
               Eq48_abs_Suu_Sd1u=abs(S[0, 0] * S[1, 0]), Eq48_abs_Suu_Sd2u=abs(S[0, 0] * S[2, 0]))
    r = dict(omega=w, **{k: float(x) for k, x in num.items()})
    r.update({k + "_ratio": float(num[k] / pub[k]) for k in pub})
    rows.append(r)
# cutoff: their Eq. (5) for K, then Omega = -omega_B^-(K) of Eq. (4)
M = v0 / cd
K = (1.0 / (1.0 / cd)) * np.sqrt(-2 + M * M / 2 + (M / 2) * np.sqrt(8 + M * M))     # K = (xi_d K) / xi_d, xi_d = 1/c_d
Omega = v0 * K - cd * K * np.sqrt(1 + 0.25 * (K / cd) ** 2)
wB = lambda q: cd * abs(q) * np.sqrt(1 + 0.25 * (q / cd) ** 2) + v0 * q             # positive-frequency wave at wavevector q
fig4 = dict(q_below=-1.2, q_above=-1.35, omega_q_below=float(wB(-1.2)), omega_q_above=float(wB(-1.35)),
            cutoff_from_Eqs_4_5=float(Omega), cutoff_code=float(omega_max(v0, cd ** 2)),
            value_printed_in_Fig3_caption=0.316, sqrt_of_cutoff=float(np.sqrt(Omega)),
            bracket_contains_cutoff=bool(wB(-1.2) < Omega < wB(-1.35)),
            bracket_contains_printed_value=bool(wB(-1.2) < 0.316 < wB(-1.35)))
save("step_published_check.json", dict(parameters=dict(v0_over_cu=v0 / cu, v0_over_cd=v0 / cd),
                                       published={k: float(x) for k, x in pub.items()},
                                       omega_max=float(omega_max(v0, cd ** 2)), rows=rows, fig4_bracket=fig4))
print("step:", {k: round(rows[0][k + "_ratio"], 6) for k in pub}); print("fig4:", fig4)

# ------------------------------------------------------------------ (2)
ps = [r for r in json.load(open(os.path.join(ROOT, "results", "parameter_scan.json"))) if r["transonic"]]
worst = max(ps, key=lambda r: r["max_residual"])
bg = Background(kind=worst["kind"], nL=1.0, nR=worst["nR"], j=worst["j"], width=1.0)
wm = omega_max(*bg.side("R")[1:])
grids = [dict(N=N, max_residual=float(max(scattering_matrix(bg, f * wm, L=40.0, N=N)["residual"]
                                          for f in (0.05, 0.2, 0.5, 0.8, 0.95)))) for N in (8000, 16000, 32000)]
save("parameter_scan_worst_case_grids.json", grids)
print("worst case:", worst["kind"], worst["j"], worst["nR"], grids)

# ------------------------------------------------------------------ (3)
bgA = Background(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0)
bgB = Background(kind="const_c", nL=1.0, nR=0.4, j=0.6, width=1.0, c2L=1.0)
out = {}
for N in (8000, 16000):
    rec = {}
    for tag, bg_, forms in (("A", bgA, ("gradient_aug", "gradient_omega")), ("B", bgB, ("gradient",))):
        wm_ = omega_max(*bg_.side("R")[1:])
        acc = {f: [0.0, 0.0, 0.0] for f in forms}
        for fr in np.linspace(0.02, 0.98, 60):
            a = scattering_matrix(bg_, fr * wm_, L=40.0, N=N); A = np.abs(a["S"]) ** 2
            for f in forms:
                b = scattering_matrix(bg_, fr * wm_, L=40.0, N=N, formulation=f); B = np.abs(b["S"]) ** 2
                acc[f] = [max(acc[f][0], float(np.max(np.abs(A - B)))), max(acc[f][1], float(np.max(np.abs(A - B) / A))),
                          max(acc[f][2], float(b["residual"]))]
        for f in forms:
            rec[f"{tag}:{f}"] = dict(max_abs=acc[f][0], max_rel=acc[f][1], max_residual=acc[f][2])
    out[str(N)] = rec
    print(N, {k: "%.3e" % v["max_abs"] for k, v in rec.items()})
save("representation_sweep_two_grids.json", out)
