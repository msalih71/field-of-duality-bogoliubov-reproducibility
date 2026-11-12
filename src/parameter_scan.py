"""Parameter scan: nine (j, n_R) pairs on each background family.
Records, per case, Mach numbers, omega_max, the worst pseudo-unitarity residual, the worst
relative representation difference, and the low-frequency ratio omega|beta|^2/T_H."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fod_scattering import Background, scattering_matrix, hawking_beta2, omega_max
np.seterr(over="ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for kind, form in (("const_g", "gradient_omega"), ("const_c", "gradient")):
    for j in (0.5, 0.6, 0.7):
        for nR in (0.3, 0.4, 0.5):
            cfg = dict(kind=kind, nL=1.0, nR=nR, j=j, width=1.0)
            bg = Background(**cfg)
            mL, mR = float(bg.mach(-40.0)), float(bg.mach(40.0))
            if not (mL < 1 < mR):
                rows.append(dict(kind=kind, j=j, nR=nR, mach_L=mL, mach_R=mR, transonic=False)); continue
            wm = omega_max(*bg.side("R")[1:])
            xh, kappa = bg.horizon(); TH = kappa / (2 * np.pi)
            res = rel = 0.0; shapes = set()
            for f in (0.05, 0.2, 0.5, 0.8, 0.95):
                a = scattering_matrix(bg, f * wm, L=40.0, N=8000)
                b = scattering_matrix(bg, f * wm, L=40.0, N=8000, formulation=form)
                A, B = np.abs(a["S"]) ** 2, np.abs(b["S"]) ** 2
                res = max(res, a["residual"]); rel = max(rel, float(np.max(np.abs(A - B) / A)))
                shapes.add(a["S"].shape)
            above = scattering_matrix(bg, 1.3 * wm, L=40.0, N=8000)
            low = scattering_matrix(bg, 0.02 * TH, L=40.0, N=8000)
            rows.append(dict(kind=kind, j=j, nR=nR, mach_L=mL, mach_R=mR, transonic=True,
                             omega_max=wm, kappa=float(kappa), omega_max_over_TH=wm / TH,
                             max_residual=res, max_rel_representation_diff=rel,
                             three_channel_below=bool(shapes == {(3, 3)}),
                             two_channel_above=bool(above["S"].shape == (2, 2)),
                             lowfreq_ratio_to_TH=0.02 * TH * hawking_beta2(low) / TH))
json.dump(rows, open(os.path.join(ROOT, "results", "parameter_scan.json"), "w"), indent=1)
for r in rows:
    print({k: (round(v, 4) if isinstance(v, float) and abs(v) > 1e-3 else v) for k, v in r.items()})
