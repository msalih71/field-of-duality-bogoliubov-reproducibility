"""Independent cutoff check from the scalar laboratory-frame dispersion.

Uses no code from fod_scattering.py and does not call the analytic omega_max
helper. The max is for the positive-k branch v*k - sqrt(c^2 k^2 + k^4/4).
"""
import json
from pathlib import Path
from scipy.optimize import minimize_scalar

v, c = 0.75, 0.5

def positive_branch(k):
    return v*k - (c*c*k*k + k**4/4.0)**0.5

res = minimize_scalar(lambda k: -positive_branch(k), bounds=(1.e-12, 4.0),
                      method="bounded", options={"xatol": 1.e-13})
assert res.success
saved = json.loads((Path(__file__).resolve().parent.parent / 'results' / 'step_published_check.json').read_text())
reference = float(saved['omega_max'])
observed = float(positive_branch(res.x))
assert abs(observed - reference) < 1.e-11, (observed, reference)
output = {'v0': v, 'c_downstream': c, 'k_turning': float(res.x),
          'omega_cutoff_independent': observed,
          'omega_cutoff_analytic_saved': reference,
          'absolute_difference': abs(observed-reference),
          'method': 'scipy.optimize.minimize_scalar bounded, independent dispersion implementation',
          'scope': 'step benchmark parameters only; does not decide provenance of published caption 0.316'}
(Path(__file__).resolve().parent.parent / 'results' / 'cutoff_independent.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps(output, indent=2))
