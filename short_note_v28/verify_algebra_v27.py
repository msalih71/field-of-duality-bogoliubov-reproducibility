"""Symbolic checks for the short locality note (v28).

This checks algebraic differentiation of the linearized equations only.
It does NOT prove the support-locality theorem, operator-domain claims,
or existence of a closed evolution generator.
"""
import sympy as sp

x = sp.symbols("x", real=True)
ell, hbar, mass, current = sp.symbols(
    "ell hbar mass current", nonzero=True
)
n = sp.Function("n")(x)
g = sp.Function("g")(x)
u = sp.Function("u")(x)
th = sp.Function("theta")(x)
v = current / n
p = ell * sp.diff(th, x)
q = ell * sp.diff(u, x)
a = hbar / (2 * mass)
D = lambda z: sp.diff(n * sp.diff(z, x), x) / n

# Time derivatives of the original relative fields at fixed x.
ut = -v * sp.diff(u, x) - a * D(th)
pt = -v * sp.diff(th, x) - 2 * g * n * u / hbar + a * D(u)
q_from = ell * sp.diff(ut, x)
p_from = ell * sp.diff(pt, x)
q_expected = -sp.diff(v * q, x) - a * sp.diff(
    sp.diff(n * p, x) / n, x
)
p_expected = -sp.diff(v * p, x) - 2 * (
    g * n * q + ell * sp.diff(g * n, x) * u
) / hbar + a * sp.diff(sp.diff(n * q, x) / n, x)

assert sp.simplify(sp.expand(q_from - q_expected)) == 0
assert sp.simplify(sp.expand(p_from - p_expected)) == 0
print("PASS: gradient differential expressions match differentiated equations")
