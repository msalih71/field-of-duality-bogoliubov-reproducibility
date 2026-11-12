"""Symbolic verification of the identities used in the proofs (SymPy).

Each check prints the simplified residual, which must be 0.
Units: hbar and m are kept symbolic.  Run:  python3 symbolic_checks.py
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
hbar, m, ell, j = sp.symbols("hbar m ell j", positive=True)
n0 = sp.Function("n0", positive=True)(x)
g = sp.Function("g", positive=True)(x)
v0 = j / n0                                   # stationary current conservation
a = hbar / (2 * m)

Dcal = lambda f: sp.diff(n0 * sp.diff(f, x), x) / n0      # weighted Laplacian
Dt = lambda f: sp.diff(f, t) + v0 * sp.diff(f, x)         # convective derivative

results = {}

# ---------------------------------------------------------------- check 1
# Frechet derivative of the quantum pressure Q[n] = -(hbar^2/2m) (sqrt n)''/sqrt n
eps = sp.symbols("epsilon")
uf = sp.Function("u")(x, t)
n_pert = n0 * (1 + 2 * eps * uf)              # delta n = 2 n0 u
Q = -hbar**2 / (2 * m) * sp.diff(sp.sqrt(n_pert), x, 2) / sp.sqrt(n_pert)
dQ = sp.diff(Q, eps).subs(eps, 0)
results["1 quantum-pressure variation  dQ + (hbar^2/2m) D u"] = sp.simplify(
    dQ + hbar**2 / (2 * m) * Dcal(uf))

# ---------------------------------------------------------------- check 2
# Linearised Madelung equations reproduce Eqs. (48)-(49)
th = sp.Function("vartheta")(x, t)
V = sp.Function("V")(x)
mu = sp.symbols("mu")
theta0 = sp.Function("theta0")(x)
# background relations
bg = {sp.diff(theta0, x): m * v0 / hbar}
n = n0 * (1 + 2 * eps * uf)
theta = theta0 - mu * t / hbar + eps * th
vel = hbar / m * sp.diff(theta, x)
cont = sp.diff(n, t) + sp.diff(n * vel, x)
Qn = -hbar**2 / (2 * m) * sp.diff(sp.sqrt(n), x, 2) / sp.sqrt(n)
bern = hbar * sp.diff(theta, t) + m * vel**2 / 2 + V + g * n + Qn
cont1 = sp.diff(cont, eps).subs(eps, 0).doit()
bern1 = sp.diff(bern, eps).subs(eps, 0).doit()
cont1 = cont1.subs(sp.diff(theta0, x, 2), sp.diff(m * v0 / hbar, x)).subs(bg)
bern1 = bern1.subs(sp.diff(theta0, x, 2), sp.diff(m * v0 / hbar, x)).subs(bg)
eq48 = Dt(uf) + a * Dcal(th)
eq49 = Dt(th) + 2 * g * n0 / hbar * uf - a * Dcal(uf)
results["2a continuity  ->  Eq.(48)"] = sp.simplify(cont1 / (2 * n0) - eq48)
results["2b Bernoulli   ->  Eq.(49)"] = sp.simplify(bern1 / hbar - eq49)

# ---------------------------------------------------------------- check 3
# Gradient equations (52)-(53) and the single non-closed term (g n0)' u
ut = -v0 * sp.diff(uf, x) - a * Dcal(th)                         # du/dt
tht = -v0 * sp.diff(th, x) - 2 * g * n0 / hbar * uf + a * Dcal(uf)  # dtheta/dt
p = ell * sp.diff(th, x)
q = ell * sp.diff(uf, x)
lhs_q = ell * sp.diff(ut, x) + sp.diff(v0 * q, x)
rhs_q = -a * sp.diff(sp.diff(n0 * p, x) / n0, x)
lhs_p = ell * sp.diff(tht, x) + sp.diff(v0 * p, x)
rhs_p = (-2 / hbar * (g * n0 * q + ell * sp.diff(g * n0, x) * uf)
         + a * sp.diff(sp.diff(n0 * q, x) / n0, x))
results["3a Eq.(52)"] = sp.simplify(lhs_q - rhs_q)
results["3b Eq.(53)"] = sp.simplify(lhs_p - rhs_p)

# ---------------------------------------------------------------- check 4
# Local conservation law behind the symplectic form and the flux:
#   d_t [ n0 w ] + d_x [ j w - a n0 (th1 th2' - th2 th1' + u1 u2' - u2 u1') ] = 0
u1, u2, t1, t2 = [sp.Function(s)(x, t) for s in ("u1", "u2", "theta1", "theta2")]
rhs_u = lambda u_, th_: -v0 * sp.diff(u_, x) - a * Dcal(th_)
rhs_t = lambda u_, th_: (-v0 * sp.diff(th_, x) - 2 * g * n0 / hbar * u_
                         + a * Dcal(u_))
w = u1 * t2 - u2 * t1
dw_dt = (rhs_u(u1, t1) * t2 + u1 * rhs_t(u2, t2)
         - rhs_u(u2, t2) * t1 - u2 * rhs_t(u1, t1))
J = j * w - a * n0 * (t1 * sp.diff(t2, x) - t2 * sp.diff(t1, x)
                      + u1 * sp.diff(u2, x) - u2 * sp.diff(u1, x))
results["4 symplectic current conservation"] = sp.simplify(
    n0 * dw_dt + sp.diff(J, x))

# ---------------------------------------------------------------- check 5
# Weighted gradient q_w = d_x (w u):  coefficients of the primitive W = w u
wgt = sp.Function("w", positive=True)(x)
W = sp.Function("W")(x, t)                    # W = w u, so u = W / w
u_of_W = W / wgt
ut_W = -v0 * sp.diff(u_of_W, x) - a * Dcal(th)
dq_w = sp.diff(wgt * ut_W, x)                 # d_t q_w
coef_q = sp.diff(dq_w, W)                     # coefficient of undifferentiated W
target_q = sp.diff(v0 * sp.diff(wgt, x) / wgt, x)
results["5a W-coefficient in q_w equation"] = sp.simplify(coef_q - target_q)
tht_W = (-v0 * sp.diff(th, x) - 2 * g * n0 / hbar * u_of_W + a * Dcal(u_of_W))
dp = ell * sp.diff(tht_W, x)
coef_p = sp.diff(dp, W)
target_p = -ell * (sp.diff(2 * g * n0 / (hbar * wgt), x)
                   + a * sp.diff(sp.diff(n0 * sp.diff(wgt, x) / wgt**2, x) / n0, x))
results["5b W-coefficient in p equation"] = sp.simplify(coef_p - target_p)

# ---------------------------------------------------------------- check 6
# Split-quaternion relations for the 2x2 real matrix algebra
N = sp.Matrix([[0, 0], [1, 0]])               # action of the dual unit, Eq.(34)
I2 = sp.eye(2)
qi = N.T - N
qj = N.T + N
qk = N.T * N - N * N.T
results["6a i^2 = -1"] = (qi * qi + I2).norm()
results["6b j^2 = +1"] = (qj * qj - I2).norm()
results["6c k^2 = +1"] = (qk * qk - I2).norm()
results["6d ij = k"] = (qi * qj - qk).norm()
results["6e rank of span{1,i,j,k}"] = 4 - sp.Matrix(
    [list(M) for M in (I2, qi, qj, qk)]).rank()

# ---------------------------------------------------------------- check 7
# Inhomogeneous quadratic Hamiltonian:  2 hbar n0 u_t = dH/dtheta,  2 hbar n0 theta_t = -dH/du
ux, thx = sp.diff(uf, x), sp.diff(th, x)
Hd = (hbar**2 * n0 / (2 * m)) * (thx**2 + ux**2) + 2 * g * n0**2 * uf**2 + 2 * hbar * j * uf * thx
dH_dth = sp.diff(Hd, th) - sp.diff(sp.diff(Hd, thx), x)
dH_du = sp.diff(Hd, uf) - sp.diff(sp.diff(Hd, ux), x)
results["7a Hamilton equation for u"] = sp.simplify(2 * hbar * n0 * ut - dH_dth)
results["7b Hamilton equation for theta"] = sp.simplify(2 * hbar * n0 * tht + dH_du)

ok = True
for name, val in results.items():
    flag = "OK " if val == 0 else "FAIL"
    ok &= (val == 0)
    print(f"[{flag}] {name}: residual = {val}")
print("ALL SYMBOLIC CHECKS PASSED" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
