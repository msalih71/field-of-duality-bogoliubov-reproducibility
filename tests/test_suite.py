"""Fast regression tests (pytest).  Run:  python3 -m pytest -q tests"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fod_scattering import (Background, scattering_matrix, step_scattering,
                            asymptotic_modes, flux_matrix, omega_max)

bgA = Background(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0)
bgB = Background(kind="const_c", nL=1.0, nR=0.4, j=0.6, width=1.0)


def test_mode_flux_formula():
    """J = 4 n0 k^2 |w - v k| sigma v_g for the un-normalised mode (Theorem 5 ii)."""
    n, v, c2, w = 0.4, 1.5, 0.4, 0.3
    M = flux_matrix(n, n * v)
    for k in np.roots([0.25, 0.0, c2 - v * v, 2 * w * v, -w * w]):
        assert abs(k.imag) < 1e-9
        k = k.real
        vec = np.array([k * k, 1j * k ** 3, -2j * (w - v * k), 2 * k * (w - v * k)])
        J = (vec.conj() @ M @ vec).real
        vg = (2 * v * (w - v * k) + 2 * c2 * k + k ** 3) / (2 * (w - v * k))
        assert abs(J - 4 * n * k * k * abs(w - v * k) * np.sign(w - v * k) * vg) < 1e-10 * abs(J)


def test_channel_counting():
    wm = omega_max(*bgA.side("R")[1:])
    below = scattering_matrix(bgA, 0.5 * wm, N=1000)
    above = scattering_matrix(bgA, 1.5 * wm, N=1000)
    assert below["S"].shape == (3, 3) and above["S"].shape == (2, 2)
    assert np.allclose(np.diag(below["eta_in"]), [1, 1, -1])
    assert np.allclose(np.diag(below["eta_out"]), [1, 1, -1])


def test_homogeneous_limit():
    bgU = Background(kind="const_g", nL=1.0, nR=1.0, j=0.6)
    r = scattering_matrix(bgU, 0.7, N=1000)
    assert np.allclose(np.abs(r["S"]) ** 2, [[0, 1], [1, 0]], atol=1e-10)


def test_pseudo_unitarity_converges_fourth_order():
    res = [scattering_matrix(bgA, 0.25, N=n)["residual"] for n in (1000, 2000, 4000)]
    assert res[2] < 1e-6
    assert 3.5 < np.log2(res[0] / res[1]) < 4.5 and 3.5 < np.log2(res[1] / res[2]) < 4.5


def test_representations_agree():
    a = np.abs(scattering_matrix(bgB, 0.15, N=4000)["S"]) ** 2
    b = np.abs(scattering_matrix(bgB, 0.15, N=4000, formulation="gradient")["S"]) ** 2
    assert np.max(np.abs(a - b)) < 1e-6
    c = np.abs(scattering_matrix(bgA, 0.25, N=4000)["S"]) ** 2
    for form in ("gradient_aug", "gradient_omega"):
        d = np.abs(scattering_matrix(bgA, 0.25, N=4000, formulation=form)["S"]) ** 2
        assert np.max(np.abs(c - d)) < 1e-6


def test_dropping_the_coupling_term_is_wrong():
    c = np.abs(scattering_matrix(bgA, 0.25, N=4000)["S"]) ** 2
    d = scattering_matrix(bgA, 0.25, N=4000, formulation="gradient")
    assert np.max(np.abs(c - np.abs(d["S"]) ** 2)) > 1e-2
    assert d["residual"] > 1e-2


def test_step_solution_is_pseudo_unitary():
    wm = omega_max(0.75, 0.25)
    for f in (0.1, 0.5, 0.9):
        assert step_scattering(f * wm, 1.0, 0.75, 1.0, 0.25)["residual"] < 1e-10


def test_evans_determinant_matches_subspace_indicator():
    """|D(w)| of the analytic 2-form determinant equals d(w)/2 of the QR-based indicator."""
    from evans_winding import Evans
    from stability_scan import Transporter
    bg = Background(kind="const_g", nL=1.0, nR=0.4, j=0.6, width=1.0)
    ev, T = Evans(bg, 14.0), Transporter(bg, 14.0)
    for w in (3 + 0.001j, -6 + 0.5j):
        assert abs(2 * abs(ev.D(np.array([w]))[0]) - T.d(w)) < 1e-6
