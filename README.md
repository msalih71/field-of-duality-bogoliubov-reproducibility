# Field of Duality — Bogoliubov Reproducibility

Reproducibility materials for the research manuscript **Locality and Symplectic Structure of Gradient Representations of Bogoliubov Dynamics**.

This repository is intended to contain **code, numerical data, verification tests and generated figures only**. The manuscript, review letters and journal correspondence are deliberately excluded.

## Scientific scope

The project studies the gradient representation of density–phase fluctuations in Gross–Pitaevskii/Bogoliubov theory. For the specified stationary one-dimensional backgrounds and reduced gradient phase space, it examines the differential-generator locality condition

```text
( g(x) n0(x) )' = 0
```

and compares relative-field and frequency-closed gradient representations for sonic-horizon scattering.

## Numerical reproducibility

The accompanying local release archive contains:

- `src/` — scattering, horizon, Evans, time evolution and independent checks.
- `derivations/` — symbolic and matrix-algebra validations.
- `tests/` — lightweight Python tests.
- `results/` — saved numerical output and test-7 reproduction.
- `figures/` — generated plots.

Install dependencies in an isolated Python environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python derivations/symbolic_checks.py
python src/test7_independent.py
```

**Important:** Some full parameter scans and time-dependent simulations are computationally expensive. A successful lightweight test suite must not be interpreted as reproducing all heavy computations.

## Acceptance test 7 — preserved failure

The original absolute acceptance threshold is `1e-7`. At `N=8000`, the test produces approximately `1.15e-7` and is **FAIL**, as originally specified. The discrepancy decreases with approximately fourth-order convergence on grids `N=4000, 8000, 16000, 32000`. An independently implemented adaptive shooting reference agrees with the two formulations at approximately `2.2e-11` across 60 frequencies. This supports the interpretation of finite-grid truncation error, not a change to the original acceptance criterion.

The implementation is an internal independent-code cross-check, **not a third-party replication**.

## Scope of stability claims

Complex-frequency Evans calculations offer numerical spectral exclusion only within the scanned region and numerical tolerance. They do not prove absolute stability, in particular below the stated positive-growth cutoff.

## Citation and licensing

This repository is a code/data companion to a manuscript in preparation/submission; it does not claim the article has been published. The Zenodo archive DOI supplied by the authors is **[10.5281/zenodo.24165601](https://doi.org/10.5281/zenodo.24165601)**. The [GitHub v1.0.0 release](https://github.com/msalih71/field-of-duality-bogoliubov-reproducibility/releases/tag/v1.0.0) is the corresponding tagged code/data release. This is **not** a DOI for the unpublished journal article. No open-source license is asserted in this GitHub repository until rights-holder approval.

## Authorship and contact

Mahgoub A. Salih and Arbab I. Arbab. Correspondence: m.salih@qu.edu.sa.

## Repository hygiene

No manuscript PDF/TeX, editor letters, reviewer material, private correspondence or publication submission metadata should be committed.
