# Release v1.0.0 — Reproducibility companion

This release contains the source code and verification outputs associated with the research manuscript **Locality and Symplectic Structure of Gradient Representations of Bogoliubov Dynamics**.

## Included

- `src/`: Bogoliubov scattering, independently implemented reference checks, Evans-function analysis, Gross–Pitaevskii time evolution, parameter scans, and table generation.
- `derivations/`: symbolic consistency checks and fixed-/multi-mode matrix-algebra checks.
- `tests/`: Python test suite.
- `results/`: saved JSON and NPZ outputs, including convergence and independent-reference comparisons.
- `figures/`: eight numbered research figures in PNG/PDF.
- `generated_tables/`: LaTeX tables and macros.

## Scientific caveats

- The original representation acceptance test 7 at `N=8000` exceeds its fixed `1e-7` tolerance (`1.1506628538882069e-7`); **the original test is FAIL**. The high-resolution trend in the archived JSON is approximately fourth-order.
- Independent-reference numerical agreement is not third-party replication or a proof of a general stability theorem.
- Evans-function scans provide numerical exclusions in specified domains; they do not establish unconditional stability.
- The single-horizon step cutoff is recorded as `0.1002542695` for its specified parameters. Comparisons against external captions need independent source-specific checks.

## Scope

The research manuscript, editorial letters and private correspondence are not included. This release is a code-and-data research artifact only.

## Citation

The code and data contributors are listed in `CITATION.cff` and `.zenodo.json`. Add the **actual** Zenodo DOI here only after archiving and verifying its record.

## Release prerequisites

Before clicking **Publish release**:

1. Confirm both authors' approval for public archival publication and decide the software/data license. Zenodo may apply a default license if none is specified.
2. Connect the GitHub account at https://zenodo.org/account/settings/github/ and enable the repository **before** publishing the GitHub release.
3. In GitHub choose **Releases → Draft a new release**, create tag `v1.0.0` from `main`, title it **Field of Duality reproducibility v1.0.0**, and use these release notes.
4. After publishing, verify that Zenodo created a record and DOI. If the archive fails, inspect Zenodo's error details before trying another version.

No DOI or GitHub Release is claimed to exist by this document.
