# Short locality note (v28): reproducibility and scope

## Paper and purpose

**Locality Obstructions in Gradient Representations of Bogoliubov Dynamics** is a focused eight-page mathematical note. It is distinct from the extended paper **Locality and Symplectic Structure of Gradient Representations of Bogoliubov Dynamics**, which led to the full numerical repository.

This directory provides the symbolic consistency check used by the focused note. **Neither the full v28 manuscript source nor the manuscript PDF is deposited here**, in keeping with the author's repository policy.

## Reproduce the algebraic step

Requirements: Python 3 and SymPy.

From the repository root:

```bash
python -m pip install sympy
python short_note_v28/verify_algebra_v27.py
```

Expected output:

```text
PASS: gradient differential expressions match differentiated equations
```

The source file name retains its original v27 derivation because the v28 revision changed references rather than the algebra.

The check symbolically verifies the differentiation of the density/phase linearization, including the term proportional to `(g*n0)'`. It does **not** establish the nonlocality obstruction, a rigorous Sobolev-domain theorem, or a closed operator generator. Those require the separate smooth-test-function and functional-analytic arguments in the note.

## Extended calculations and acceptance test

The parent repository contains a **separate, extended numerical project** involving scattering, Evans calculations, GP time evolution and independent reference integrations. None of those numerical calculations proves the short note's locality theorem.

- Original extended-project test 7: `results/verification.json`
- Independent reference and grid convergence: `results/test7_independent.json`
- Original tolerance: `1e-7`; the `N=8000` result is **FAIL**.
- At `N=16000`, the saved comparison satisfies the same tolerance. This is grid convergence evidence, not a replacement for the original test.

## Archive / citation scope

- Extended GitHub release: [v1.0.0](https://github.com/msalih71/field-of-duality-bogoliubov-reproducibility/releases/tag/v1.0.0)
- DOI supplied for the **extended** code/data archive: [10.5281/zenodo.24165601](https://doi.org/10.5281/zenodo.24165601)

**Important:** The existing v1.0.0 tag predates these short-note files. Zenodo's v1.0.0 snapshot must not be described as containing this directory. These files are in the GitHub main branch until a subsequent tagged release is made and separately archived.

The DOI's deposited payload, permissions, and license need to be independently checked against the live Zenodo record; a GitHub link is not a substitute for that check.

## Manuscript availability statement (suggested)

"The symbolic differentiation check for the focused locality analysis is available in the `short_note_v28/` directory of the public GitHub repository. Supporting scattering and convergence data from the separate extended Bogoliubov study are also available there. The previously archived v1.0.0 code-and-data release is associated with DOI 10.5281/zenodo.24165601; that release predates the present short-note verification script. The numerical data are not presented as a proof of the locality theorem."

## Rights and authorship

Do not commit the manuscript, cover letters, referee correspondence, or private files. No reuse license is granted here in the absence of rights-holder approval. Verify coauthor approval and any license terms before a new archival release.
