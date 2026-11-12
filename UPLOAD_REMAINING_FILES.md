# Upload completed

The reproducibility code/data upload was verified on 12 November 2026.

The GitHub repository now includes the original research code, numerical JSON/NPZ result files, figures, symbolic checks, test suite and documentation, organised under:

- `src/`
- `results/`
- `figures/`
- `derivations/`
- `tests/`
- `generated_tables/`

The original acceptance test 7 at `N=8000` remains **FAIL** against the fixed absolute threshold `1e-7`. Higher-resolution and independent-reference checks are available in `results/test7_independent.json`. These do not retroactively change the original result.

The manuscript PDF/source and editorial correspondence are not included. This repository has no permanent archival DOI yet.

Run the lightweight checks in a clean Python environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python derivations/symbolic_checks.py
```

To run the independent test-7 reference, use `python src/test7_independent.py`. Heavy computations take longer.

Previous local upload instructions are retained only for provenance; no additional browser upload is needed.
