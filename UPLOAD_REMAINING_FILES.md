# Complete the repository from the prepared local archive

**Repository:** https://github.com/msalih71/field-of-duality-bogoliubov-reproducibility

This repository has been initialized with documentation and a small set of verification files. The full prepared archive (`field-of-duality-github-ready.zip`) includes the remaining code, numerical data, and figures.

## Recommended method: preserve remote history and upload all missing files

Download the prepared ZIP from the accompanying ChatGPT conversation and save it on your computer. Extract it to a folder called `fod-upload`. In a terminal with Git installed:

```bash
git clone https://github.com/msalih71/field-of-duality-bogoliubov-reproducibility.git
cd field-of-duality-bogoliubov-reproducibility
```

Then copy the **contents** of the extracted `fod-upload` folder into this cloned directory, preserving `src/`, `results/`, `tests/`, `derivations/` and `figures/`. The GitHub README can be kept or replaced with the archive's README. Do **not** copy the `fod-upload` parent directory as an extra nested level.

```bash
git status --short
git add README.md .gitignore requirements.txt SUPPLEMENT_DESCRIPTION.txt src/ tests/ derivations/ results/ figures/
git commit -m "Add complete Field of Duality reproducibility materials"
git push origin main
```

Use GitHub Desktop instead of these commands if preferred.

## Safety checks

- Only commit code, generated plots, numerical data and documentation.
- **Do not** upload manuscript source/PDF, cover letters, editor or reviewer correspondence, personal credentials, or internal submission packages.
- The original test-7 threshold is `1e-7`, and the `N=8000` result is **FAIL**. Keep it unchanged in `results/verification.json`.
- An internal independent solver cross-check is not a third-party replication.
- A license should only be added with rights-holder authorization.

## Verification

After all files are uploaded:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python derivations/symbolic_checks.py
python src/test7_independent.py
```

The last command can require substantial computation; heavy Evans and GP runs are not covered by the quick test suite.
