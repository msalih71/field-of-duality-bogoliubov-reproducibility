# Complete the Field of Duality reproducibility upload

## Current state

This GitHub repository was initialized through the connected GitHub integration, but the full 56-file local archive is **not yet** uploaded. No claim of complete online reproducibility should be made until all files appear.

## Verified full upload kit

The accompanying ChatGPT conversation provides `field-of-duality-upload-kit.zip`, which contains:

- `field-of-duality-github-ready.zip` — exact 56-file code/data-only archive
- `upload_to_github.ps1` — Git for Windows upload script
- `upload-manifest.json` — SHA-256 checksums for all 56 files

Full source archive SHA-256: `5c9546e4fada7a2ba3f084b93ff19d8a3843cbbe630c769f3a3c767d1ce0e0f0`.

On Windows, extract the upload kit, open PowerShell in its folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\upload_to_github.ps1
```

The script uses the Git Credential Manager or existing git login; no password or access token needs to be pasted into ChatGPT. It clones this repository, copies only approved code/data files, checks the 56-file archive for unexpected filenames, commits, and pushes the result. The manuscript, editorial correspondence and referee materials are excluded.

## After uploading

Check that GitHub includes `src/fod_scattering.py`, `src/test7_independent.py`, `results/test7_independent.json`, `results/verification.json`, and the `figures/` folder.

The original test-7 tolerance is `1e-7`, and the `N=8000` result remains **FAIL**.

Do not cite this repository as the complete data/code archive until the expected assets have been verified online.
