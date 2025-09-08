# ilSNe
intermediately lensed supernovae for breaking the MSD

## Package Layout (src/ilsne)
- `__init__.py` — public API exports
- `cosmology_small.py` — Δμ helpers with astropy or numerical fallback
- `pivot.py` — pivot magnitude utilities with a minimal flat ΛCDM and full
  support for any `astropy.cosmology` instance via `distmod`
- `magnification.py` — LensedSN/LensDataset and λ grid posterior
- `microlens.py` — constant/parametric/CSV microlensing scatter models
- `diagnostics.py` — quick λ sanity checks from pivot mismatch
- `sensitivity.py` — λ sensitivity sweeps vs microlensing assumptions
- `stacking.py` — combine independent lens constraints on a shared λ
- `viz.py` — plotting for 1D λ posteriors
- `io.py` — build LensDataset from pandas tables
- `slsim_glue.py` — unified SLSim catalog interface with env‑adapter hook
- `slsim_yields.py` — SurveyConfig and yield synthesis wrapper
- `hierarc_glue.py` — Pantheon+ pivot via hierArc or local fallback
- `pantheon_local.py` — local Pantheon+ loader and pivot fit
- `roman_interface.py` — stubs to map Roman cadence to SurveyConfig
- `slsim_api_adapter.py` — lightweight in‑repo catalog adapter
- `pivot_cache.py` — tiny in‑memory cache for pivot results
- `cli.py` — command‑line entry point

## Cosmology Support

Most functions that need distances accept either:
- the local `FlatLCDM` (kept for lightweight, dependency‑free use), or
- any `astropy.cosmology.Cosmology` instance (e.g., `Planck18`, `Flatw0waCDM`, `LambdaCDM`).

Examples:
- Inference with Astropy realization: `infer_lambda_for_lens(lens, Planck18, z_pivot=0.1, ...)`
- Custom Astropy class: `Flatw0waCDM(H0=70, Om0=0.3, w0=-0.9, wa=-0.5)`
- Convenience resolver: `from ilsne import resolve_cosmology; cosmo = resolve_cosmology("Flatw0waCDM", H0=70, Om0=0.3, w0=-0.9, wa=-0.5)`

The CLI continues to default to the local `FlatLCDM()`; programmatic APIs support passing an Astropy cosmology directly.
