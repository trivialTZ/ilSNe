# ilSNe — Breaking the Mass‑Sheet Degeneracy with Standardized SN Ia Magnification

Intermediately lensed SNe Ia (ilSNe; ≲10″ from a deflector) provide absolute magnification information that breaks the mass‑sheet degeneracy (MSD). This repository implements a lightweight, transparent pipeline to infer the MST scale λ from standardized SN fluxes, propagate microlensing and selection assumptions, and combine ilSNe with strongly lensed SN (slSN) time‑delay distances to calibrate H0.

This README focuses on the science and equations; the demo notebook ties concepts to code: `ilsne_mst_demo.ipynb`.

## Science Goal (one‑liner)

Use standardized magnification of SNe Ia in the intermediate lensing regime (≲10″) to break the MSD and, when combined with slSN time‑delay distances, enable an ≈1% H0 measurement; do this with a hierarchical Bayesian pipeline that models microlensing and selection biases and is fed by SLSim‑backed yields.

## Why ilSNe

- MSD is the blocker: strong‑lensing imaging alone cannot fix the Fermat potential because of MSD; it rescales H0 and absolute magnifications together.
- Standardized magnification breaks MSD: SNe Ia give an absolute flux anchor; under MSD the absolute magnification rescales (μ_lens → μ_lens/λ²), so brightness vs. redshift constrains λ.
- ilSNe live where it helps: the intermediate (blue) regime extends to ~10″, covers more area and yields more events than the strongly lensed regime; microlensing is milder at larger radii.
- Together with slSNe time delays: ilSNe’s absolute magnification constraints + slSNe’s DΔt give a path to ~1% H0.

## The Mass‑Sheet Transform (MST)

For a lens model subject to an MST with scale λ,

$$
\kappa'(\theta) = \lambda\,\kappa(\theta) + (1-\lambda),\qquad \beta' = \lambda\,\beta.
$$

Key observables transform as

$$
\mu' = \mu/\lambda^2,\qquad \Delta t' = \lambda\,\Delta t,\qquad \Delta\phi' = \lambda\,\Delta\phi.
$$

Thus, without an absolute flux anchor, the product H0×(1/λ) remains degenerate with lensing. Standard candles break this by directly constraining λ through absolute magnification.

## SN Ia Standardization with a Pivot

We parameterize the unlensed apparent magnitude with a pivot magnitude m_p at redshift z_pivot. Let DM(z) be the distance modulus. The SN model is

$$
m_\mathrm{th}(z) = m_p + \big[\mathrm{DM}(z) - \mathrm{DM}(z_\mathrm{pivot})\big].
$$

For a lensed SN i with model magnification μ_{\rm lens,i}, the predicted magnitude under an MST with scale λ is

$$
m_i^{\rm pred} = m_p + \Delta\mathrm{DM}(z_i) - 2.5\,\log_{10}\mu_{\rm lens,i} + 5\,\log_{10}\lambda.
$$

The last term implements μ' = μ/λ². Observed magnitudes are modeled as

$$
m_i^{\rm obs} = m_i^{\rm pred} + \epsilon_i,\qquad \epsilon_i \sim \mathcal N\!\left(0,\; \sigma_{i}^2\right),
$$

with per‑image variance

$$
\sigma_i^2 = \sigma_{m,i}^2 + \sigma_\mathrm{int}^2 + \sigma_\mathrm{micro}^2(R_i).
$$

## Likelihood and Analytic Marginalization over m_p

For a given lens with N ilSNe, collect the residual vector after removing everything except m_p,

$$
\mathbf r(\lambda) = \mathbf m^{\rm obs} - \Big[\Delta\mathrm{DM}(\mathbf z) - 2.5\log_{10}\boldsymbol{\mu}_{\rm lens} + 5\log_{10}\lambda\,\mathbf 1\Big].
$$

Assume a Gaussian prior on the pivot m_p ~ N(m_{p,\rm mean}, \sigma_{m_p}^2). With diagonal S = diag(\sigma_i^2), the log‑likelihood marginalized analytically over m_p is

$$
\begin{aligned}
 a &= \mathbf 1^\top S^{-1}\mathbf 1 + \sigma_{m_p}^{-2},\\
 b &= \mathbf 1^\top S^{-1}\mathbf r + m_{p,\rm mean}\,\sigma_{m_p}^{-2},\\
 c &= \mathbf r^\top S^{-1}\mathbf r + m_{p,\rm mean}^2\,\sigma_{m_p}^{-2},\\[4pt]
 \log\mathcal L_\mathrm{marg}(\lambda) &= -\tfrac12\Big(c - \tfrac{b^2}{a}\Big) - \tfrac12\log a.\
\end{aligned}
$$

In code, see `src/ilsne/magnification.py:63` (analytic marginalization inside `_loglike_marg_mp`) and `src/ilsne/pivot.py:258`/`src/ilsne/pivot.py:310` (pivot model and `profile_m_p_gaussian_prior`).

We evaluate this on a 1D grid in λ and normalize to obtain p(λ | data) with summary statistics {λ_MAP, ⟨λ⟩, σ(λ)}.

## Microlensing and Selection Effects

Microlensing scatter is added in quadrature via σ_\mathrm{micro}(R). The library supports:

- Constant scatter: σ_\mathrm{micro}(R) = const (`sigma_micro_constant`).
- Radial/log profile: σ_\mathrm{micro}(R) = α + β log10(R/R_eff) (`sigma_micro_parametric`).
- CSV profiles interpolated from simulations/measurements (`load_sigma_micro_from_csv`).

These controls enable sensitivity studies on how microlensing assumptions shift p(λ). See `src/ilsne/microlens.py` and `src/ilsne/sensitivity.py`.

## LOS Convergence κ_ext

External convergence induces an effective MST: λ_tot = (1 − κ_ext) λ_int. We transform posteriors via LOS sampling:

$$
p(\lambda_\mathrm{int}) = \int p(\lambda_\mathrm{tot})\,p(\kappa_\mathrm{ext})\, \delta\!\left(\lambda_\mathrm{tot} - (1-\kappa_\mathrm{ext})\lambda_\mathrm{int}\right)\,\mathrm d\kappa_\mathrm{ext}.
$$

Implementation draws κ_ext from SLSim when available or a Gaussian fallback and applies the Jacobian (1−κ_ext). See `src/ilsne/los_posterior.py` and `src/ilsne/los.py`.

## From ilSNe to H0 with slSNe

Time delays satisfy

$$
\Delta t = \frac{D_{\Delta t}}{c}\,(1+z_l)\,\Delta\phi.
$$

Under MST, Δφ → λ Δφ and Δt → λ Δt, leaving the imaging‑only inference degenerate in H0/λ. ilSNe determine λ independently via absolute magnification, so combining an slSN time‑delay distance D_{Δt} with the ilSN‑inferred λ removes the degeneracy, enabling percent‑level H0 when aggregated over a sample.

## What the Demo Notebook Shows

The notebook `ilsne_mst_demo.ipynb` walks through:

- Pivoted SN model and Gaussian prior on m_p (Pantheon+ or fallback).
- SLSim‑backed yield synthesis and assignment of microlensing scenarios.
- Per‑lens λ posteriors with analytic m_p marginalization.
- Sensitivity of fractional precision σ(λ)/⟨λ⟩ vs. number of ilSN per lens.
- Stacking independent lenses: product of posteriors for a shared λ and a simple hierarchical λ model (μ_λ, τ).
- Optional LOS κ_ext marginalization to map λ_tot → λ_int.

Typical demo outputs illustrate that with tens of ilSN per lens and moderate microlensing scatter, per‑lens λ constraints at the ~1% level are attainable; stacking or hierarchical combination improves precision further, setting up a joint analysis with slSNe Δt.

## How Code Maps to the Equations

- Pivot model and cosmology: `src/ilsne/pivot.py` (Flat ΛCDM; Astropy cosmologies via `distmod`).
- Likelihood and λ posterior: `src/ilsne/magnification.py` (`LensedSN`, `LensDataset`, `infer_lambda_for_lens`).
- Microlensing models: `src/ilsne/microlens.py` (constant, parametric, CSV‑driven).
- Sensitivity scans: `src/ilsne/sensitivity.py`.
- Posterior stacking (shared λ): `src/ilsne/stacking.py`.
- LOS handling: `src/ilsne/los.py`, `src/ilsne/los_posterior.py`.
- Diagnostics (pivot mismatch → implied λ): `src/ilsne/diagnostics.py`.
- Yield synthesis front‑end and SLSim glue: `src/ilsne/slsim_yields.py`, `src/ilsne/slsim_glue.py`.

## Minimal Usage Example

```python
from ilsne import FlatLCDM, LensedSN, LensDataset, infer_lambda_for_lens
from ilsne import sigma_micro_constant

# 1) Build a per‑lens dataset (toy numbers)
lens = LensDataset(
    name="L1",
    sne=[
        LensedSN(z=0.8, mu_model=2.0, m_obs=23.15, sigma_m=0.08, R_arcsec=3.0),
        LensedSN(z=0.9, mu_model=1.5, m_obs=23.30, sigma_m=0.08, R_arcsec=4.5),
        # ... more ilSN behind the same deflector
    ],
)

# 2) Cosmology + pivot prior (Pantheon+‑like)
cosmo = FlatLCDM(H0=70.0, Omega_m=0.3)
z_pivot, mp_mean, mp_sigma = 0.1, 18.954, 0.004

# 3) Optional microlensing model
sigma_micro_fn = sigma_micro_constant(0.03)

# 4) Infer λ for this lens
post = infer_lambda_for_lens(
    lens, cosmo, z_pivot, mp_mean, mp_sigma,
    lam_grid=(0.6, 1.4, 1601), sigma_micro_fn=sigma_micro_fn, sigma_int=0.06,
)
print(post["lam_mean"], post["lam_std"])  # ≈ λ ± σ(λ)
```

See the notebook for a fuller workflow including stacking and LOS handling.

## Project Potential and Roadmap

- Hierarchical λ with realistic priors: move from the demo’s simple random‑effects model to a full hierarchical Bayesian treatment with selection functions and microlensing hyper‑priors.
- Joint ilSN + slSN analysis: combine λ posteriors with slSN Δt likelihoods (e.g., via a TDMag block) for end‑to‑end H0 inference.
- SLSim‑backed yields at scale: refine adapters to consume native SLSim LOS, cadence, and deflector populations; validate against survey‑specific simulations (e.g., Roman HLTDS).
- Bias audits: diagnostics for pivot prior shifts, microlensing mis‑specification, and LOS uncertainties; bake into pipeline validation.

## Package Layout (src/ilsne)

- `__init__.py` — public API exports
- `pivot.py` — pivot magnitude + minimal flat ΛCDM or Astropy cosmology
- `magnification.py` — λ likelihood with analytic m_p marginalization
- `microlens.py` — constant/parametric/CSV microlensing scatter models
- `diagnostics.py` — quick λ sanity checks from pivot mismatch
- `sensitivity.py` — λ sensitivity sweeps vs microlensing assumptions
- `stacking.py` — combine independent lens constraints on a shared λ
- `viz.py` — plotting for 1D λ posteriors
- `slsim_glue.py` — SLSim catalog interface with environment‑switchable adapter
- `slsim_yields.py` — `SurveyConfig` and yield synthesis wrapper
- `hierarc_glue.py` — Pantheon+ pivot via hierArc or local fallback
- `pantheon_local.py` — local Pantheon+ loader and pivot fit
- `roman_interface.py` — stubs to map Roman cadence to SurveyConfig
- `slsim_api_adapter.py` — in‑repo catalog adapter (fallback path)
- `pivot_cache.py` — tiny in‑memory cache for pivot results
- `cli.py` — command‑line entry point

## Cosmology Support

- Works with local `FlatLCDM` (lightweight) or any `astropy.cosmology.Cosmology` via `distmod`.
- Examples:
  - Astropy: `infer_lambda_for_lens(lens, Planck18, z_pivot=0.1, ...)`
  - Custom: `Flatw0waCDM(H0=70, Om0=0.3, w0=-0.9, wa=-0.5)`
  - Resolver: `from ilsne import resolve_cosmology; resolve_cosmology("Flatw0waCDM", H0=70, Om0=0.3, w0=-0.9, wa=-0.5)`

The CLI defaults to the local `FlatLCDM()`; programmatic APIs accept Astropy cosmologies directly.
