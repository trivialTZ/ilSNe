"""
Prototype adapter that returns an ilSNe-like catalog matching the standardized
ilsne schema used downstream.

Compatibility note
------------------
Recent SLSim versions changed the ``LensPop`` constructor signature from a
string-based API (e.g., ``LensPop(deflector_type=..., source_type=...)``) to
object-based inputs (``LensPop(deflector_population=..., source_population=...)``).
Rather than pinning a specific SLSim version or building full population objects
here, this adapter now uses a lightweight in-repo synthesis (same schema as used
in tests). If SLSim provides native catalog builders (e.g.,
``slsim.generate_ilsne_catalog``), ``ilsne.slsim_glue.slsim_catalog`` will find
and use them before falling back to this adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cosmology_small import delta_mu


def from_slsim(
    *,
    n_sn: int,
    survey: dict[str, Any],
    cosmo: Any,  # kept for API compatibility; not used by the lightweight synth
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,  # unused knob in the lightweight synth; retained for signature stability
    sigma_int: float,
    sigma_model: float,
    seed: int | None = None,
) -> pd.DataFrame:
    """Lightweight, version-agnostic ilSN catalog synthesis.

    This does not depend on SLSim internals. It generates a simple ilSN-only
    catalog with SIS-like magnification scaling, matching the schema validated by
    ``ilsne.slsim_glue._validate_catalog``. It mirrors the behavior of
    ``ilsne.test_adapters.dummy_slsim_adapter`` used in tests, with the same
    microlensing hooks via ``survey["micro"]``.
    """
    rng = np.random.default_rng(seed or 0)
    n = int(n_sn)

    # Simple identical-deflector toy model
    theta_E = 1.0  # arcsec
    z_lens = 0.4
    Rmin, Rmax = 0.5, 5.0  # arcsec

    # Draw source redshifts and radii
    z_sn = rng.uniform(0.2, 1.0, size=n)
    R = rng.uniform(Rmin, Rmax, size=n)

    # SIS-like magnification; fiducial equals true for demo (lambda=1)
    mu_true = 1.0 + theta_E / np.maximum(R, 1e-3)
    mu_fid = mu_true.copy()
    if not np.all(mu_true > 0):
        raise AssertionError("mu_true must be > 0")

    # Microlensing scatter control
    micro = (survey or {}).get("micro", {})
    if micro.get("kind", "const") == "const":
        sig_micro = float(micro.get("sigma", 0.0))
        sigma_micro = np.full(n, sig_micro)
    else:
        a = float(micro.get("a", 0.02))
        b = float(micro.get("b", 0.0))
        sigma_micro = a + b * R

    # Photometric scatter terms
    sigma_int_arr = np.full(n, float(sigma_int))
    sigma_model_arr = np.full(n, float(sigma_model))

    # True pivot construction
    omegam = float(getattr(cosmo, "Omega_m", getattr(cosmo, "omegam", 0.3)))
    w0 = float(getattr(cosmo, "w0", -1.0))
    wa = float(getattr(cosmo, "wa", 0.0))
    dmu = np.array([delta_mu(float(z), float(z_pivot), omegam, w0, wa) for z in z_sn])
    m_unlensed = float(mp_mean) + dmu
    m_lensed_true = m_unlensed - 2.5 * np.log10(mu_true)
    sigma_tot = np.sqrt(sigma_int_arr**2 + sigma_model_arr**2 + sigma_micro**2)
    m_obs = m_lensed_true + rng.normal(0.0, sigma_tot, size=n)
    assert np.all(np.isfinite(m_obs))

    df = pd.DataFrame(
        {
            "lens_id": np.arange(n, dtype=int),
            "sn_id": np.arange(n, dtype=int),
            "image_id": np.zeros(n, dtype=int),
            "kind": ["il"] * n,
            "z_sn": z_sn,
            "z_lens": np.full(n, z_lens),
            "R_arcsec": R,
            "mu_true": mu_true,
            "mu_fid": mu_fid,
            # Convenience columns used by downstream demos
            "mu_model": mu_fid,
            # Macro (SIS) fields for clarity; external LOS fields are zero here
            "kappa_macro": theta_E / (2.0 * R),
            "gamma_macro": theta_E / (2.0 * R),
            "kappa_ext": np.zeros(n),
            "gamma1_ext": np.zeros(n),
            "gamma2_ext": np.zeros(n),
            "kappa_star": np.nan,
            # Historical column name kept for compatibility
            "m_unlensed_pivot": m_unlensed,
            "m_lensed_true": m_lensed_true,
            "m_obs": m_obs,
            "sigma_int_mag": sigma_int_arr,
            "sigma_micro_mag": sigma_micro,
            "sigma_model_mag": sigma_model_arr,
            "passed_detection": np.ones(n, dtype=bool),
            "p_det": np.ones(n),
            "n_epochs": np.full(n, int((survey or {}).get("epochs", 6))),
            "filters_hit": ["ri"] * n,
            "t_first": np.zeros(n),
            "t_last": np.zeros(n),
            "mult": np.ones(n, dtype=int),
            "is_il": np.ones(n, dtype=bool),
            "is_sl": np.zeros(n, dtype=bool),
        }
    )
    return df
