"""In-repo SLSim adapters used in development and tests.

These provide deterministic, dependency-light catalog generators that match the
schema validated by :mod:`ilsne.slsim_glue`. They are handy for CI and for
users without SLSim installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cosmology_small import delta_mu


def dummy_slsim_adapter(
    *,
    n_sn: int,
    survey: dict[str, Any],
    cosmo: Any,
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,
    sigma_int: float,
    sigma_model: float,
    seed: int | None = None,
) -> pd.DataFrame:
    """Minimal adapter that avoids an SLSim dependency.

    Synthesizes ilSNe around identical deflectors, drawing ``R`` uniformly in an
    annulus and using an SIS-like magnification model. The fiducial model
    matches the truth so the pipeline self-consistency yields ``λ≈1``.

    Returns a DataFrame with the standard ilSNe schema.
    """
    rng = np.random.default_rng(seed or 0)
    theta_E = 1.0  # arcsec
    Rmin, Rmax = 0.5, 5.0
    z_lens = 0.4
    z_sn = rng.uniform(0.2, 1.0, size=n_sn)
    R = rng.uniform(Rmin, Rmax, size=n_sn)
    mu_true = 1.0 + theta_E / R
    mu_fid = mu_true.copy()
    micro = (survey or {}).get("micro", {})
    if micro.get("kind", "const") == "const":
        sig_micro = float(micro.get("sigma", 0.0))
        sigma_micro = np.full(n_sn, sig_micro)
    else:
        a = float(micro.get("a", 0.02))
        b = float(micro.get("b", 0.0))
        sigma_micro = a + b * (R)
    sigma_int_arr = np.full(n_sn, float(sigma_int))
    sigma_model_arr = np.full(n_sn, float(sigma_model))
    m_unlensed = float(mp_mean) + np.array(
        [delta_mu(float(z), float(z_pivot), 0.3, -1.0, 0.0) for z in z_sn]
    )
    m_lensed_true = m_unlensed - 2.5 * np.log10(mu_true)
    m_obs = m_lensed_true + np.random.default_rng(123).normal(
        0.0,
        np.sqrt(sigma_int_arr**2 + sigma_model_arr**2 + sigma_micro**2),
        size=n_sn,
    )
    df = pd.DataFrame(
        {
            "lens_id": np.arange(n_sn, dtype=int),
            "sn_id": np.arange(n_sn, dtype=int),
            "image_id": np.zeros(n_sn, dtype=int),
            "kind": ["il"] * n_sn,
            "z_sn": z_sn,
            "z_lens": np.full(n_sn, z_lens),
            "R_arcsec": R,
            "mu_true": mu_true,
            "mu_fid": mu_fid,
            # Convenience columns used by demos (not required by validator)
            "mu_model": mu_fid,
            # Macro (SIS) fields; external LOS fields are zeros in this toy
            "kappa_macro": theta_E / (2.0 * R),
            "gamma_macro": theta_E / (2.0 * R),
            "kappa_ext": np.zeros(n_sn),
            "gamma1_ext": np.zeros(n_sn),
            "gamma2_ext": np.zeros(n_sn),
            "kappa_star": np.nan,
            # keep historical column name for compatibility
            "m_unlensed_pivot": m_unlensed,
            "m_lensed_true": m_lensed_true,
            "m_obs": m_obs,
            "sigma_int_mag": sigma_int_arr,
            "sigma_micro_mag": sigma_micro,
            "sigma_model_mag": sigma_model_arr,
            "passed_detection": np.ones(n_sn, dtype=bool),
            "p_det": np.ones(n_sn),
            "n_epochs": np.full(n_sn, int((survey or {}).get("epochs", 6))),
            "filters_hit": ["ri"] * n_sn,
            "t_first": np.zeros(n_sn),
            "t_last": np.zeros(n_sn),
            "mult": np.ones(n_sn, dtype=int),
            "is_il": np.ones(n_sn, dtype=bool),
            "is_sl": np.zeros(n_sn, dtype=bool),
        }
    )
    return df


def bad_adapter_missing_cols(**kwargs: Any) -> pd.DataFrame:
    """Adapter intentionally returning an invalid schema (used in tests)."""

    return pd.DataFrame({"lens_id": [0]})
