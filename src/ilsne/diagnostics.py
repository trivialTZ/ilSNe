"""Diagnostics for sanity-checking λ inferences.

Provides quick consistency indicators, such as the λ value needed to absorb a
mismatch between the pivot prior mean and the data-driven intercept at λ=1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .magnification import LensDataset
from .pivot import FlatLCDM, distmod


def diagnose_lambda_bias(
    lens: LensDataset,
    cosmo: FlatLCDM | object,
    z_pivot: float,
    mp_mean: float,
    sigma_int: float = 0.0,
    sigma_micro_fn: Callable[[float | None], float] | None = None,
) -> dict[str, float | str]:
    """Back-of-the-envelope estimate of a λ bias from pivot mismatch.

    Parameters
    ----------
    lens : LensDataset
        Per-lens supernova measurements and model magnifications.
    cosmo : FlatLCDM or astropy.cosmology.Cosmology
        Cosmology used to compute distance-modulus differences.
    z_pivot : float
        Pivot redshift anchoring the magnitude zero-point.
    mp_mean : float
        Mean of the Gaussian prior on ``m_p`` (mag).
    sigma_int : float, optional
        Intrinsic per-SN magnitude scatter to add in quadrature.
    sigma_micro_fn : callable, optional
        Microlensing scatter model ``σ_micro(R)`` (mag) to add in quadrature.

    Returns
    -------
    dict
        Dictionary with keys
        - ``lens``: lens name
        - ``mp_prior``: prior mean on ``m_p`` (mag)
        - ``mp_hat_no_prior``: weighted intercept at λ=1 without the prior (mag)
        - ``delta_mp``: ``mp_prior - mp_hat_no_prior`` (mag)
        - ``lam_to_absorb_delta``: λ that would absorb ``delta_mp`` via 5 log10 λ
        - ``mean_residual`` and ``wmean_residual``: residual summaries at λ=1.
    """

    z = np.array([sn.z for sn in lens.sne], float)
    mu = np.array([sn.mu_model for sn in lens.sne], float)
    m = np.array([sn.m_obs for sn in lens.sne], float)
    s0 = np.array([sn.sigma_m for sn in lens.sne], float)

    if sigma_micro_fn is not None:
        s_micro = np.array(
            [float(sigma_micro_fn(getattr(sn, "R_arcsec", None))) for sn in lens.sne], float
        )
    else:
        s_micro = np.zeros_like(z)
    s = np.sqrt(s0**2 + sigma_int**2 + s_micro**2)

    # Support either local FlatLCDM or any astropy.cosmology cosmology
    dmu = np.asarray(distmod(cosmo, z)) - float(distmod(cosmo, float(z_pivot)))
    model_wo_mp = dmu - 2.5 * np.log10(mu)  # λ=1
    r = m - model_wo_mp  # this is d' in the derivation

    w = 1.0 / np.maximum(s, 1e-12) ** 2
    mp_hat = float(np.sum(w * r) / np.sum(w))

    delta = float(mp_mean - mp_hat)
    lam_comp = 10.0 ** (-(delta / 5.0))

    return dict(
        lens=lens.name,
        mp_prior=float(mp_mean),
        mp_hat_no_prior=mp_hat,
        delta_mp=delta,
        lam_to_absorb_delta=float(lam_comp),
        mean_residual=float(np.mean(r)),
        wmean_residual=float(np.sum(w * r) / np.sum(w)),
    )
