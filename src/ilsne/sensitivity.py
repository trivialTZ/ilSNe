"""Sensitivity studies for microlensing assumptions.

Convenience wrappers to sweep either the parameters of a parametric
``σ_micro(R)`` relation or a global scaling applied to a baseline scatter,
and measure the impact on the inferred mass-sheet parameter ``λ``.
"""

from __future__ import annotations

import itertools

import pandas as pd

from .magnification import LensDataset, infer_lambda_for_lens
from .microlens import sigma_micro_constant, sigma_micro_parametric
from .pivot import FlatLCDM


def sweep_sigma_micro_params(
    lens: LensDataset,
    cosmo: FlatLCDM,
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,
    alphas: list[float],
    betas: list[float],
    Reff: float,
    lam_grid: tuple[float, float, int] = (0.7, 1.4, 601),
    sigma_int: float = 0.0,
) -> pd.DataFrame:
    """Sweep parametric microlensing scatter parameters.

    Evaluates the posterior sensitivity of the mass-sheet transform scale
    ``λ`` over a grid of ``α`` and ``β`` values in the microlensing model

    .. math::
       σ_\mathrm{micro}(R) = α + β \log_{10}(R/R_\mathrm{eff}).

    Parameters
    ----------
    lens : LensDataset
        Lensed SN dataset.
    cosmo : FlatLCDM
        Cosmology model.
    z_pivot : float
        Pivot redshift anchoring ``m_p``.
    mp_mean, mp_sigma : float
        Gaussian prior on ``m_p`` (mag).
    alphas, betas : list[float]
        Grids of ``α`` and ``β`` coefficients in mag.
    Reff : float
        Effective radius :math:`R_\mathrm{eff}` in arcsec.
    lam_grid : tuple, optional
        Grid ``(min,max,npts)`` over ``λ``.
    sigma_int : float, optional
        Intrinsic scatter added in quadrature (mag).

    Returns
    -------
    pandas.DataFrame
        Table of sweep results with columns ``alpha``, ``beta``, ``lam_mean``,
        ``lam_std`` and ``lam_map``.
    """
    rows = []
    for a, b in itertools.product(alphas, betas):
        fn = sigma_micro_parametric(a, b, Reff)
        res = infer_lambda_for_lens(
            lens,
            cosmo,
            z_pivot,
            mp_mean,
            mp_sigma,
            lam_grid=lam_grid,
            sigma_micro_fn=fn,
            sigma_int=sigma_int,
        )
        rows.append(
            dict(
                alpha=a,
                beta=b,
                Reff=Reff,
                lam_mean=float(res["lam_mean"]),
                lam_std=float(res["lam_std"]),
                lam_map=float(res["lam_map"]),
            )
        )
    return pd.DataFrame(rows)


def sweep_sigma_micro_scale(
    lens: LensDataset,
    cosmo: FlatLCDM,
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,
    scales: list[float],
    base_sigma: float,
    lam_grid: tuple[float, float, int] = (0.7, 1.4, 601),
    sigma_int: float = 0.0,
) -> pd.DataFrame:
    """Sweep a global scaling of a base microlensing scatter.

    Parameters
    ----------
    lens : LensDataset
        Lensed SN dataset.
    cosmo : FlatLCDM
        Cosmology model.
    z_pivot : float
        Pivot redshift anchoring ``m_p``.
    mp_mean, mp_sigma : float
        Gaussian prior on ``m_p`` (mag).
    scales : list[float]
        Multiplicative factors applied to ``base_sigma``.
    base_sigma : float
        Baseline microlensing scatter (mag).
    lam_grid : tuple, optional
        Grid ``(min,max,npts)`` over ``λ``.
    sigma_int : float, optional
        Intrinsic scatter added in quadrature (mag).

    Returns
    -------
    pandas.DataFrame
        Table of sweep results with columns ``scale``, ``lam_mean``,
        ``lam_std`` and ``lam_map``.
    """
    rows = []
    for s in scales:
        fn = sigma_micro_constant(base_sigma * s)
        res = infer_lambda_for_lens(
            lens,
            cosmo,
            z_pivot,
            mp_mean,
            mp_sigma,
            lam_grid=lam_grid,
            sigma_micro_fn=fn,
            sigma_int=sigma_int,
        )
        rows.append(
            dict(
                scale=s,
                base_sigma=base_sigma,
                lam_mean=float(res["lam_mean"]),
                lam_std=float(res["lam_std"]),
                lam_map=float(res["lam_map"]),
            )
        )
    return pd.DataFrame(rows)
