"""Magnification and mass-sheet transform inference utilities.

This module provides a light-weight likelihood for the mass-sheet transform
scale ``λ`` using standardizable supernova magnitudes anchored at a pivot
redshift. It supports optional analytic marginalization over the pivot
zero-point ``m_p`` and simple microlensing scatter models.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .pivot import FlatLCDM, distmod


@dataclass
class LensedSN:
    r"""Single lensed SN measurement.

    Parameters
    ----------
    z : float
        Source redshift.
    mu_model : float
        Model magnification :math:`\mu` (dimensionless).
    m_obs : float
        Observed apparent magnitude in mag.
    sigma_m : float
        :math:`1\sigma` magnitude uncertainty in mag.
    R_arcsec : float, optional
        Image-plane radius in arcsec used for microlensing scatter.
    """

    z: float
    mu_model: float
    m_obs: float
    sigma_m: float
    R_arcsec: float | None = None


@dataclass
class LensDataset:
    """Collection of lensed SNe behind a single foreground lens.

    Parameters
    ----------
    name : str
        Identifier for the lens (used in diagnostics and plots).
    sne : list[LensedSN]
        Measurements of lensed SNe associated with this lens.
    """

    name: str
    sne: list[LensedSN] = field(default_factory=list)


def _loglike_marg_mp(
    z: Sequence[float] | np.ndarray,
    mu_model: Sequence[float] | np.ndarray,
    m_obs: Sequence[float] | np.ndarray,
    sigma_m: Sequence[float] | np.ndarray,
    lam: float,
    cosmo: FlatLCDM | object,
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,
) -> float:
    r"""Log-likelihood with analytic :math:`m_p` marginalization.

    The predicted magnitude for each SN is

    .. math::
       m_i^{\mathrm{pred}} = m_p + \Delta\mu(z_i) - 2.5\log_{10}\mu_i + 5\log_{10}\lambda,

    where :math:`\Delta\mu(z_i) = \mu(z_i) - \mu(z_\text{pivot})`. The pivot
    magnitude :math:`m_p` has a Gaussian prior with mean ``mp_mean`` and width
    ``mp_sigma``.

    Parameters
    ----------
    z, mu_model, m_obs, sigma_m : array_like
        Per-SN redshift, model magnification, observed magnitude and uncertainty.
    lam : float
        Trial value of the mass-sheet scaling :math:`\lambda`.
    cosmo : FlatLCDM or astropy.cosmology.Cosmology
        Cosmology model used for distance-modulus differences.
    z_pivot : float
        Pivot redshift anchoring :math:`m_p`.
    mp_mean, mp_sigma : float
        Mean and standard deviation of the Gaussian prior on :math:`m_p` (mag).

    Returns
    -------
    float
        Log-likelihood value marginalized over :math:`m_p`.
    """
    z = np.asarray(z, float)
    mu_model = np.asarray(mu_model, float)
    m_obs = np.asarray(m_obs, float)
    sigma_m = np.asarray(sigma_m, float)
    if lam <= 0:
        return -np.inf
    if not (
        np.all(np.isfinite(z))
        and np.all(np.isfinite(mu_model))
        and np.all(np.isfinite(m_obs))
        and np.all(np.isfinite(sigma_m))
    ):
        raise ValueError("Non-finite inputs to _loglike_marg_mp")
    # Support either local FlatLCDM or any astropy.cosmology cosmology
    dmu = np.asarray(distmod(cosmo, z)) - float(distmod(cosmo, float(z_pivot)))
    model = dmu - 2.5 * np.log10(mu_model) + 5.0 * np.log10(lam)
    Cinv = np.diag(1.0 / np.maximum(sigma_m, 1e-12) ** 2)
    one = np.ones_like(z)
    # Classic Gaussian integral over m_p:
    # a = 1^T C^{-1} 1 + 1/s_p**2
    # b = 1^T C^{-1} r + m_p,prior / s_p**2
    # c = r^T C^{-1} r + (m_p,prior)**2 / s_p**2
    # logL = -0.5*(c - b**2/a) - 0.5*log(a)   (constants dropped)
    r = m_obs - model
    a = float(one @ (Cinv @ one)) + 1.0 / (mp_sigma**2)
    b = float(one @ (Cinv @ r)) + mp_mean / (mp_sigma**2)
    c = float(r @ (Cinv @ r)) + (mp_mean**2) / (mp_sigma**2)
    return float(-0.5 * (c - (b**2) / a) - 0.5 * np.log(a))


def _posterior_1d(
    grid: np.ndarray, logL: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """Normalize a 1D posterior from log-likelihood values.

    Parameters
    ----------
    grid : ndarray
        Monotonic grid of trial ``λ`` values.
    logL : ndarray
        Log-likelihood evaluated at ``grid``.

    Returns
    -------
    tuple
        Normalized PDF on ``grid`` and summary statistics ``(λ_MAP, λ_mean, λ_std)``.
    """
    logL = np.asarray(logL, float)
    grid = np.asarray(grid, float)
    logL -= np.max(logL)
    pdf = np.exp(logL)
    pdf /= np.trapezoid(pdf, grid)
    idx = int(np.argmax(pdf))
    if idx <= 1 or idx >= len(grid) - 2:
        warnings.warn(
            "MAP at λ grid edge; expand grid or check model sign",
            RuntimeWarning,
            stacklevel=2,
        )
    lam_map = float(grid[idx])
    lam_mean = float(np.trapezoid(grid * pdf, grid))
    lam_var = float(np.trapezoid((grid - lam_mean) ** 2 * pdf, grid))
    return pdf, lam_map, lam_mean, float(np.sqrt(max(lam_var, 0.0)))


def infer_lambda_for_lens(
    lens: LensDataset,
    cosmo: FlatLCDM | object,
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,
    mp_block: Any | None = None,
    lam_grid: tuple[float, float, int] = (0.5, 1.5, 2001),
    sigma_micro_fn: Callable[[float | None], float] | None = None,
    sigma_int: float = 0.0,
    sigma_micro_included_in_sigma_m: bool = False,
) -> dict[str, float | np.ndarray]:
    r"""Grid posterior for :math:`\lambda` per lens.

    Parameters
    ----------
    lens : LensDataset
        Dataset of lensed SNe.
    cosmo : FlatLCDM or astropy.cosmology.Cosmology
        Cosmology model.
    z_pivot : float
        Pivot redshift for the SN magnitude anchor.
    mp_mean, mp_sigma : float
        Mean and width of the Gaussian prior on :math:`m_p` (mag). Ignored if
        ``mp_block`` is provided.
    mp_block : optional
        Optional :class:`PantheonPivotBlock` providing ``(mp_mean, mp_sigma)``
        via :meth:`fit_pivot_mean_std`. When supplied, it overrides the explicit
        ``mp_mean``/``mp_sigma`` arguments.
    lam_grid : tuple, optional
        ``(min, max, npts)`` grid over :math:`\lambda`.
    sigma_micro_fn : callable, optional
        Function returning microlensing scatter ``σ_\mathrm{micro}(R)`` in mag given
        an image radius in arcsec. If provided, it is added in quadrature to
        the effective per-SN uncertainty.
    sigma_int : float, optional
        Intrinsic magnitude scatter :math:`\sigma_\mathrm{int}` in mag added in
        quadrature to each SN.
    sigma_micro_included_in_sigma_m : bool, optional
        Set to ``True`` if the per-SN ``sigma_m`` already includes microlensing
        scatter, in which case ``sigma_micro_fn`` is ignored. Default ``False``.

    Returns
    -------
    dict
        Dictionary with the ``lam_grid``, ``loglike``, normalized ``posterior``
        and summary statistics ``lam_map``, ``lam_mean`` and ``lam_std``.
    """
    z = [sn.z for sn in lens.sne]
    mu = [sn.mu_model for sn in lens.sne]
    m = [sn.m_obs for sn in lens.sne]
    s = []
    if mp_block is not None:
        mp_mean, mp_sigma = mp_block.fit_pivot_mean_std()
        if not (np.isfinite(mp_mean) and np.isfinite(mp_sigma) and mp_sigma > 0):
            raise ValueError("PantheonPivotBlock returned invalid (mp_mean, mp_sigma)")
    for sn in lens.sne:
        sig_micro = 0.0
        if sigma_micro_fn is not None and not sigma_micro_included_in_sigma_m:
            sig_micro = float(sigma_micro_fn(getattr(sn, "R_arcsec", None)))
        s.append(math.sqrt(sn.sigma_m**2 + sigma_int**2 + sig_micro**2))

    lam_vals = np.linspace(*lam_grid)
    logL = np.array(
        [_loglike_marg_mp(z, mu, m, s, lam, cosmo, z_pivot, mp_mean, mp_sigma) for lam in lam_vals]
    )
    post, lam_map, lam_mean, lam_std = _posterior_1d(lam_vals, logL)
    return {
        "lam_grid": lam_vals,
        "loglike": logL,
        "posterior": post,
        "lam_map": float(lam_map),
        "lam_mean": float(lam_mean),
        "lam_std": float(lam_std),
    }
