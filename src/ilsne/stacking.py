"""Stacking utilities for combining lens posteriors.

This module offers two notions of combination:

- Multiplying independent constraints on a shared mass-sheet parameter ``λ``
  (Gaussian precision product or grid product in log-space).
- Convolving independent per-lens posteriors when modeling a sum of ``λ_i``
  (rarely used for MST but useful for pedagogy and tests).
"""

from __future__ import annotations

import numpy as np


def product_stack_common_lambda(posts: list[dict], ngrid: int = 2001) -> dict:
    """Multiply independent λ posteriors in log-space.

    Parameters
    ----------
    posts : list of dict
        Each dict must contain ``lam_grid`` and normalized ``posterior`` arrays.
        Grids across lenses must overlap; the common overlapping range is used.
    ngrid : int, optional
        Number of points in the final common grid.

    Returns
    -------
    dict
        Dictionary with keys ``lam_grid``, ``posterior``, ``lam_mean``,
        ``lam_std`` and ``lam_map`` for the shared ``λ``.
    """

    if not posts:
        raise ValueError("No posteriors provided")
    lo = max(float(np.asarray(p["lam_grid"])[0]) for p in posts)
    hi = min(float(np.asarray(p["lam_grid"])[-1]) for p in posts)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError("Incompatible λ grids among lenses")
    final_grid = np.linspace(lo, hi, ngrid)
    log_pdf = np.zeros_like(final_grid)
    for p in posts:
        g = np.asarray(p["lam_grid"], float)
        f = np.asarray(p["posterior"], float)
        f /= max(float(np.trapezoid(f, g)), 1e-300)
        yi = np.interp(final_grid, g, f, left=1e-300, right=1e-300)
        log_pdf += np.log(np.clip(yi, 1e-300, None))
    pdf_final = np.exp(log_pdf - log_pdf.max())
    pdf_final /= np.trapezoid(pdf_final, final_grid)
    lam_mean = float(np.trapezoid(final_grid * pdf_final, final_grid))
    lam_var = float(np.trapezoid((final_grid - lam_mean) ** 2 * pdf_final, final_grid))
    lam_map = float(final_grid[np.argmax(pdf_final)])
    out = {
        "lam_grid": final_grid,
        "posterior": pdf_final,
        "lam_mean": lam_mean,
        "lam_std": float(np.sqrt(lam_var)),
        "lam_map": lam_map,
    }
    out["lam_tot_mean"] = out["lam_mean"]
    out["lam_tot_std"] = out["lam_std"]
    out["lam_tot_map"] = out["lam_map"]
    return out


def gaussian_precision_stack(means: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    r"""
    Multiply independent Gaussian constraints on a shared
    :math:`\lambda \sim \mathcal{N}(\mu_i, \sigma_i^2)`.

    .. math::
        \frac{1}{\sigma_{\mathrm{post}}^2} = \sum_i \frac{1}{\sigma_i^2},\quad
        \mu_{\mathrm{post}} = \sigma_{\mathrm{post}}^2 \sum_i \frac{\mu_i}{\sigma_i^2}.

    Parameters
    ----------
    means : array_like
        Means :math:`\mu_i` (dimensionless).
    variances : array_like
        Variances :math:`\sigma_i^2` (dimensionless).

    Returns
    -------
    tuple of float
        Posterior mean and variance for :math:`\lambda` (dimensionless).
    """

    w = 1.0 / np.asarray(variances, float)
    mu = float(np.sum(means * w) / np.sum(w))
    sig2 = float(1.0 / np.sum(w))
    return mu, sig2


def gaussian_stack_lambda(posteriors: list[dict[str, float]]) -> dict[str, float]:
    r"""DEPRECATED: now performs shared-λ precision-product.

    Historically this function summed :math:`\lambda` and variances across
    lenses, which is incorrect for the mass-sheet transform. It now multiplies
    independent Gaussian constraints on a *common* :math:`\lambda` so that

    .. math::
       \frac{1}{\sigma_{\text{post}}^2} = \sum_i \frac{1}{\sigma_i^2},\quad
       \mu_{\text{post}} = \sigma_{\text{post}}^2 \sum_i \frac{\mu_i}{\sigma_i^2}.

    Parameters
    ----------
    posteriors : list of dict
        Each must contain ``lam_mean`` and ``lam_std`` (dimensionless).

    Returns
    -------
    dict
        Dictionary with ``lam_mean`` and ``lam_std`` (also legacy aliases
        ``lam_tot_mean``/``lam_tot_std``).
    """
    import warnings

    warnings.warn(
        "gaussian_stack_lambda is deprecated; use gaussian_precision_stack()",
        DeprecationWarning,
        stacklevel=2,
    )
    means = np.array([p["lam_mean"] for p in posteriors], float)
    stds = np.array([p["lam_std"] for p in posteriors], float)
    if means.size == 0:
        raise ValueError("No posteriors provided")
    mu, var = gaussian_precision_stack(means, stds**2)
    out = {"lam_mean": float(mu), "lam_std": float(np.sqrt(var))}
    out["lam_tot_mean"] = out["lam_mean"]
    out["lam_tot_std"] = out["lam_std"]
    return out


def exact_convolution_stack(
    posteriors: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray | float]:
    r"""Numerically convolve independent :math:`\lambda` posteriors.

    For independent lenses with posteriors :math:`p_i(\lambda)` the total
    distribution for the sum :math:`\lambda_\text{tot} = \sum_i \lambda_i`
    is their convolution

    .. math::
       p_\text{tot}(\lambda) = (p_1 * p_2 * \dots * p_n)(\lambda).

    Parameters
    ----------
    posteriors : list of dict
        Each dict must contain ``lam_grid`` and ``posterior`` arrays describing
        a normalized probability density for an individual lens.

    Returns
    -------
    dict
        Dictionary with the common ``lam_grid``, ``posterior`` on that grid and
        summary statistics ``lam_tot_mean``, ``lam_tot_std`` and ``lam_tot_map``.
    """

    mins = [float(np.asarray(p["lam_grid"])[0]) for p in posteriors]
    maxes = [float(np.asarray(p["lam_grid"])[-1]) for p in posteriors]
    final_grid = np.linspace(np.sum(mins), np.sum(maxes), 2001)
    dx = final_grid[1] - final_grid[0]

    pdf_tot: np.ndarray | None = None
    grid_tot: np.ndarray | None = None
    for p in posteriors:
        g = np.asarray(p["lam_grid"], dtype=float)
        pdf = np.asarray(p["posterior"], dtype=float)
        norm = float(np.trapezoid(pdf, g))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Input posterior must integrate to >0")
        pdf /= norm
        g_dense = np.arange(g[0], g[-1] + dx, dx)
        pdf_dense = np.interp(g_dense, g, pdf, left=0.0, right=0.0)
        if pdf_tot is None:
            pdf_tot = pdf_dense
            grid_tot = g_dense
        else:
            conv = np.convolve(pdf_tot, pdf_dense, mode="full") * dx
            grid_tot = np.linspace(grid_tot[0] + g_dense[0], grid_tot[-1] + g_dense[-1], conv.size)
            pdf_tot = conv

    assert pdf_tot is not None and grid_tot is not None
    pdf_final = np.interp(final_grid, grid_tot, pdf_tot, left=0.0, right=0.0)
    pdf_final = np.clip(pdf_final, 0.0, np.inf)
    pdf_final /= np.trapezoid(pdf_final, final_grid)

    mean = float(np.trapezoid(final_grid * pdf_final, final_grid))
    var = float(np.trapezoid((final_grid - mean) ** 2 * pdf_final, final_grid))
    std = float(np.sqrt(var))
    lam_map = float(final_grid[np.argmax(pdf_final)])
    return {
        "lam_grid": final_grid,
        "posterior": pdf_final,
        "lam_tot_mean": mean,
        "lam_tot_std": std,
        "lam_tot_map": lam_map,
    }
