from __future__ import annotations

import numpy as np

from .los import sample_kappa_ext


def posterior_lambda_intrinsic_from_total(
    lam_grid: np.ndarray,
    p_lambda_tot: np.ndarray,
    z_l: float,
    z_s: float,
    nsamp: int = 4000,
    sigma_kext: float = 0.025,
    nonlinear_h5: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a ``λ_tot`` posterior into a ``λ_int`` posterior via LOS sampling.

    Parameters
    ----------
    lam_grid : ndarray
        Monotonic grid of ``λ`` values.
    p_lambda_tot : ndarray
        Posterior PDF defined on ``lam_grid`` for the total (observed) ``λ``.
    z_l, z_s : float
        Lens and source redshifts for the LOS draw.
    nsamp : int, optional
        Number of LOS samples to draw.
    sigma_kext : float, optional
        Fallback Gaussian width for ``κ_ext`` when LOS tables are not available.
    nonlinear_h5 : str, optional
        Path to SLSim nonlinear LOS correction HDF5 (optional).

    Returns
    -------
    tuple[ndarray, ndarray]
        The input grid and the normalized posterior ``p(λ_int | λ_tot, κ_ext)``.
    """
    lam = np.asarray(lam_grid, dtype=float)
    p0 = np.asarray(p_lambda_tot, dtype=float)
    if lam.shape != p0.shape:
        raise ValueError("lam_grid and p_lambda_tot must have the same shape")

    Z0 = float(np.trapezoid(p0, lam))
    p0 = p0 / (Z0 + 1e-300)

    kappas = sample_kappa_ext(
        int(nsamp), z_l=z_l, z_s=z_s, sigma_fallback=float(sigma_kext), nonlinear_h5=nonlinear_h5
    )

    hist = np.zeros_like(lam, dtype=float)
    for k in kappas:
        # λ_tot = (1 - κ_ext) λ_int  =>  λ_int = λ_tot / (1 - κ_ext)
        lam_tot = (1.0 - float(k)) * lam
        # Jacobian |dλ_tot/dλ_int| = (1 - κ_ext)
        hist += np.interp(lam_tot, lam, p0, left=0.0, right=0.0) * max(0.0, 1.0 - float(k))

    Z = float(np.trapezoid(hist, lam))
    if not np.isfinite(Z) or Z <= 0.0:
        return lam, np.zeros_like(lam)
    return lam, hist / Z


__all__ = ["posterior_lambda_intrinsic_from_total"]

