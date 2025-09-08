from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import numpy as np

try:  # Optional dependency; sampling falls back to Gaussian if unavailable
    from slsim.LOS.los_pop import LOSPop  # type: ignore

    _SLSIM_OK = True
except Exception:  # pragma: no cover - slsim not installed or missing LOS
    _SLSIM_OK = False


def sample_kappa_ext(
    n: int,
    z_l: Union[float, Sequence[float]],
    z_s: Union[float, Sequence[float]],
    *,
    sigma_fallback: float = 0.025,
    nonlinear_h5: Optional[str] = None,
) -> np.ndarray:
    """Sample external convergence ``κ_ext`` for lens-source pairs.

    Uses SLSim's LOSPop when available; otherwise draws from a Gaussian prior
    with zero mean and width ``sigma_fallback``.

    Parameters
    ----------
    n : int
        Number of samples to return.
    z_l, z_s : float or sequence
        Lens and source redshifts. Scalars are broadcast to length ``n``.
    sigma_fallback : float, optional
        Standard deviation for the Gaussian fallback when SLSim LOS is not
        available.
    nonlinear_h5 : str, optional
        Path to SLSim nonlinear LOS correction HDF5. If ``None``, uses the
        ``ILSNE_SLSIM_NONLINEAR_H5`` environment variable when set.
    """
    from .utils import as_array  # tiny helper to broadcast scalars

    zl = as_array(z_l, n)
    zs = as_array(z_s, n)

    if nonlinear_h5 is None:
        nonlinear_h5 = os.environ.get("ILSNE_SLSIM_NONLINEAR_H5")

    if _SLSIM_OK:
        los = LOSPop(
            los_bool=True,
            nonlinear_los_bool=bool(nonlinear_h5),
            nonlinear_correction_path=nonlinear_h5,
            no_correction_path=None,
        )
        out = []
        for zli, zsi in zip(zl, zs):
            try:
                k = float(
                    los.draw_los(
                        source_redshift=float(zsi), deflector_redshift=float(zli)
                    ).convergence()
                )
            except Exception:
                # Robustness: if LOS draw fails, fall back to Gaussian
                k = float(np.random.normal(0.0, float(sigma_fallback)))
            out.append(k)
        return np.asarray(out, dtype=float)

    # Fallback when SLSim LOS is not available
    return np.random.normal(0.0, float(sigma_fallback), size=int(n)).astype(float)


__all__ = ["sample_kappa_ext"]

