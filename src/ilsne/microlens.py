"""Microlensing scatter helpers.

Utilities to represent and load simple prescriptions for the additional
photometric scatter induced by microlensing, either as a constant value,
as a parametric function of image-plane radius, or interpolated from a CSV
profile measured or simulated for the lens population of interest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def sigma_micro_constant(value_mag: float) -> Callable[[float | None], float]:
    r"""Return a constant microlensing scatter.

    Parameters
    ----------
    value_mag : float
        Magnitude scatter :math:`\sigma_\text{micro}` in mag.

    Returns
    -------
    callable
        Function of optional radius ``R_arcsec`` returning ``value_mag``.

    Examples
    --------
    >>> fn = sigma_micro_constant(0.1)
    >>> fn(1.0)
    0.1
    Use ``sigma_micro_included_in_sigma_m=True`` in the likelihood if these
    values are already folded into ``sigma_m`` to avoid double-counting.
    """

    def _fn(_R_arcsec: float | None) -> float:
        return float(value_mag)

    return _fn


def sigma_micro_parametric(
    alpha: float, beta: float, Reff_arcsec: float
) -> Callable[[float | None], float]:
    """Return a parametric microlensing scatter model.

    The scatter follows

    .. math::
       \sigma_\text{micro}(R) = \alpha + \beta \log_{10}(R/\mathrm{R_{eff}})

    and is clipped to be non-negative. If ``R`` is ``None`` the function
    returns ``0``.

    Parameters
    ----------
    alpha, beta : float
        Parameters of the log-scaling relation (mag).
    Reff_arcsec : float
        Effective radius :math:`R_\mathrm{eff}` in arcsec.

    Returns
    -------
    callable
        Function of optional radius ``R_arcsec`` returning the scatter in mag.
    """

    alpha = float(alpha)
    beta = float(beta)
    Reff_arcsec = float(Reff_arcsec)

    def _fn(R_arcsec: float | None) -> float:
        if R_arcsec is None:
            return 0.0
        R = float(R_arcsec)
        val = alpha + beta * np.log10(R / Reff_arcsec)
        return float(max(0.0, val))

    return _fn


def load_sigma_micro_from_csv(
    path: str,
    r_col_candidates: Sequence[str] = ("R_arcsec", "R", "radius_arcsec", "theta_arcsec"),
    sig_col_candidates: Sequence[str] = (
        "sigma_micro_mag",
        "sigma_micro",
        "sig_micro_mag",
        "sigma_mag",
    ),
) -> Callable[[float | None], float]:
    r"""Interpolate :math:`\sigma_\text{micro}(R)` from a CSV table.

    Parameters
    ----------
    path : str
        Path to a CSV file containing radius and scatter columns.
    r_col_candidates, sig_col_candidates : sequence of str, optional
        Candidate column names for radius (arcsec) and scatter (mag).

    Returns
    -------
    callable
        Function returning ``σ_micro`` in mag for a given ``R_arcsec``. If
        ``R_arcsec`` is ``None`` the largest-radius value is returned.

    Examples
    --------
    >>> fn = load_sigma_micro_from_csv("profile.csv")  # doctest: +SKIP
    >>> fn(0.5)  # doctest: +SKIP
    0.07
    When these values are already included in the photometric uncertainties,
    call the likelihood with ``sigma_micro_included_in_sigma_m=True`` to avoid
    double-counting.
    """

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def _pick(cands: Sequence[str]) -> str:
        for k in cands:
            if k.lower() in cols:
                return cols[k.lower()]
        raise KeyError(f"Required column not found. Tried: {cands} in {list(df.columns)}")

    r_col = _pick(r_col_candidates)
    sig_col = _pick(sig_col_candidates)
    xs = np.asarray(df[r_col], dtype=float)
    ys = np.asarray(df[sig_col], dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    def _fn(R_arcsec: float | None) -> float:
        if R_arcsec is None:
            return float(ys[-1])
        R = float(R_arcsec)
        if R <= xs[0]:
            return float(ys[0])
        if R >= xs[-1]:
            return float(ys[-1])
        i = np.searchsorted(xs, R)
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        t = (R - x0) / (x1 - x0)
        return float(y0 * (1 - t) + y1 * t)

    return _fn
