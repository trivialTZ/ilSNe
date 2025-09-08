"""Lightweight cosmology helpers for distance modulus differences.

The module prefers :mod:`astropy` for accurate computations and falls back to
an internal numerical integrator when astropy is unavailable.  All distances
are expressed through the distance modulus

.. math::

   \mu(z) = 5 \log_{10}\!\left(\frac{D_L(z)}{10\,\text{pc}}\right),

so the difference

.. math::

   \Delta\mu(z; z_a) = \mu(z) - \mu(z_a)

is independent of the absolute distance scale.  Units are magnitudes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def delta_mu_astropy(
    z: float | Iterable[float],
    z_anchor: float,
    omegam: float = 0.3,
    w0: float = -1.0,
    wa: float = 0.0,
    H0: float = 70.0,
) -> float | np.ndarray:
    """Δμ using :mod:`astropy`'s :class:`~astropy.cosmology.Flatw0waCDM`.

    Parameters
    ----------
    z, z_anchor : float or array-like
        Redshift(s) and anchor redshift (dimensionless).
    omegam, w0, wa, H0 : float, optional
        Cosmological parameters. ``H0`` is in km/s/Mpc.

    Returns
    -------
    numpy.ndarray
        Difference in distance modulus :math:`\Delta\mu` [mag].
    """

    from astropy.cosmology import Flatw0waCDM

    cosmo = Flatw0waCDM(H0=H0, Om0=omegam, w0=w0, wa=wa)
    z_arr = np.asarray(z, dtype=float)
    mu = cosmo.distmod(z_arr).value
    mu_anchor = cosmo.distmod(float(z_anchor)).value
    res = mu - mu_anchor
    return float(res) if np.isscalar(z_arr) else res


# --- Numerical fallback ----------------------------------------------------


def _E(a: float, omegam: float, w0: float, wa: float) -> float:
    """Dimensionless expansion rate :math:`E(a)`."""

    de = a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(-3.0 * wa * (1.0 - a))
    return math.sqrt(omegam * a ** (-3.0) + (1.0 - omegam) * de)


def comoving_distance(
    z: float,
    omegam: float = 0.3,
    w0: float = -1.0,
    wa: float = 0.0,
    steps: int = 2048,
) -> float:
    """Line-of-sight comoving distance in units of ``c/H0`` [dimensionless]."""

    z = float(z)
    if z <= 0.0:
        return 0.0
    a_min = 1.0 / (1.0 + z)
    a_max = 1.0
    n = steps if steps % 2 == 0 else steps + 1
    h = (a_max - a_min) / n
    s = 0.0
    for i in range(n + 1):
        a = a_min + i * h
        coeff = 4.0 if i % 2 == 1 else 2.0
        if i == 0 or i == n:
            coeff = 1.0
        s += coeff * (1.0 / (a * a * _E(a, omegam, w0, wa)))
    return (h / 3.0) * s


def distance_modulus(
    z: float,
    omegam: float = 0.3,
    w0: float = -1.0,
    wa: float = 0.0,
) -> float:
    """Distance modulus :math:`\mu(z)` via internal integration."""

    chi = comoving_distance(z, omegam, w0, wa)
    dl_tilde = (1.0 + z) * chi
    return 5.0 * math.log10(max(dl_tilde, 1e-30))


def delta_mu_numerical(
    z: float | Iterable[float],
    z_anchor: float,
    omegam: float = 0.3,
    w0: float = -1.0,
    wa: float = 0.0,
) -> float | np.ndarray:
    """Δμ computed with the numerical integrator (astropy fallback)."""

    z_arr = np.asarray(z, dtype=float)
    mu = np.array([distance_modulus(float(zi), omegam, w0, wa) for zi in np.atleast_1d(z_arr)])
    mu_anchor = distance_modulus(float(z_anchor), omegam, w0, wa)
    res = mu - mu_anchor
    return float(res[0]) if np.isscalar(z_arr) else res


def delta_mu(
    z: float | Iterable[float],
    z_anchor: float,
    omegam: float = 0.3,
    w0: float = -1.0,
    wa: float = 0.0,
    H0: float = 70.0,
) -> float | np.ndarray:
    """Public Δμ accessor with astropy preference and numerical fallback."""

    try:
        return delta_mu_astropy(z, z_anchor, omegam=omegam, w0=w0, wa=wa, H0=H0)
    except Exception:  # pragma: no cover - exercised in environments without astropy
        return delta_mu_numerical(z, z_anchor, omegam=omegam, w0=w0, wa=wa)


__all__ = [
    "delta_mu",
    "delta_mu_astropy",
    "delta_mu_numerical",
    "comoving_distance",
    "distance_modulus",
]
