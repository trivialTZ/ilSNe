"""Cosmology and SN magnitude pivot utilities.

This module provides a minimal flat :math:`\Lambda`CDM cosmology and helpers
for the supernova magnitude pivot parameterization

.. math::
   m_\mathrm{th}(z) = m_p + [\mu(z) - \mu(z_\text{pivot})],

where :math:`m_p` is the apparent magnitude at a chosen pivot redshift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FlatLCDM:
    r"""Flat :math:`\Lambda`CDM cosmology.

    Parameters
    ----------
    H0 : float, optional
        Hubble constant in km/s/Mpc. Default is ``70``.
    Omega_m : float, optional
        Matter density parameter. Default is ``0.3``.
    c : float, optional
        Speed of light in km/s. Default is ``299792.458``.

    Notes
    -----
    The dimensionless expansion rate is

    .. math::

       E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_\Lambda},

    where :math:`\Omega_\Lambda = 1-\Omega_m` in a spatially flat universe.
    """

    H0: float = 70.0
    Omega_m: float = 0.3
    c: float = 299792.458

    @property
    def Omega_L(self) -> float:
        r"""Dark-energy density parameter :math:`\Omega_\Lambda`."""
        return 1.0 - self.Omega_m

    def E(self, z: float) -> float:
        r"""Dimensionless Hubble parameter :math:`E(z)`.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Value of :math:`E(z)`.
        """
        return math.sqrt(self.Omega_m * (1 + z) ** 3 + self.Omega_L)

    def comoving_distance(self, z: float, nsteps: int = 2048) -> float:
        r"""Line-of-sight comoving distance.

        Parameters
        ----------
        z : float
            Redshift.
        nsteps : int, optional
            Number of Simpson rule integration steps. Default is ``2048``.

        Returns
        -------
        float
            Comoving distance in Mpc.

        Notes
        -----
        Uses Simpson's rule to evaluate

        .. math::
           D_C(z) = \frac{c}{H_0} \int_0^z \frac{\mathrm{d}z'}{E(z')}.
        """
        if z <= 0:
            return 0.0
        zs = np.linspace(0.0, z, nsteps + 1)
        Ez = np.sqrt(self.Omega_m * (1 + zs) ** 3 + self.Omega_L)
        w = np.ones_like(zs)
        w[1:-1:2] = 4
        w[2:-1:2] = 2
        integral = np.sum(w / Ez) * (z / nsteps) / 3.0
        return (self.c / self.H0) * integral

    def luminosity_distance(self, z: float) -> float:
        r"""Luminosity distance :math:`D_L`.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Luminosity distance in Mpc given by :math:`D_L = (1+z) D_C`.
        """
        return (1.0 + z) * self.comoving_distance(z)

    def distance_modulus(self, z: float) -> float:
        r"""Distance modulus :math:`\mu`.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Distance modulus in mag defined by

            .. math::
               \mu = 5\log_{10}\left(\frac{D_L}{10\,\mathrm{pc}}\right).
        """
        if z <= 0:
            return 0.0
        DL_Mpc = self.luminosity_distance(z)
        return 5.0 * math.log10(DL_Mpc * 1.0e6) - 5.0


def m_sn_from_pivot(z: float, z_pivot: float, m_p: float, cosmo: FlatLCDM) -> float:
    r"""SN apparent magnitude relative to a pivot redshift.

    Parameters
    ----------
    z : float
        Source redshift.
    z_pivot : float
        Pivot redshift anchoring the magnitude zero-point.
    m_p : float
        Apparent magnitude at the pivot redshift.
    cosmo : FlatLCDM
        Cosmology used to compute distance moduli.

    Returns
    -------
    float
        Theoretical apparent magnitude :math:`m(z)` in mag defined by

        .. math::
           m(z) = m_p + \mu(z) - \mu(z_{\text{pivot}}),

        where :math:`\mu` is the distance modulus.
    """
    dmu = cosmo.distance_modulus(z) - cosmo.distance_modulus(z_pivot)
    return m_p + dmu


def predict_unlensed_mag(z: float, m_p: float, z_pivot: float, cosmo: FlatLCDM) -> float:
    """Alias for :func:`m_sn_from_pivot` with explicit argument order.

    Parameters
    ----------
    z : float
        Source redshift.
    m_p : float
        Apparent magnitude at the pivot redshift.
    z_pivot : float
        Pivot redshift anchoring the magnitude zero-point.
    cosmo : FlatLCDM
        Cosmology used to compute distance moduli.

    Returns
    -------
    float
        Unlensed apparent magnitude at ``z`` in mag.
    """

    return m_sn_from_pivot(z, z_pivot, m_p, cosmo)


def profile_m_p_gaussian_prior(
    dprime: np.ndarray, invS: np.ndarray, mp_mean: float, mp_sigma: float
) -> tuple[float, float, float, float]:
    r"""Profile the pivot magnitude :math:`m_p` with a Gaussian prior.

    Let ``dprime`` be the data vector after subtracting all model terms
    except ``m_p`` and ``invS`` the inverse covariance matrix.  The quadratic
    form coefficients are

    .. math::
       a &= \mathbf{1}^\top S^{-1} \mathbf{1} + \sigma_{m_p}^{-2}\\
       b &= \mathbf{1}^\top S^{-1} d' + m_{p,\text{mean}}\,\sigma_{m_p}^{-2}\\
       c &= d'^\top S^{-1} d' + m_{p,\text{mean}}^2\,\sigma_{m_p}^{-2},

    yielding the marginalized log-likelihood

    .. math::
       \log \mathcal{L}_\mathrm{marg} = -\tfrac{1}{2}\left(c-\frac{b^2}{a}\right)
       -\tfrac{1}{2}\log a.

    Parameters
    ----------
    dprime : ndarray
        Data vector ``d'`` in mag.
    invS : ndarray
        Inverse covariance matrix :math:`S^{-1}`.
    mp_mean : float
        Mean of the Gaussian prior on :math:`m_p` in mag.
    mp_sigma : float
        Standard deviation of the prior in mag.

    Returns
    -------
    tuple of floats
        The coefficients ``(a, b, c)`` and the marginalized log-likelihood.
    """

    one = np.ones_like(dprime)
    a = float(one @ (invS @ one) + 1.0 / (mp_sigma**2))
    b = float(one @ (invS @ dprime) + mp_mean / (mp_sigma**2))
    c = float(dprime @ (invS @ dprime) + (mp_mean**2) / (mp_sigma**2))
    logL_marg = -0.5 * (c - (b**2) / a) - 0.5 * np.log(a)
    return a, b, c, float(logL_marg)
