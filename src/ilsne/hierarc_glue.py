"""
Pantheon+ pivot prior glue with hierArc-only backend.

This module requires hierArc>=1.1 at runtime for the Pantheon+ data access.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from .cosmology_small import delta_mu
try:  # soft import to allow module import without hierArc installed
    from hierarc.Likelihood.SneLikelihood.sne_pantheon_plus import (
        PantheonPlusData as _PantheonPlusData,
    )
except Exception:  # pragma: no cover - exercised in tests by monkeypatching
    _PantheonPlusData = None  # type: ignore[assignment]

logger = logging.getLogger("ilsne.hierarc")


class PantheonPivotBlock:
    """
    Unified pivot interface.

    Parameters
    ----------
    z_anchor : float
        Pivot redshift anchoring m_p.
    data_path : str, optional
        Path to Pantheon+ dataset. May be either a directory containing the
        covariance and data table, or a direct path to the ``.cov``/``.dat`` file
        (in which case its parent directory is used). If None, local defaults
        under ``data/`` are searched.
    config : dict, optional
        Minimal cosmology dict (keys: 'omegam','w0','wa') for Δμ(z) shape.
    """

    def __init__(
        self,
        z_anchor: float,
        data_path: str | None = None,
        config: dict | None = None,
    ):
        self.z_anchor = float(z_anchor)
        # 'data_path' preserved for API compatibility but unused in hierArc-only mode
        self._data_dir = data_path
        self._config = config or {"omegam": 0.3, "w0": -1.0, "wa": 0.0}
        self._cache: tuple[float, float] | None = None
        # Fail fast if hierArc is not available
        if _PantheonPlusData is None:
            raise ImportError(
                "hierArc is required for Pantheon+ pivot. Please install hierarc>=1.1."
            )
        # Expose the constructor for tests to monkeypatch if needed
        self._PantheonPlusData = _PantheonPlusData

    def fit_pivot_mean_std(self) -> tuple[float, float]:
        """
        Return (mp_mean, mp_sigma) in magnitudes.

        When hierArc is present, we load the Pantheon+ arrays and solve the GLS intercept
        (m_p = (1^T C^{-1} y)/(1^T C^{-1} 1)) where y = m_obs - [mu(z) - mu(z_anchor)] in mag.
        Otherwise compute from local Pantheon+ vector+covariance.
        """
        if self._cache is not None:
            return self._cache

        # Suppress pandas FutureWarning emitted by hierArc's use of
        # pd.read_csv(..., delim_whitespace=True). Safe to ignore here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r".*delim_whitespace.*",
            )
            pp = self._PantheonPlusData()

        # Extract arrays from hierArc Pantheon+ data object (with light shims)
        z_arr = getattr(pp, "zCMB", None)
        if z_arr is None:
            z_arr = getattr(pp, "zcmb", None)
        m_arr = getattr(pp, "m_obs", None)
        C_arr = getattr(pp, "cov_mag_b", None)
        if C_arr is None:
            C_arr = getattr(pp, "cov_mag", None)
        if z_arr is None or m_arr is None or C_arr is None:
            raise AttributeError(
                "PantheonPlusData missing expected attributes (zCMB, m_obs, cov_mag_b)."
            )
        z = np.asarray(z_arr, dtype=float)
        m = np.asarray(m_arr, dtype=float)
        C = np.asarray(C_arr, dtype=float)

        # Build Δμ relative to the anchor redshift using local cosmology util
        om = float(self._config.get("omegam", 0.3))
        w0 = float(self._config.get("w0", -1.0))
        wa = float(self._config.get("wa", 0.0))
        dmu = np.array([delta_mu(zi, self.z_anchor, om, w0, wa) for zi in z], dtype=float)
        y = m - dmu

        # Solve GLS intercept via stable Cholesky solves
        L = np.linalg.cholesky(C)
        ones = np.ones_like(y)
        y1 = np.linalg.solve(L, ones)
        y2 = np.linalg.solve(L, y)
        Cinv1 = np.linalg.solve(L.T, y1)
        Cinv_y = np.linalg.solve(L.T, y2)
        denom = float(np.dot(ones, Cinv1))  # 1^T C^{-1} 1
        numer = float(np.dot(ones, Cinv_y))  # 1^T C^{-1} y
        mp_mean = numer / denom
        mp_sigma = float(np.sqrt(1.0 / denom))
        self._cache = (float(mp_mean), float(mp_sigma))
        return self._cache


class TDMagBlock:
    """Placeholder for time-delay glue block (Stage 7)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise NotImplementedError("Time-delay glue is Stage 7; not implemented yet.")
