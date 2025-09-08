"""Survey yield synthesis front-end.

Provides a small configuration dataclass and a wrapper that calls an SLSim
adapter to generate ilSN/slSN catalogs in a standardized schema used by the
likelihood and demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .slsim_glue import slsim_catalog


@dataclass
class SurveyConfig:
    """Minimal survey description for yield synthesis.

    Parameters
    ----------
    name : str, optional
        Identifier embedded into outputs.
    epochs : int, optional
        Number of visits per field.
    days_between : float, optional
        Cadence spacing in days.
    depths : dict[str, float], optional
        Per-band 5σ limiting magnitudes.
    micro : dict, optional
        Microlensing scatter control; either ``{"kind": "const", "sigma": 0.02}``
        or ``{"kind": "linear", "a": 0.02, "b": 0.0}`` for ``a + b R``.
    slsim : dict, optional
        Native SLSim overrides (e.g., pipeline options) when available. The
        native adapter recognizes an optional ``{"los": {...}}`` subtree with
        keys like ``enable`` (bool), ``nonlinear`` (bool), ``nonlinear_h5``
        (path), and ``no_correction_h5`` (path) to configure LOS sampling.
    """

    name: str = "DEMO"
    epochs: int = 6
    days_between: float = 5.0
    depths: dict[str, float] | None = None
    micro: dict[str, Any] | None = None
    slsim: dict[str, Any] | None = None  # pass-through overrides for native SLSim

    def __post_init__(self) -> None:
        if self.depths is None:
            self.depths = {"r": 25.0, "i": 24.5}
        if self.micro is None:
            self.micro = {"kind": "const", "sigma": 0.0}
        if self.slsim is None:
            self.slsim = {}


def synthesize_yields(
    *,
    n_sn: int,
    survey: SurveyConfig,
    cosmo: Any,
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,
    sigma_int: float,
    sigma_model: float,
    seed: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Generate an ilSN/slSN catalog with a standardized schema.

    Parameters
    ----------
    n_sn : int
        Number of supernova sources to draw.
    survey : SurveyConfig
        Observing setup (cadence and depth) and microlensing controls.
    cosmo : Any
        Cosmology object passed through to the adapter; exact type depends on
        the adapter implementation.
    z_pivot, mp_mean, mp_sigma : float
        Pivot-redshift and Gaussian prior describing the SN magnitude anchor.
    sigma_int, sigma_model : float
        Intrinsic and modeling photometric scatter components [mag].
    seed : int, optional
        RNG seed for reproducibility.
    verbose : bool, optional
        If ``True``, print which adapter path was used.

    Returns
    -------
    pandas.DataFrame
        Table with one row per SN image following the schema validated by
        :func:`ilsne.slsim_glue._validate_catalog` (columns include ``lens_id``,
        ``sn_id``, ``image_id``, ``kind``, ``z_sn``, ``R_arcsec``, ``mu_fid``,
        and scatter components in magnitudes).
    """
    df = slsim_catalog(
        n_sn=n_sn,
        survey=survey,
        cosmo=cosmo,
        z_pivot=z_pivot,
        mp_mean=mp_mean,
        mp_sigma=mp_sigma,
        sigma_int=sigma_int,
        sigma_model=sigma_model,
        seed=seed,
        verbose=verbose,
    )
    if "is_il" not in df.columns:
        df["is_il"] = df["mult"].astype(int) == 1
    if "is_sl" not in df.columns:
        df["is_sl"] = df["mult"].astype(int) > 1
    # Backward/forward compatibility aliases used in demos/notebooks
    # Prefer explicit *_mag names; create plain aliases if missing.
    if "sigma_int" not in df.columns and "sigma_int_mag" in df.columns:
        df["sigma_int"] = df["sigma_int_mag"]
    if "sigma_model" not in df.columns and "sigma_model_mag" in df.columns:
        df["sigma_model"] = df["sigma_model_mag"]
    if "sigma_micro" not in df.columns and "sigma_micro_mag" in df.columns:
        df["sigma_micro"] = df["sigma_micro_mag"]
    return df


__all__ = ["SurveyConfig", "synthesize_yields"]
