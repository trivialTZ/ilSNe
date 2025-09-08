"""Glue utilities for :mod:`slsim`.

This module adapts the subset of SLSim needed by
``ilsne.slsim_yields.synthesize_yields`` to produce a catalog-level DataFrame
of ilSNe/slSNe.  Population synthesis and detection live in SLSim; this glue
layer standardizes the output schema expected by the ilSN likelihood.

Expected output schema (one row per SN image; units in mag and arcsec):
- lens_id: int — lens identifier
- sn_id: int — source/SN identifier
- image_id: int — image index for a given (lens_id, sn_id)
- kind: str — "il" (single) or "sl" (multiple)
- z_sn: float — source redshift
- z_lens: float — lens redshift
- R_arcsec: float — image-plane radius [arcsec]
- mu_fid: float — fiducial macro magnification for λ=1
- m_obs: float — observed lensed magnitude
- sigma_int_mag: float — intrinsic scatter component [mag]
- sigma_micro_mag: float — microlensing scatter component [mag]
- sigma_model_mag: float — modeling scatter component [mag]
- passed_detection: bool — selection flag
- p_det: float — detection probability
- n_epochs: int — number of survey visits
- filters_hit: str — concatenated band labels with detections
- t_first: float — time of first detection [days]
- t_last: float — time of last detection [days]
- mult: int — image multiplicity for the source
- is_il: bool — multiplicity == 1
- is_sl: bool — multiplicity > 1

If your SLSim version exposes a different API surface, you can provide an
external adapter via the environment variable ``ILSNE_SLSIM_ADAPTER`` with the
value ``"module.submodule:callable"``. The callable will be invoked with the
same keyword arguments as :func:`slsim_catalog` and must return a
``pandas.DataFrame`` with the schema above.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

logger = logging.getLogger("ilsne.slsim")

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .pivot import FlatLCDM
    from .slsim_yields import SurveyConfig

# Public entry: slsim_catalog(...)


def slsim_catalog(
    *,
    n_sn: int,
    survey: SurveyConfig,
    cosmo: FlatLCDM,
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,
    sigma_int: float,
    sigma_model: float,
    seed: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Build a catalog by invoking SLSim (or a pluggable adapter).

    Parameters
    ----------
    n_sn : int
        Number of SNe to draw.
    survey : SurveyConfig
        Observing configuration.
    cosmo : FlatLCDM
        Cosmology instance.
    z_pivot : float
        Pivot redshift where ``m_p`` is defined.
    mp_mean, mp_sigma : float
        Pivot magnitude prior :math:`m_p \sim \mathcal{N}(\mu,\sigma)` [mag].
    sigma_int, sigma_model : float
        Intrinsic and model uncertainties [mag].
    seed : int, optional
        RNG seed.
    verbose : bool, optional
        If ``True``, log and ``print`` which SLSim path is taken.

    Returns
    -------
    pandas.DataFrame
        Table with one row per SN image, including magnifications and
        shear/convergence fields (units: magnitudes and arcseconds).
    """

    # 1) Allow an external adapter override first
    adapter_env = os.environ.get("ILSNE_SLSIM_ADAPTER")
    if adapter_env:
        mod_name, _, fn_name = adapter_env.partition(":")
        if not mod_name or not fn_name:
            raise RuntimeError(
                "ILSNE_SLSIM_ADAPTER must be of the form 'module.submodule:callable'"
            )
        mod = __import__(mod_name, fromlist=[fn_name])
        fn = getattr(mod, fn_name, None)
        if fn is None or not callable(fn):
            raise RuntimeError("SLSim adapter not callable; check ILSNE_SLSIM_ADAPTER target")
        df = fn(
            n_sn=n_sn,
            survey={
                "name": survey.name,
                "epochs": survey.epochs,
                "days_between": survey.days_between,
                "depths": dict(survey.depths),
                "micro": dict(getattr(survey, "micro", {}) or {}),
                "slsim": dict(getattr(survey, "slsim", {}) or {}),
            },
            cosmo=cosmo,  # pass through; adapter may ignore or use
            z_pivot=z_pivot,
            mp_mean=mp_mean,
            mp_sigma=mp_sigma,
            sigma_int=sigma_int,
            sigma_model=sigma_model,
            seed=seed,
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "SLSim env adapter did not return a pandas.DataFrame; got " + type(df).__name__
            )
        if verbose:
            msg = f"slsim_catalog: using env adapter {adapter_env}"
            logger.info(msg)
            print(f"[ilsne] {msg}")
        return _validate_catalog(df)

    # 2) Try native SLSim adapter
    try:
        df = _native_slsim_catalog(
            n_sn=n_sn,
            survey=survey,
            cosmo=cosmo,
            z_pivot=z_pivot,
            mp_mean=mp_mean,
            mp_sigma=mp_sigma,
            sigma_int=sigma_int,
            sigma_model=sigma_model,
            seed=seed,
        )
        if verbose:
            msg = "slsim_catalog: using native SLSim adapter"
            logger.info(msg)
            print(f"[ilsne] {msg}")
        return _validate_catalog(df)
    except Exception:
        pass

    # 3) Fallback to legacy hooks / in-repo adapter
    candidate_hooks: Sequence[tuple[str, str]] = [
        ("ilsne.slsim_api_adapter", "from_slsim"),
    ]
    for mod_name, fn_name in candidate_hooks:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                df = fn(
                    n_sn=n_sn,
                    survey={
                        "name": survey.name,
                        "epochs": survey.epochs,
                        "days_between": survey.days_between,
                        "depths": dict(survey.depths),
                        "micro": dict(getattr(survey, "micro", {}) or {}),
                        "slsim": dict(getattr(survey, "slsim", {}) or {}),
                    },
                    cosmo=cosmo,
                    z_pivot=z_pivot,
                    mp_mean=mp_mean,
                    mp_sigma=mp_sigma,
                    sigma_int=sigma_int,
                    sigma_model=sigma_model,
                    seed=seed,
                )
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(
                        "SLSim hook did not return a pandas.DataFrame; got " + type(df).__name__
                    )
                if verbose:
                    msg = f"slsim_catalog: using fallback {mod_name}.{fn_name}"
                    logger.info(msg)
                    print(f"[ilsne] {msg}")
                return _validate_catalog(df)
        except Exception:
            continue
    raise RuntimeError("Could not locate a suitable SLSim catalog entry point")


def _native_slsim_catalog(
    *,
    n_sn: int,
    survey: SurveyConfig,
    cosmo: FlatLCDM,
    z_pivot: float,
    mp_mean: float,
    mp_sigma: float,
    sigma_int: float,
    sigma_model: float,
    seed: int | None,
) -> pd.DataFrame:
    """Native catalog generation using the SLSim LensPop pipeline.

    Parameters
    ----------
    n_sn : int
        Number of supernovae to draw.
    survey : SurveyConfig
        Survey configuration providing cadence and depth.
    cosmo : FlatLCDM
        Cosmology for distance calculations.
    z_pivot, mp_mean, mp_sigma : float
        Unused here but kept for a uniform adapter interface.
    sigma_int, sigma_model : float
        Intrinsic and modeling magnitude scatter [mag].
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Catalog rows matching the schema documented in :mod:`ilsne.slsim_glue`.
    """

    import os
    import pickle

    import numpy as np
    import slsim
    from astropy import units as u
    from astropy.cosmology import Flatw0waCDM
    from slsim.Deflectors.DeflectorPopulation.all_lens_galaxies import AllLensGalaxies
    from slsim.ImageSimulation.image_simulation import point_source_coordinate_properties
    from slsim.Lenses.lens_pop import LensPop
    try:
        from slsim.LOS.los_pop import LOSPop  # type: ignore
    except Exception:  # pragma: no cover - optional LOS support
        LOSPop = None  # type: ignore
    from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
    from slsim.Sources.SourcePopulation.point_plus_extended_sources import (
        PointPlusExtendedSources,
    )

    cosmo_astropy = Flatw0waCDM(H0=cosmo.H0, Om0=cosmo.Omega_m, w0=-1.0, wa=0.0)
    cfg = getattr(survey, "slsim", {}) or {}
    sky_area_deg2 = float(cfg.get("sky_area_deg2", 1.0))
    sky_area = sky_area_deg2 * u.deg**2
    pipe = SkyPyPipeline(
        skypy_config=cfg.get("skypy_config", None),
        sky_area=sky_area,
        filters=cfg.get("filters", None),
        cosmo=cosmo_astropy,
    )
    def_cfg = cfg.get("deflectors", {}) or {}
    def_sky_area = float(def_cfg.get("sky_area_deg2", sky_area_deg2)) * u.deg**2
    deflectors = AllLensGalaxies(
        red_galaxy_list=pipe.red_galaxies,
        blue_galaxy_list=pipe.blue_galaxies,
        kwargs_cut=def_cfg.get("kwargs_cut", {"z_min": 0.01, "z_max": 1.0}),
        kwargs_mass2light=def_cfg.get("kwargs_mass2light", {}),
        cosmo=cosmo_astropy,
        sky_area=def_sky_area,
    )

    pkl = os.path.join(
        os.path.dirname(slsim.__file__),
        "Sources",
        "SourceCatalogues",
        "SupernovaeCatalog",
        "supernovae_data.pkl",
    )
    with open(pkl, "rb") as f:
        supernovae_data = pickle.load(f)

    src_cfg = cfg.get("sources", {}) or {}
    sources = PointPlusExtendedSources(
        point_plus_extended_sources_list=supernovae_data,
        cosmo=cosmo_astropy,
        sky_area=sky_area,
        kwargs_cut=src_cfg.get("kwargs_cut", {"z_min": 0.01, "z_max": 1.5}),
        list_type=src_cfg.get("list_type", "list"),
    )

    # Optional LOS configuration
    los_pop_obj = None
    try:
        los_cfg = (cfg.get("los") or {})
    except Exception:
        los_cfg = {}
    if LOSPop is not None:
        try:
            nonlinear_h5 = los_cfg.get("nonlinear_h5")
            if not nonlinear_h5:
                nonlinear_h5 = os.environ.get("ILSNE_SLSIM_NONLINEAR_H5")
            los_pop_obj = LOSPop(
                los_bool=bool(los_cfg.get("enable", True)),
                nonlinear_los_bool=bool(los_cfg.get("nonlinear", False) or bool(nonlinear_h5)),
                nonlinear_correction_path=nonlinear_h5,
                no_correction_path=los_cfg.get("no_correction_h5", None),
            )
        except Exception:
            los_pop_obj = None

    lp_kwargs = dict(
        deflector_population=deflectors,
        source_population=sources,
        cosmo=cosmo_astropy,
        sky_area=sky_area,
    )
    if los_pop_obj is not None:
        lp_kwargs["los_pop"] = los_pop_obj
    lens_pop = LensPop(**lp_kwargs)

    rows = []
    np.random.seed(seed or 0)
    for i in range(int(n_sn)):
        lens = lens_pop.select_lens_at_random()
        z_lens = float(lens.deflector_redshift)
        z_sn = float(lens.max_redshift_source_class.redshift)
        phot = cfg.get("photometry", {}) or {}
        band = phot.get("band", "i")
        mag_zero_point = float(phot.get("mag_zero_point", 27))
        delta_pix = float(phot.get("delta_pix", 0.2))
        num_pix = int(phot.get("num_pix", 32))
        tpa = phot.get("transform_pix2angle", [[delta_pix, 0.0], [0.0, delta_pix]])
        transform_pix2angle = np.array(tpa, dtype=float)
        props = point_source_coordinate_properties(
            lens=lens,
            band=band,
            mag_zero_point=mag_zero_point,
            delta_pix=delta_pix,
            num_pix=num_pix,
            transform_pix2angle=transform_pix2angle,
        )
        pix = np.array(props["image_pix"])
        mult = len(pix)
        ang = pix @ transform_pix2angle
        r_arcsec = np.hypot(ang[:, 0], ang[:, 1])
        m_unl = lens.point_source_magnitude(band=band, lensed=False)[0]
        m_img = np.asarray(lens.point_source_magnitude(band=band, lensed=True)[0])
        mu_img = 10 ** (0.4 * (m_unl - m_img))
        micro = getattr(survey, "micro", {}) or {}
        if micro.get("kind", "const") == "const":
            sig_micro = float(micro.get("sigma", 0.0))
            sigma_micro_vec = np.full(mult, sig_micro, dtype=float)
        else:
            a = float(micro.get("a", 0.02))
            b = float(micro.get("b", 0.0))
            sigma_micro_vec = a + b * r_arcsec
        # External LOS fields (optional; default to zeros if missing)
        try:
            kext = float(getattr(lens.los, "convergence", lambda: 0.0)())
        except Exception:
            kext = 0.0
        try:
            gext = getattr(lens.los, "shear", (0.0, 0.0))
            if callable(gext):
                gext = gext()
            gamma1_ext, gamma2_ext = (
                (float(gext[0]), float(gext[1])) if isinstance(gext, (list, tuple)) else (0.0, 0.0)
            )
        except Exception:
            gamma1_ext, gamma2_ext = 0.0, 0.0

        for j in range(mult):
            rows.append(
                {
                    "lens_id": i,
                    "sn_id": i,
                    "image_id": j,
                    "kind": "sl" if mult > 1 else "il",
                    "z_sn": z_sn,
                    "z_lens": z_lens,
                    "R_arcsec": float(r_arcsec[j]),
                    "mu_fid": float(mu_img[j]),
                    "m_obs": float(m_img[j]),
                    "sigma_int_mag": float(sigma_int),
                    "sigma_micro_mag": float(sigma_micro_vec[j]),
                    "sigma_model_mag": float(sigma_model),
                    "passed_detection": True,
                    "p_det": 1.0,
                    "n_epochs": int(survey.epochs),
                    "filters_hit": "".join(sorted(survey.depths)),
                    "t_first": 0.0,
                    "t_last": survey.days_between * (survey.epochs - 1),
                    "mult": mult,
                    "is_il": mult == 1,
                    "is_sl": mult > 1,
                    # External LOS convergence and shear
                    "kappa_ext": float(kext),
                    "gamma1_ext": float(gamma1_ext),
                    "gamma2_ext": float(gamma2_ext),
                }
            )

    return pd.DataFrame(rows)


def _validate_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has the required columns and basic types.

    This is a light runtime check to catch adapter mismatch early with a clear message.
    """

    required = [
        "lens_id",
        "sn_id",
        "image_id",
        "kind",
        "z_sn",
        "z_lens",
        "R_arcsec",
        "mu_fid",
        "m_obs",
        "sigma_int_mag",
        "sigma_micro_mag",
        "sigma_model_mag",
        "passed_detection",
        "p_det",
        "n_epochs",
        "filters_hit",
        "t_first",
        "t_last",
        "mult",
        "is_il",
        "is_sl",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("SLSim catalog is missing required columns: " + ", ".join(missing))
    return df


__all__ = ["slsim_catalog"]
