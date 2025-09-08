from __future__ import annotations

import numpy as np
import pandas as pd

from ilsne.pivot import FlatLCDM
from ilsne.slsim_glue import slsim_catalog
from ilsne.slsim_yields import SurveyConfig


def test_slsim_catalog_env_adapter_valid(monkeypatch):
    monkeypatch.setenv("ILSNE_SLSIM_ADAPTER", "ilsne.test_adapters:dummy_slsim_adapter")
    cfg = SurveyConfig(name="t", epochs=5, days_between=2.0, depths={"r": 25.0})
    df = slsim_catalog(
        n_sn=20,
        survey=cfg,
        cosmo=FlatLCDM(),
        z_pivot=0.1,
        mp_mean=19.0,
        mp_sigma=0.01,
        sigma_int=0.1,
        sigma_model=0.05,
        seed=123,
    )
    assert isinstance(df, pd.DataFrame)
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
    for c in required:
        assert c in df.columns
    # basic invariants
    assert np.all((df["mu_fid"]) > 0)


def test_slsim_catalog_env_adapter_missing_columns(monkeypatch):
    monkeypatch.setenv("ILSNE_SLSIM_ADAPTER", "ilsne.test_adapters:bad_adapter_missing_cols")
    cfg = SurveyConfig(name="t", epochs=5, days_between=2.0, depths={"r": 25.0})
    from pytest import raises

    with raises(ValueError):
        slsim_catalog(
            n_sn=5,
            survey=cfg,
            cosmo=FlatLCDM(),
            z_pivot=0.1,
            mp_mean=19.0,
            mp_sigma=0.01,
            sigma_int=0.1,
            sigma_model=0.05,
            seed=1,
        )
