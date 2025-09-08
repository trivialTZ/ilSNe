import numpy as np
import pytest

from ilsne.pivot import FlatLCDM
from ilsne.slsim_glue import slsim_catalog
from ilsne.slsim_yields import SurveyConfig


def test_native_slsim_smoke():
    cfg = SurveyConfig()
    try:
        df = slsim_catalog(
            n_sn=10,
            survey=cfg,
            cosmo=FlatLCDM(),
            z_pivot=0.1,
            mp_mean=19.0,
            mp_sigma=0.01,
            sigma_int=0.1,
            sigma_model=0.05,
            seed=1,
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"native slsim path unavailable: {e}")
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
    ]
    for c in required:
        assert c in df.columns
    assert np.all(df["is_il"] == (df["mult"] == 1))
    assert np.all(df["is_sl"] == (df["mult"] > 1))
    assert df.shape[0] > 0
