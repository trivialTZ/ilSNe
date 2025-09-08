from __future__ import annotations

import numpy as np

from ilsne.pivot import FlatLCDM
from ilsne.slsim_yields import SurveyConfig, synthesize_yields


def test_synthesize_yields_with_env_adapter(monkeypatch):
    # Use the env adapter; synthesize_yields should route through slsim_catalog which honors it
    monkeypatch.setenv("ILSNE_SLSIM_ADAPTER", "ilsne.test_adapters:dummy_slsim_adapter")
    cfg = SurveyConfig(name="demo", epochs=8, days_between=2.5, depths={"r": 25.0, "i": 24.7})
    df = synthesize_yields(
        n_sn=30,
        survey=cfg,
        cosmo=FlatLCDM(),
        z_pivot=0.1,
        mp_mean=18.966,
        mp_sigma=0.008,
        sigma_int=0.1,
        sigma_model=0.05,
        seed=7,
    )
    assert len(df) == 30
    # Check type/units invariants
    assert set(df["kind"].unique()).issubset({"il", "sl"})
    assert df["n_epochs"].nunique() == 1
    # Boolean consistency
    assert np.all(df["is_il"] == (df["mult"] == 1))
    assert np.all(df["is_sl"] == (df["mult"] > 1))
