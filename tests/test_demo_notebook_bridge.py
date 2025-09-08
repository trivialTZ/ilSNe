from __future__ import annotations

import numpy as np

from ilsne.pivot import FlatLCDM
from ilsne.slsim_yields import SurveyConfig, synthesize_yields


def test_demo_notebook_sigma_columns_present_and_usable(monkeypatch):
    # Ensure we use the in-repo adapter (no external slsim requirement)
    monkeypatch.delenv("ILSNE_SLSIM_ADAPTER", raising=False)
    cfg = SurveyConfig(name="demo", epochs=6, days_between=5.0, depths={"r": 25.0, "i": 24.5})
    df = synthesize_yields(
        n_sn=10,
        survey=cfg,
        cosmo=FlatLCDM(),
        z_pivot=0.2,
        mp_mean=19.0,
        mp_sigma=0.01,
        sigma_int=0.10,
        sigma_model=0.02,
        seed=123,
    )
    # Columns used by the demo notebook code path
    assert "sigma_int" in df.columns  # alias added by synthesize_yields
    assert "sigma_micro_mag" in df.columns
    assert "sigma_model_mag" in df.columns

    # Can compute the combined scatter exactly as in the notebook snippet
    sig = np.sqrt(df["sigma_int"] ** 2 + df["sigma_micro_mag"] ** 2 + df["sigma_model_mag"] ** 2)
    assert np.all(np.isfinite(sig))
