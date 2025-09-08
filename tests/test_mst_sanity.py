"""Sanity checks for mass-sheet transform invariance and injection."""

import numpy as np
import pandas as pd

from ilsne import FlatLCDM, infer_lambda_for_lens, m_sn_from_pivot
from ilsne.io import lens_from_dataframe


def _toy_df() -> pd.DataFrame:
    cosmo = FlatLCDM()
    z_pivot = 0.10
    m_p_true = 19.0
    rows = []
    for z, mu, s, r in [
        (0.6, 1.8, 0.17, 0.9),
        (0.7, 2.5, 0.18, 1.2),
        (0.8, 3.2, 0.19, 1.6),
        (0.9, 4.0, 0.20, 2.1),
    ]:
        m_app = m_sn_from_pivot(z, z_pivot, m_p_true, cosmo)
        m_obs = m_app - 2.5 * np.log10(mu)
        rows.append(dict(z=z, mu_model=mu, m_obs=m_obs, sigma_m=s, R_arcsec=r))
    return pd.DataFrame(rows)


def _toy_lens():
    return lens_from_dataframe("toy", _toy_df())


def test_pivot_blocks_H0_dependence():
    lens = _toy_lens()
    posts = []
    lam_grid = None
    for H0 in (60.0, 70.0, 80.0):
        res = infer_lambda_for_lens(
            lens,
            FlatLCDM(H0=H0, Omega_m=0.3),
            z_pivot=0.10,
            mp_mean=19.0,
            mp_sigma=0.01,
            lam_grid=(0.8, 1.2, 801),
            sigma_int=0.1,
        )
        lam_grid = res["lam_grid"]
        posts.append(res["posterior"] / np.trapezoid(res["posterior"], lam_grid))
    means = [np.trapezoid(lam_grid * p, lam_grid) for p in posts]
    assert max(means) - min(means) < 1e-2


def test_mst_injection_is_recovered():
    lam_true = 1.12
    df = _toy_df()
    df_inj = df.copy()
    df_inj["m_obs"] = df_inj["m_obs"] + 5.0 * np.log10(lam_true)
    lens_inj = lens_from_dataframe("toy_inj", df_inj)
    res = infer_lambda_for_lens(
        lens_inj,
        FlatLCDM(),
        z_pivot=0.10,
        mp_mean=19.0,
        mp_sigma=0.01,
        lam_grid=(0.9, 1.3, 801),
        sigma_int=0.1,
    )
    assert abs(res["lam_mean"] - lam_true) < 0.02
