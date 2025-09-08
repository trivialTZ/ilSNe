import numpy as np
import pandas as pd

from ilsne import FlatLCDM, infer_lambda_for_lens, sigma_micro_constant  # noqa: F401
from ilsne.io import lens_from_dataframe


def _toy_lens(
    N=1,
    lam_true=1.0,
    mu=2.0,
    z=0.6,
    z_pivot=0.1,
    mp_mean=38.21,
    mp_sigma=0.01,
    sig_int=0.10,
    sig_micro=0.10,
):
    rng = np.random.default_rng(123)
    cosmo = FlatLCDM()
    dm = cosmo.distance_modulus(z) - cosmo.distance_modulus(z_pivot)
    m_unlensed = mp_mean + dm
    m_true = m_unlensed - 2.5 * np.log10(mu) + 5 * np.log10(lam_true)
    m_obs = m_true + rng.normal(0, np.sqrt(sig_int**2 + sig_micro**2), size=N)
    df = pd.DataFrame(
        {
            "z": [z] * N,
            "mu_model": [mu] * N,
            "m_obs": list(m_obs),
            "sigma_m": [np.sqrt(sig_int**2 + sig_micro**2)] * N,
            "R_arcsec": [1.0] * N,
        }
    )
    return lens_from_dataframe("T", df)


def test_single_sn_map_near_one():
    L = _toy_lens(N=1)
    res = infer_lambda_for_lens(
        L,
        FlatLCDM(),
        z_pivot=0.1,
        mp_mean=38.21,
        mp_sigma=0.01,
        sigma_micro_fn=None,
        sigma_int=0.0,
        sigma_micro_included_in_sigma_m=True,
    )
    assert 0.9 < res["lam_map"] < 1.1


def test_sigma_scales_with_rootN():
    L5 = _toy_lens(N=5)
    L50 = _toy_lens(N=50)
    res5 = infer_lambda_for_lens(
        L5,
        FlatLCDM(),
        z_pivot=0.1,
        mp_mean=38.21,
        mp_sigma=0.01,
        sigma_micro_fn=None,
        sigma_int=0.0,
        sigma_micro_included_in_sigma_m=True,
    )
    res50 = infer_lambda_for_lens(
        L50,
        FlatLCDM(),
        z_pivot=0.1,
        mp_mean=38.21,
        mp_sigma=0.01,
        sigma_micro_fn=None,
        sigma_int=0.0,
        sigma_micro_included_in_sigma_m=True,
    )
    assert res50["lam_std"] < 0.5 * res5["lam_std"]


def test_edge_warning_when_map_at_grid_edge(recwarn):
    pass
