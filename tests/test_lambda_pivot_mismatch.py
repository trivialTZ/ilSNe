from __future__ import annotations

import math

from ilsne import FlatLCDM, LensDataset, LensedSN, infer_lambda_for_lens


def _make_lens(mp_data: float, z_pivot: float = 0.10) -> LensDataset:
    cosmo = FlatLCDM()
    rows = []
    for z, mu, s, r in [
        (0.70, 2.5, 0.18, 1.2),
        (0.60, 1.8, 0.18, 0.9),
        (0.90, 4.0, 0.20, 2.1),
        (0.50, 1.3, 0.18, 0.6),
        (0.80, 3.2, 0.19, 1.6),
    ]:
        dmu = cosmo.distance_modulus(z) - cosmo.distance_modulus(z_pivot)
        m_unl = mp_data + dmu
        m_obs = m_unl - 2.5 * math.log10(mu)
        rows.append(LensedSN(z=z, mu_model=mu, m_obs=m_obs, sigma_m=s, R_arcsec=r))
    return LensDataset(name="demo", sne=rows)


def test_lambda_unity_when_prior_matches_data():
    lens = _make_lens(mp_data=19.0)
    res = infer_lambda_for_lens(
        lens=lens,
        cosmo=FlatLCDM(),
        z_pivot=0.10,
        mp_mean=19.0,
        mp_sigma=0.01,
        lam_grid=(0.7, 1.4, 1201),
        sigma_int=0.05,
    )
    assert abs(res["lam_mean"] - 1.0) < 0.02


def test_lambda_bias_matches_implied_from_pivot_shift():
    # Prior is shifted by +0.436 mag → λ ≈ 10^{-0.436/5} ≈ 0.819
    lens = _make_lens(mp_data=19.0)
    res = infer_lambda_for_lens(
        lens=lens,
        cosmo=FlatLCDM(),
        z_pivot=0.10,
        mp_mean=19.436,
        mp_sigma=0.01,
        lam_grid=(0.7, 1.4, 1201),
        sigma_int=0.05,
    )
    lam_implied = 10 ** (-(19.436 - 19.0) / 5.0)
    assert abs(res["lam_mean"] - lam_implied) < 0.01
