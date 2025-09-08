from __future__ import annotations

import numpy as np
import pandas as pd

from ilsne import FlatLCDM
from ilsne.io import lens_from_dataframe
from ilsne.magnification import infer_lambda_for_lens
from ilsne.stacking import product_stack_common_lambda


def _make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.2, 0.8, size=n)
    mu = np.exp(rng.normal(0.0, 0.15, size=n))  # lognormal-ish around 1
    m_unl = 20.0 + 5.0 * np.log10(1.0 + z)  # arbitrary monotonic
    m_obs = m_unl - 2.5 * np.log10(mu)  # true λ=1
    sigma_m = np.full(n, 0.12)  # includes int+micro in quadrature
    R = rng.uniform(0.3, 6.0, size=n)
    return pd.DataFrame({"z": z, "mu_model": mu, "m_obs": m_obs, "sigma_m": sigma_m, "R_arcsec": R})


def _stack_sigma(rows: pd.DataFrame, n: int, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    samp = rows.sample(n=n, random_state=int(rng.integers(1, 1_000_000)))
    posts = []
    cosmo = FlatLCDM(H0=70, Omega_m=0.3)
    for _, r in samp.iterrows():
        lens = lens_from_dataframe("L", pd.DataFrame([r]))
        post = infer_lambda_for_lens(
            lens,
            cosmo=cosmo,
            z_pivot=0.1,
            mp_mean=19.0,
            mp_sigma=0.10,
            lam_grid=(0.6, 1.4, 1201),
        )
        posts.append(post)
    joint = product_stack_common_lambda(posts, ngrid=1601)
    return float(joint["lam_std"])


def test_sigma_lambda_decreases_with_N() -> None:
    rows = _make_rows(128, seed=7)
    s8 = _stack_sigma(rows, 8)
    s32 = _stack_sigma(rows, 32)
    s64 = _stack_sigma(rows, 64)
    # should shrink roughly ~ 1/sqrt(N)
    assert s64 < s32 < s8
