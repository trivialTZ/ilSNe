import numpy as np

from ilsne import FlatLCDM, predict_unlensed_mag, profile_m_p_gaussian_prior


def test_profile_mp_matches_numeric_maximum():
    # build synthetic dataset with known m_p_true; no lensing (mu=1, lam=1)
    cosmo = FlatLCDM()
    z_pivot = 0.1
    m_p_true = 19.0
    zs = np.array([0.3, 0.5, 0.7, 0.9])
    m_obs = np.array([predict_unlensed_mag(z, m_p_true, z_pivot, cosmo) for z in zs])
    sigma = 0.05
    Cinv = np.diag(np.full_like(zs, 1.0 / sigma**2, dtype=float))
    # no prior (very broad) vs numeric scan
    dmu = np.array([cosmo.distance_modulus(z) - cosmo.distance_modulus(z_pivot) for z in zs])
    # model: m_obs ≈ m_p + dmu  => d' = m_obs - dmu
    dprime = m_obs - dmu
    a = (np.ones_like(zs) @ (Cinv @ np.ones_like(zs))) + 1.0 / (1e6**2)
    b = (np.ones_like(zs) @ (Cinv @ dprime)) + m_p_true / (1e6**2)
    _, _, _, logL_prof = profile_m_p_gaussian_prior(dprime, Cinv, m_p_true, 1e6)
    # numeric grid
    mp_grid = np.linspace(m_p_true - 0.5, m_p_true + 0.5, 1001)
    logLs = []
    for mp in mp_grid:
        r = dprime - mp * np.ones_like(zs)
        logLs.append(-0.5 * (r @ (Cinv @ r)))
    logLs = np.array(logLs)
    mp_num = mp_grid[np.argmax(logLs)]
    assert abs(mp_num - (b / a)) < 1e-3
    assert np.isfinite(logL_prof) and logL_prof < 0
