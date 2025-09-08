import numpy as np

from ilsne import FlatLCDM, LensDataset, LensedSN, infer_lambda_for_lens, m_sn_from_pivot


def test_lambda_recovers_truth():
    rng = np.random.default_rng(0)
    cosmo = FlatLCDM()
    z_pivot = 0.1
    m_p_true = 20.0
    lam_true = 1.12
    sigma_m = 0.05
    z = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    mu = np.array([1.5, 2.0, 1.8, 2.2, 1.6])

    m_obs = []
    for zi, mui in zip(z, mu, strict=False):
        base = m_sn_from_pivot(zi, z_pivot, m_p_true, cosmo)
        mag = base - 2.5 * np.log10(mui) + 5.0 * np.log10(lam_true)
        m_obs.append(mag + rng.normal(0.0, sigma_m))

    sne = [
        LensedSN(z=zi, mu_model=mui, m_obs=mi, sigma_m=sigma_m)
        for zi, mui, mi in zip(z, mu, m_obs, strict=False)
    ]
    lens = LensDataset(name="L", sne=sne)
    res = infer_lambda_for_lens(
        lens,
        cosmo,
        z_pivot=z_pivot,
        mp_mean=m_p_true,
        mp_sigma=0.01,
        sigma_int=0.05,
    )

    assert abs(res["lam_map"] - lam_true) <= res["lam_std"]
