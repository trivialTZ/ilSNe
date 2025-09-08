"""Basic tests for the pivot-magnitude and λ inference utilities."""

from ilsne import (
    FlatLCDM,
    LensDataset,
    LensedSN,
    infer_lambda_for_lens,
    m_sn_from_pivot,
    sigma_micro_constant,
)


def test_pivot_relative_dm() -> None:
    """Relative distance modulus should be positive and smallish."""
    cosmo = FlatLCDM()
    m0 = m_sn_from_pivot(0.3, 0.1, m_p=19.0, cosmo=cosmo) - 19.0
    assert abs(m0) > 0 and abs(m0) < 5.0


def test_lambda_recovery_constant_sigma() -> None:
    """Smoke test for λ inference with constant microlensing scatter."""
    cosmo = FlatLCDM()
    sne = [
        LensedSN(z=0.6, mu_model=2.0, m_obs=22.0, sigma_m=0.2),
        LensedSN(z=0.8, mu_model=3.0, m_obs=22.5, sigma_m=0.2),
        LensedSN(z=0.7, mu_model=1.5, m_obs=21.8, sigma_m=0.2),
        LensedSN(z=0.9, mu_model=4.0, m_obs=23.1, sigma_m=0.2),
        LensedSN(z=0.5, mu_model=1.3, m_obs=21.7, sigma_m=0.2),
    ]
    lens = LensDataset(name="A", sne=sne)
    res = infer_lambda_for_lens(
        lens,
        cosmo,
        z_pivot=0.1,
        mp_mean=18.966,
        mp_sigma=0.02,
        sigma_micro_fn=sigma_micro_constant(0.05),
    )
    assert 0.7 <= res["lam_mean"] <= 1.4
    assert res["lam_std"] > 0
