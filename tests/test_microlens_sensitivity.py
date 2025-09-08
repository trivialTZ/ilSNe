from ilsne.microlens import sigma_micro_parametric


def test_parametric_non_negative():
    fn = sigma_micro_parametric(alpha=0.05, beta=0.02, Reff_arcsec=1.0)
    assert fn(0.5) >= 0 and fn(2.0) >= 0


def test_parametric_monotonic_beta_pos():
    fn = sigma_micro_parametric(alpha=0.01, beta=0.05, Reff_arcsec=1.0)
    assert fn(0.5) < fn(1.0) < fn(2.0)
