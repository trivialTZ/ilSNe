from ilsne.microlens import sigma_micro_parametric


def test_sigma_micro_parametric_monotonic_and_nonnegative():
    fn = sigma_micro_parametric(0.1, 0.2, 1.0)
    r_vals = [0.5, 1.0, 2.0]
    sigs = [fn(r) for r in r_vals]
    assert fn(None) == 0.0
    assert all(s >= 0 for s in sigs)
    assert sigs[0] < sigs[1] < sigs[2]

    fn2 = sigma_micro_parametric(-0.5, -0.1, 1.0)
    assert fn2(1.0) == 0.0
