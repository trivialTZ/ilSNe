import numpy as np

from ilsne.magnification import LensDataset, LensedSN, infer_lambda_for_lens
from ilsne.pivot import FlatLCDM


def _toy_lens(lam_true=1.08):
    # one toy SN; make the observed mag consistent with lam_true
    z = np.array([0.7])
    mu_model = np.array([2.5])
    cosmo = FlatLCDM()
    z_pivot = 0.10
    dmu = cosmo.distance_modulus(z[0]) - cosmo.distance_modulus(z_pivot)
    # model: m = m_p + dmu - 2.5 log10(mu_model) + 5 log10(lam_true)
    mp = 19.0
    m = mp + dmu - 2.5 * np.log10(mu_model[0]) + 5 * np.log10(lam_true)
    sn = LensedSN(z=float(z[0]), mu_model=float(mu_model[0]), m_obs=float(m), sigma_m=0.05)
    return LensDataset("toy", [sn])


def test_lambda_injection_recovery():
    cosmo = FlatLCDM()
    lens = _toy_lens(lam_true=1.08)
    res = infer_lambda_for_lens(
        lens, cosmo, z_pivot=0.10, mp_mean=19.0, mp_sigma=0.01, lam_grid=(0.8, 1.4, 1201)
    )
    assert abs(res["lam_mean"] - 1.08) < 0.02


def test_H0_invariance():
    lens = _toy_lens(lam_true=1.00)
    res1 = infer_lambda_for_lens(
        lens, FlatLCDM(H0=70.0), 0.10, 19.0, 0.01, lam_grid=(0.8, 1.2, 801)
    )
    res2 = infer_lambda_for_lens(
        lens, FlatLCDM(H0=75.0), 0.10, 19.0, 0.01, lam_grid=(0.8, 1.2, 801)
    )
    assert abs(res1["lam_mean"] - res2["lam_mean"]) < 1e-3


def test_pantheon_block_interface(monkeypatch):
    # Patch hierArc access to avoid disk and ensure deterministic output
    import ilsne.hierarc_glue as glue
    from tests.helpers import make_fake_pantheon

    z = [0.1, 0.2]
    # Choose m such that the analytic solution is easy to verify
    # With C = I and dmu = 0 at z=0.1 for the first point, the exact values are not important here;
    # we only verify that the block returns a sensible (mean, sigma) tuple.
    m = [19.0, 19.2]
    cov = np.eye(2)
    monkeypatch.setattr(glue, "_PantheonPlusData", make_fake_pantheon(z, m, cov), raising=True)

    mp = glue.PantheonPivotBlock(z_anchor=0.1).fit_pivot_mean_std()
    assert isinstance(mp, tuple) and len(mp) == 2
    assert all(np.isfinite(mp)) and mp[1] > 0
