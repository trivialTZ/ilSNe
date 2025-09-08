import numpy as np

import ilsne.hierarc_glue as glue
from tests.helpers import make_fake_pantheon


def test_gls_pivot_matches_bruteforce(monkeypatch):
    # Synthetic Pantheon+ dataset
    rng = np.random.default_rng(0)
    z = np.linspace(0.01, 1.0, 60)
    z_anchor = 0.1
    true_mp = 24.0
    from ilsne.cosmology_small import delta_mu

    dmu = np.array([delta_mu(zi, z_anchor) for zi in z])
    # Identity covariance with known sigma
    sig = 0.02
    cov = np.eye(z.size) * sig**2
    m = true_mp + dmu + rng.normal(0, sig, size=z.size)

    # Mock hierArc's PantheonPlusData
    monkeypatch.setattr(glue, "_PantheonPlusData", make_fake_pantheon(z, m, cov), raising=True)

    # Analytic GLS solution via our block
    block = glue.PantheonPivotBlock(z_anchor=z_anchor)
    mp_mean, mp_sigma = block.fit_pivot_mean_std()

    # Brute-force check of the intercept likelihood over a grid
    y = m - dmu
    Ci = np.linalg.inv(cov)
    one = np.ones_like(y)
    grid = np.linspace(true_mp - 0.1, true_mp + 0.1, 2001)
    # logL(mp) ∝ -0.5 (y - mp*1)^T C^{-1} (y - mp*1)
    def logL(mp):
        r = y - mp * one
        return -0.5 * float(r @ Ci @ r)

    vals = np.array([logL(g) for g in grid])
    mp_bf = float(grid[np.argmax(vals)])

    # Tolerances reflect grid spacing and stochastic noise
    assert abs(mp_mean - mp_bf) < 2e-3
    # Sigma^2 = 1 / (1^T C^{-1} 1) for C = sig^2 I → sig/sqrt(N)
    assert abs(mp_sigma - sig / (len(z) ** 0.5)) < 5e-4
