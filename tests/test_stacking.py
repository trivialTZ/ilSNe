import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ilsne.stacking import exact_convolution_stack, gaussian_precision_stack, gaussian_stack_lambda


def _gaussian_posterior(mu: float, sigma: float) -> dict:
    grid = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1001)
    post = np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
    post /= np.trapezoid(post, grid)
    return {"lam_grid": grid, "posterior": post, "lam_mean": mu, "lam_std": sigma}


def test_gaussian_stack_matches_precision_product():
    g1 = _gaussian_posterior(0.9, 0.05)
    g2 = _gaussian_posterior(1.1, 0.08)
    res = gaussian_stack_lambda(
        [
            {"lam_mean": g1["lam_mean"], "lam_std": g1["lam_std"]},
            {"lam_mean": g2["lam_mean"], "lam_std": g2["lam_std"]},
        ]
    )
    mu_true, var_true = gaussian_precision_stack(
        np.array([g1["lam_mean"], g2["lam_mean"]]),
        np.array([g1["lam_std"], g2["lam_std"]]) ** 2,
    )
    assert np.isclose(res["lam_mean"], mu_true, atol=1e-9)
    assert np.isclose(res["lam_std"], np.sqrt(var_true), atol=1e-9)


def test_exact_convolution_stack_recovers_gaussian_sum():
    g1 = _gaussian_posterior(0.9, 0.05)
    g2 = _gaussian_posterior(1.1, 0.08)
    out = exact_convolution_stack(
        [
            {"lam_grid": g1["lam_grid"], "posterior": g1["posterior"]},
            {"lam_grid": g2["lam_grid"], "posterior": g2["posterior"]},
        ]
    )
    mu_true = g1["lam_mean"] + g2["lam_mean"]
    sig_true = np.sqrt(g1["lam_std"] ** 2 + g2["lam_std"] ** 2)
    assert np.isclose(out["lam_tot_mean"], mu_true, atol=1e-3)
    assert np.isclose(out["lam_tot_std"], sig_true, atol=1e-3)
    assert np.isclose(out["lam_tot_map"], mu_true, atol=1e-3)
