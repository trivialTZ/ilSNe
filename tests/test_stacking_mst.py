import numpy as np

from ilsne.stacking import gaussian_precision_stack, product_stack_common_lambda


def test_gaussian_precision_stack_matches_grid_product():
    mus = np.array([1.02, 0.99, 1.01])
    sig = np.array([0.05, 0.06, 0.04])
    # build per-lens grid posteriors
    grid = np.linspace(0.8, 1.2, 2001)
    posts = []
    for m, s in zip(mus, sig, strict=True):
        pdf = np.exp(-0.5 * ((grid - m) / s) ** 2)
        pdf /= np.trapezoid(pdf, grid)
        posts.append({"lam_grid": grid, "posterior": pdf})
    prod = product_stack_common_lambda(posts)
    mu_g, v_g = gaussian_precision_stack(mus, sig**2)
    assert abs(prod["lam_mean"] - mu_g) < 5e-3
    assert abs(prod["lam_std"] - np.sqrt(v_g)) < 5e-3
