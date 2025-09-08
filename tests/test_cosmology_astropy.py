import numpy as np

from ilsne.cosmology_small import delta_mu_astropy, delta_mu_numerical


def test_delta_mu_astropy_matches_numerical():
    z = np.linspace(0.01, 2.0, 20)
    for wa in [-0.5, 0.0, 0.5]:
        astro = delta_mu_astropy(z, 0.1, wa=wa)
        num = delta_mu_numerical(z, 0.1, wa=wa)
        assert np.allclose(astro, num, atol=1e-3)


def test_vector_input_shape():
    z = np.array([0.1, 0.3, 0.7])
    res = delta_mu_astropy(z, 0.05)
    assert res.shape == z.shape
