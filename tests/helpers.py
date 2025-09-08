from __future__ import annotations

import numpy as np


class FakePantheonPlusData:
    def __init__(self, zCMB, m_obs, cov_mag_b):
        self.zCMB = np.asarray(zCMB, dtype=float)
        self.m_obs = np.asarray(m_obs, dtype=float)
        self.cov_mag_b = np.asarray(cov_mag_b, dtype=float)


def make_fake_pantheon(z, m, cov):
    """Return a constructor returning a fake PantheonPlusData object.

    Usage:
        monkeypatch.setattr(ilsne.hierarc_glue, "_PantheonPlusData", make_fake_pantheon(z, m, cov))
    """

    def _ctor():
        return FakePantheonPlusData(z, m, cov)

    return _ctor

