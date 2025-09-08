"""Tiny in-memory cache for pivot parameters.

This helper avoids recomputing Pantheon+ pivot intercepts when scanning
multiple options within a single process. It is intentionally ephemeral; the
API also allows loading initial values from a JSON file for convenience.
"""

from __future__ import annotations

import json
import os

_CACHE: dict[str, tuple[float, float]] = {}


def get_cached_mp(z_anchor: float) -> tuple[float, float] | None:
    """Return cached ``(m_p_mean, m_p_sigma)`` for a given anchor redshift.

    Parameters
    ----------
    z_anchor : float
        Pivot redshift used to anchor the SN magnitude.

    Returns
    -------
    tuple | None
        Cached ``(mean, sigma)`` in magnitudes if present, else ``None``.
    """

    key = f"{z_anchor:.5f}"
    return _CACHE.get(key)


def put_cached_mp(z_anchor: float, mean: float, sigma: float) -> None:
    """Insert a cache entry for ``z_anchor``.

    All values are stored as floats.
    """

    _CACHE[f"{z_anchor:.5f}"] = (float(mean), float(sigma))


def load_cache_json(path: str) -> None:
    """Seed the in-memory cache from a JSON file if it exists.

    The file is expected to contain a mapping from string keys (formatted
    redshifts like ``"0.10000"``) to ``[mean, sigma]`` lists in magnitudes.
    Missing files are silently ignored.
    """

    if os.path.isfile(path):
        _CACHE.update({k: tuple(v) for k, v in json.load(open(path)).items()})
