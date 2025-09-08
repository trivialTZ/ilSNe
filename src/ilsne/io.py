"""Input/output helpers for lens datasets.

Currently provides a convenience constructor to build a :class:`LensDataset`
from a tabular file or :class:`pandas.DataFrame` produced by surveys or
simulation pipelines.
"""

from __future__ import annotations

import pandas as pd

from .magnification import LensDataset, LensedSN


def lens_from_dataframe(name: str, df: pd.DataFrame) -> LensDataset:
    r"""Build a :class:`LensDataset` from a DataFrame.

    Parameters
    ----------
    name : str
        Lens identifier.
    df : pandas.DataFrame
        Must contain columns ``z``, ``mu_model``, ``m_obs``, ``sigma_m`` and
        optionally ``R_arcsec``.

    Returns
    -------
    LensDataset
        Dataset populated from ``df`` rows.
    """
    cols = {c.lower(): c for c in df.columns}

    def col(k: str) -> str | None:
        return cols[k] if k in cols else None

    # Validate required columns and provide a clear error if missing
    required = ["z", "mu_model", "m_obs", "sigma_m"]
    missing = [k for k in required if col(k) is None]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Present columns: {list(df.columns)}"
        )

    zs = df[col("z")].values
    mu = df[col("mu_model")].values
    m = df[col("m_obs")].values
    s = df[col("sigma_m")].values
    Rcol = col("r_arcsec")
    sne = []
    for i in range(len(df)):
        sne.append(
            LensedSN(
                z=float(zs[i]),
                mu_model=float(mu[i]),
                m_obs=float(m[i]),
                sigma_m=float(s[i]),
                R_arcsec=(float(df[Rcol].values[i]) if Rcol else None),
            )
        )
    return LensDataset(name=name, sne=sne)
