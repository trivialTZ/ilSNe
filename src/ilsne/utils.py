from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np


def as_array(x: Union[float, Sequence[float], np.ndarray], n: int) -> np.ndarray:
    """Return ``x`` as a float array of length ``n``.

    - If ``x`` is a scalar, broadcast to length ``n``.
    - If ``x`` is a sequence/array, cast to ``float`` and validate length ``n``.
    """
    if isinstance(x, (float, int)):
        return np.full(int(n), float(x), dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("as_array expects a 1D sequence or a scalar")
    if len(arr) != int(n):
        raise ValueError(f"as_array length mismatch: expected {n}, got {len(arr)}")
    return arr


__all__ = ["as_array"]

