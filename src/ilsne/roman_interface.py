"""Roman interface (placeholder).

This module collects Roman-specific hooks, kept separate from the rest of the
code to limit Roman-only dependencies to one place. It provides a simple cadence
structure and helpers to map it into the Stage‑6 yield synthesis ``SurveyConfig``.

TODOs:
* Ingest HLTDS cadence & visit depth curves from official products
* Expose WFI bandpasses/zeropoints for photometry
* Map SLSim pixel-level configs (PSFs, noise) for validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .slsim_yields import SurveyConfig


@dataclass
class RomanCadence:
    """Simple cadence description for Roman HLTDS."""

    epochs: int
    days_between: float
    per_epoch_depth_5sig: dict[str, float]


def load_hltds_stub() -> RomanCadence:
    """Return a placeholder cadence for HLTDS.

    Returns
    -------
    RomanCadence
        Hard-coded number of epochs, spacing in days, and per-band 5σ depth.
    """

    return RomanCadence(
        epochs=30, days_between=2.5, per_epoch_depth_5sig={"Y": 26.5, "J": 26.3, "H": 26.1}
    )


def to_survey_config(cadence: RomanCadence) -> SurveyConfig:
    """Convert a Roman cadence into a Stage‑6 ``SurveyConfig``.

    Returns an instance of :class:`ilsne.slsim_yields.SurveyConfig` without importing
    ``slsim_yields`` at module import time to avoid circular imports. Safe to call even
    when SLSim is not installed.

    Parameters
    ----------
    cadence : RomanCadence
        Roman cadence description with epochs, spacing and per-band 5σ depths.

    Returns
    -------
    ilsne.slsim_yields.SurveyConfig
        A survey configuration suitable for ``synthesize_yields``.
    """
    from .slsim_yields import SurveyConfig  # local import to avoid cycles

    return SurveyConfig(
        name="roman-hltds",
        epochs=int(cadence.epochs),
        days_between=float(cadence.days_between),
        depths=dict(cadence.per_epoch_depth_5sig),
    )
