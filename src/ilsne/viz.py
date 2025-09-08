"""Visualization helpers for λ posteriors and results summaries."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


def plot_lambda_posterior(
    res: dict[str, Any],
    title: str = "λ posterior",
    truth: float | list[float] | dict[str, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Plot a 1D :math:`\lambda` posterior.

    Parameters
    ----------
    res : dict
        Result dictionary from :func:`ilsne.magnification.infer_lambda_for_lens`.
    title : str, optional
        Plot title. Default is ``"λ posterior"``.
    truth : float | list[float] | dict[str, float], optional
        Optional ground-truth λ value(s) to overlay as vertical line(s).
        - If a single float, draws a dashed red line at that λ with label "truth".
        - If a list of floats, draws dashed lines for each with numeric labels.
        - If a dict mapping label→value, draws dashed lines with the given labels.

    Returns
    -------
    tuple
        ``(fig, ax)`` from :mod:`matplotlib`.
    """
    x, y = res["lam_grid"], res["posterior"]
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=120)
    ax.plot(x, y)
    ax.set_xlabel("λ")
    ax.set_ylabel("Posterior")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    # Optional truth overlays
    if truth is not None:
        # Normalize into iterable of (label, value)
        items: list[tuple[str, float]]
        if isinstance(truth, dict):
            items = [(str(k), float(v)) for k, v in truth.items()]
        elif isinstance(truth, (list, tuple)):  # noqa: UP038
            items = [(f"truth[{i}]", float(v)) for i, v in enumerate(truth)]
        else:
            items = [("truth", float(truth))]
        for lab, val in items:
            ax.axvline(val, color="crimson", linestyle="--", alpha=0.8, label=lab)
        # Expand x-limits slightly if truth lies near edges
        xlo, xhi = float(x[0]), float(x[-1])
        span = xhi - xlo
        truths = [val for _, val in items]
        if min(truths) <= xlo + 0.01 * span:
            xlo -= 0.02 * span
        if max(truths) >= xhi - 0.01 * span:
            xhi += 0.02 * span
        ax.set_xlim(xlo, xhi)
        # Avoid duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq: dict[str, Any] = {}
            for h, lab in zip(handles, labels):  # noqa: B905
                if lab not in uniq:
                    uniq[lab] = h
            ax.legend(list(uniq.values()), list(uniq.keys()), frameon=False, fontsize=9)
    return fig, ax
