"""Coefficient forest plot.

Public API
----------
plot_coef_forest(result, alpha, exclude_const, figsize) -> Figure
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from econtools.models._results import RegressionResult

_CONST_NAMES: frozenset[str] = frozenset({"const", "Intercept"})


def plot_coef_forest(
    result: RegressionResult,
    alpha: float = 0.05,
    exclude_const: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Coefficient forest plot with confidence-interval error bars.

    Each regressor is shown as a point estimate with horizontal CI bars.
    A dashed vertical line marks zero.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    alpha:
        Significance level for confidence intervals (default 0.05 → 95% CI).
    exclude_const:
        Omit the constant/intercept term (default True).
    figsize:
        Figure size in inches.  If None, auto-sized by number of regressors.
    ax:
        Existing Axes to plot on.  If None, creates a new figure.

    Returns
    -------
    matplotlib Figure.
    """
    from econtools.inference.hypothesis import conf_int as compute_ci

    ci_df = compute_ci(result, alpha=alpha)
    params = result.params.copy()
    lower = ci_df["lower"]
    upper = ci_df["upper"]

    if exclude_const:
        mask = ~params.index.isin(_CONST_NAMES)
        params = params[mask]
        lower = lower[mask]
        upper = upper[mask]

    n = len(params)
    if figsize is None:
        figsize = (8, max(4, n * 0.6 + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_pos = np.arange(n)
    xerr_lower = (params.values - lower.values).clip(min=0)
    xerr_upper = (upper.values - params.values).clip(min=0)

    ax.errorbar(
        params.values,
        y_pos,
        xerr=[xerr_lower, xerr_upper],
        fmt="o",
        color="steelblue",
        capsize=4,
        lw=1.5,
        ms=6,
    )
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params.index.tolist())
    ax.set_xlabel("Coefficient estimate")
    ax.set_title(
        f"Coefficient Forest Plot ({int((1 - alpha) * 100)}% CI, "
        f"{result.cov_type} SEs)"
    )

    fig.tight_layout()
    return fig
