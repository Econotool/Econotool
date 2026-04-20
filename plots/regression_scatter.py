"""Scatter plot with fitted regression line and inset statistics box.

Used for added-variable plots, first-stage / reduced-form figures,
and any paper figure that pairs a scatter with a fitted regression.

Public API
----------
scatter_with_fit(ax, df, x, y, ...) -> Estimate
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from econtools._core.types import Estimate
from econtools.fit import fit_model
from econtools.model.spec import ModelSpec
from econtools.plots.style import PALETTE


def scatter_with_fit(
    ax: Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    fit: Estimate | None = None,
    weight_col: str | None = None,
    cov_type: str = "HC1",
    annotate: bool = True,
    annotate_loc: str = "lower right",
    scatter_color: str | None = None,
    line_color: str | None = None,
    scatter_alpha: float = 0.55,
    scatter_size: float = 10.0,
    line_width: float = 1.4,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    title_fontsize: float = 9.5,
) -> Estimate:
    """Draw a scatter of ``y`` vs ``x`` on ``ax`` with the fitted OLS/WLS line.

    Parameters
    ----------
    ax:
        Target Axes.
    df:
        Source DataFrame.
    x, y:
        Column names.
    fit:
        Pre-fit ``Estimate``. If ``None``, fits OLS (or WLS when
        ``weight_col`` is set) with the given ``cov_type`` internally.
    weight_col:
        Name of weights column. Triggers WLS when ``fit`` is not provided.
    cov_type:
        Covariance type label for the internal fit (unused when ``fit``
        is supplied).
    annotate:
        If True, place an inset box showing ``coef / robust SE / t``.
    annotate_loc:
        One of ``"lower right"``, ``"lower left"``, ``"upper right"``,
        ``"upper left"``.
    scatter_color, line_color:
        Override the default palette colours.
    xlabel, ylabel, title:
        Axis / title text. If None, uses column names / no title.

    Returns
    -------
    The ``Estimate`` that generated the line (either ``fit`` or the
    internal one).
    """
    scatter_color = scatter_color or PALETTE["primary"]
    line_color = line_color or PALETTE["accent3"]

    if fit is None:
        estimator = "wls" if weight_col else "ols"
        spec = ModelSpec(
            dep_var=y,
            exog_vars=[x],
            estimator=estimator,
            weights_col=weight_col,
            cov_type=cov_type,
        )
        fit = fit_model(spec, df)

    slope = float(fit.params[x])
    intercept = float(fit.params.get("const", 0.0))
    se = float(fit.bse[x])
    t = float(fit.tvalues[x])

    ax.scatter(
        df[x], df[y],
        s=scatter_size, alpha=scatter_alpha,
        color=scatter_color, edgecolor="none",
    )
    xs = np.linspace(df[x].min(), df[x].max(), 100)
    ax.plot(xs, intercept + slope * xs, color=line_color, lw=line_width)

    ax.set_xlabel(xlabel or x, fontsize=9)
    ax.set_ylabel(ylabel or y, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.3, color="#d0d0d0")
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=6, loc="center")

    if annotate:
        loc_map: dict[str, tuple[float, float, str, str]] = {
            "lower right": (0.97, 0.06, "right", "bottom"),
            "lower left":  (0.03, 0.06, "left",  "bottom"),
            "upper right": (0.97, 0.94, "right", "top"),
            "upper left":  (0.03, 0.94, "left",  "top"),
        }
        xa, ya, ha, va = loc_map.get(annotate_loc, loc_map["lower right"])
        ax.annotate(
            f"coef = {slope:.2f}, robust SE = {se:.2f}, $t$ = {t:.2f}",
            xy=(xa, ya), xycoords="axes fraction",
            ha=ha, va=va, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#888888", lw=0.6),
        )

    return fit


__all__ = ["scatter_with_fit"]
