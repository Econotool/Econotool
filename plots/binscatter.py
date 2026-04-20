"""Binscatter plot — the workhorse of applied microeconometrics.

Bins the x-variable into quantile bins, computes the conditional mean
of y within each bin, and plots the result.  Optionally overlays a
linear or polynomial fit through the bin means (or the raw data).

Public API
----------
plot_binscatter(df, y, x, ...) -> matplotlib.Figure
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_binscatter(
    df: pd.DataFrame,
    y: str,
    x: str,
    *,
    n_bins: int = 20,
    controls: Optional[Sequence[str]] = None,
    absorb: Optional[Sequence[str]] = None,
    fit_line: bool = True,
    poly_order: int = 1,
    ci: bool = False,
    weights: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    scatter_kw: Optional[dict] = None,
    line_kw: Optional[dict] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Create a binscatter plot.

    Bins ``x`` into ``n_bins`` equal-sized quantile bins and plots the
    mean of ``y`` within each bin.  If ``controls`` are specified, both
    ``y`` and ``x`` are first residualised on the controls (Frisch-Waugh).

    Parameters
    ----------
    df:
        Source DataFrame.
    y:
        Name of the dependent (y-axis) variable.
    x:
        Name of the independent (x-axis) variable.
    n_bins:
        Number of quantile bins (default 20).
    controls:
        List of control variable names.  If given, the plot shows the
        partial relationship between *y* and *x* after partialling out
        the controls (added back to means for interpretability).
    absorb:
        Categorical variables to absorb as fixed effects (demeaned out)
        before residualising.  Applied before ``controls``.
    fit_line:
        Overlay a regression fit line (default True).
    poly_order:
        Polynomial order for the fit line (default 1 = linear).
    ci:
        Show 95% confidence bands around the fit line (default False).
    weights:
        Column name for analytic weights (applied to bin means).
    ax:
        Existing matplotlib Axes to plot on.  If None, creates a new figure.
    scatter_kw:
        Additional kwargs passed to ``ax.scatter()``.
    line_kw:
        Additional kwargs passed to ``ax.plot()`` for the fit line.
    title, xlabel, ylabel:
        Axis labels and title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = df[[y, x]].copy()
    if controls:
        for c in controls:
            data[c] = df[c]
    if absorb:
        for a in absorb:
            data[a] = df[a]
    if weights:
        data["_w"] = df[weights]

    data = data.dropna()

    y_vals = data[y].values.astype(float)
    x_vals = data[x].values.astype(float)

    # --- Absorb fixed effects ---
    if absorb:
        for a in absorb:
            groups = data[a]
            for arr_name, arr in [("y", y_vals), ("x", x_vals)]:
                group_means = pd.Series(arr).groupby(groups.values).transform("mean")
                if arr_name == "y":
                    y_vals = arr - group_means.values + np.mean(arr)
                else:
                    x_vals = arr - group_means.values + np.mean(arr)
            if controls:
                for c in controls:
                    c_vals = data[c].values.astype(float)
                    group_means = pd.Series(c_vals).groupby(groups.values).transform("mean")
                    data[c] = c_vals - group_means.values + np.mean(c_vals)

    # --- Partial out controls via OLS ---
    if controls:
        from numpy.linalg import lstsq

        C = data[list(controls)].values.astype(float)
        C = np.column_stack([np.ones(len(C)), C])

        # Residualise y
        beta_y, _, _, _ = lstsq(C, y_vals, rcond=None)
        y_resid = y_vals - C @ beta_y
        y_vals = y_resid + np.mean(y_vals)  # add back mean for scale

        # Residualise x
        beta_x, _, _, _ = lstsq(C, x_vals, rcond=None)
        x_resid = x_vals - C @ beta_x
        x_vals = x_resid + np.mean(x_vals)

    # --- Bin ---
    n_bins_actual = min(n_bins, len(np.unique(x_vals)))
    try:
        bin_edges = np.percentile(
            x_vals, np.linspace(0, 100, n_bins_actual + 1)
        )
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        n_bins_actual = len(bin_edges) - 1
        bin_idx = np.digitize(x_vals, bin_edges[1:-1])
    except Exception:
        bin_idx = np.zeros(len(x_vals), dtype=int)
        n_bins_actual = 1

    bin_x_means = np.array(
        [np.mean(x_vals[bin_idx == b]) for b in range(n_bins_actual)]
    )
    bin_y_means = np.array(
        [np.mean(y_vals[bin_idx == b]) for b in range(n_bins_actual)]
    )

    # Filter out any NaN bins
    valid = ~(np.isnan(bin_x_means) | np.isnan(bin_y_means))
    bin_x_means = bin_x_means[valid]
    bin_y_means = bin_y_means[valid]

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    skw = {"s": 50, "color": "#2C5F8A", "edgecolors": "white", "zorder": 5}
    if scatter_kw:
        skw.update(scatter_kw)
    ax.scatter(bin_x_means, bin_y_means, **skw)

    # --- Fit line ---
    if fit_line and len(bin_x_means) > poly_order:
        # Fit on raw data for correct inference, plot smooth line
        coeffs = np.polyfit(x_vals, y_vals, poly_order)
        x_grid = np.linspace(bin_x_means.min(), bin_x_means.max(), 200)
        y_hat = np.polyval(coeffs, x_grid)

        lkw = {"color": "#D14B4B", "linewidth": 2, "zorder": 4}
        if line_kw:
            lkw.update(line_kw)
        ax.plot(x_grid, y_hat, **lkw)

        if ci:
            # Bootstrap CI for the fit line
            rng = np.random.default_rng(42)
            n_boot = 200
            boot_curves = np.zeros((n_boot, len(x_grid)))
            for b in range(n_boot):
                idx = rng.integers(0, len(x_vals), len(x_vals))
                bc = np.polyfit(x_vals[idx], y_vals[idx], poly_order)
                boot_curves[b] = np.polyval(bc, x_grid)
            lo = np.percentile(boot_curves, 2.5, axis=0)
            hi = np.percentile(boot_curves, 97.5, axis=0)
            ax.fill_between(x_grid, lo, hi, alpha=0.15, color=lkw["color"])

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig
