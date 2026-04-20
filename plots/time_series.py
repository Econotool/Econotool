"""Time-series plots: series plot and correlogram (ACF)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import scipy.stats
from statsmodels.tsa.stattools import acf as sm_acf


def plot_time_series(
    y: Sequence[float] | pd.Series,
    x: Sequence[float] | pd.Series | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 4),
    ax: Axes | None = None,
) -> Figure:
    """Plot a single time series.

    Parameters
    ----------
    y:
        Series values.
    x:
        Optional time index for the x-axis (e.g., year). If None, uses 1..n.
    title, xlabel, ylabel:
        Optional labels for the plot.
    figsize:
        Figure size in inches.
    ax:
        Existing Axes to plot on.  If None, creates a new figure.
    """
    y_vals = np.asarray(y, dtype=float)
    if x is None:
        x_vals = np.arange(1, len(y_vals) + 1, dtype=float)
    else:
        x_vals = np.asarray(x, dtype=float)
        if x_vals.shape[0] != y_vals.shape[0]:
            raise ValueError("x and y must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.plot(x_vals, y_vals, color="steelblue", lw=1.8)
    ax.set_xlabel(xlabel or "Time")
    ax.set_ylabel(ylabel or "Value")
    if title:
        ax.set_title(title)
    return fig


def plot_distribution(
    y: Iterable[float] | pd.Series,
    *,
    bins: int = 20,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    density: bool = True,
    figsize: tuple[float, float] = (6, 4),
    ax: Axes | None = None,
) -> Figure:
    """Plot a univariate distribution as a histogram.

    Parameters
    ----------
    y:
        Series values.
    bins:
        Number of histogram bins.
    title, xlabel, ylabel:
        Optional labels.
    density:
        If True, plot density (area = 1). If False, plot counts.
    figsize:
        Figure size in inches.
    ax:
        Existing Axes to plot on.  If None, creates a new figure.
    """
    vals = np.asarray(list(y), dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        raise ValueError("Series has no finite observations.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.hist(vals, bins=bins, density=density, color="steelblue", alpha=0.8, edgecolor="white")
    ax.set_xlabel(xlabel or "Value")
    ax.set_ylabel(ylabel or ("Density" if density else "Count"))
    if title:
        ax.set_title(title)
    return fig


def plot_series_with_trend(
    y: Iterable[float] | pd.Series,
    trend: Iterable[float] | pd.Series,
    x: Sequence[float] | pd.Series | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 4),
    ax: Axes | None = None,
) -> Figure:
    """Plot a series with an overlaid trend line."""
    y_vals = np.asarray(list(y), dtype=float)
    t_vals = np.asarray(list(trend), dtype=float)
    if y_vals.shape[0] != t_vals.shape[0]:
        raise ValueError("y and trend must have the same length.")

    if x is None:
        x_vals = np.arange(1, len(y_vals) + 1, dtype=float)
    else:
        x_vals = np.asarray(x, dtype=float)
        if x_vals.shape[0] != y_vals.shape[0]:
            raise ValueError("x and y must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.plot(x_vals, y_vals, color="steelblue", lw=1.6, label="Series")
    ax.plot(x_vals, t_vals, color="darkorange", lw=2.0, label="Trend")
    ax.set_xlabel(xlabel or "Time")
    ax.set_ylabel(ylabel or "Value")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    return fig


def plot_series_with_trends(
    y: Iterable[float] | pd.Series,
    trends: dict[str, Iterable[float] | pd.Series],
    x: Sequence[float] | pd.Series | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 4),
    ax: Axes | None = None,
) -> Figure:
    """Plot a series with multiple trend lines (dashed)."""
    y_vals = np.asarray(list(y), dtype=float)
    if x is None:
        x_vals = np.arange(1, len(y_vals) + 1, dtype=float)
    else:
        x_vals = np.asarray(x, dtype=float)
        if x_vals.shape[0] != y_vals.shape[0]:
            raise ValueError("x and y must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.plot(x_vals, y_vals, color="black", lw=1.8, label="Series")
    for name, trend in trends.items():
        t_vals = np.asarray(list(trend), dtype=float)
        if t_vals.shape[0] != y_vals.shape[0]:
            raise ValueError("trend length must match y length.")
        ax.plot(x_vals, t_vals, lw=1.6, ls="--", label=name)
    ax.set_xlabel(xlabel or "Time")
    ax.set_ylabel(ylabel or "Value")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    return fig


def plot_residual_diagnostics(
    resid: Iterable[float] | pd.Series,
    x: Sequence[float] | pd.Series | None = None,
    *,
    lags: int = 12,
    title_prefix: str = "Residuals",
) -> dict[str, Figure]:
    """Generate residual diagnostics: series plot, correlogram, distribution."""
    resid_vals = np.asarray(list(resid), dtype=float)
    figs: dict[str, Figure] = {}

    figs["series"] = plot_time_series(
        resid_vals,
        x=x,
        title=f"{title_prefix} Time Series",
        xlabel="Time",
        ylabel=title_prefix,
        figsize=(8, 3.5),
    )
    figs["correlogram"] = plot_correlogram(
        resid_vals,
        lags=lags,
        title=f"{title_prefix} Correlogram",
        figsize=(8, 3.5),
        ci_method="bartlett",
        ma_q=0,
        alpha=0.05,
    )
    figs["distribution"] = plot_distribution(
        resid_vals,
        bins=15,
        title=f"{title_prefix} Distribution",
        xlabel=title_prefix,
        ylabel="Density",
        density=True,
        figsize=(6, 4),
    )
    return figs


def plot_correlogram(
    y: Iterable[float] | pd.Series,
    *,
    lags: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 4),
    ci_method: str = "bartlett",
    ma_q: int = 0,
    alpha: float = 0.05,
    ax: Axes | None = None,
) -> Figure:
    """Plot sample autocorrelation function (correlogram).

    Parameters
    ----------
    y:
        Series values.
    lags:
        Number of lags to display. Defaults to min(40, nobs - 1).
    title:
        Optional plot title.
    figsize:
        Figure size in inches.
    ci_method:
        Confidence interval method: "bartlett" or "none".
        For "bartlett", uses Bartlett bands with optional MA(q) adjustment.
    ma_q:
        MA(q) order for Bartlett variance (q=0 gives white-noise bands).
    alpha:
        Significance level for bands (default 0.05 -> 95% CI).
    ax:
        Existing Axes to plot on.  If None, creates a new figure.
    """
    vals = np.asarray(list(y), dtype=float)
    n_obs = len(vals)
    if n_obs < 2:
        raise ValueError("Series must have at least 2 observations.")
    if lags is None:
        lags = min(40, n_obs - 1)
    if lags <= 0:
        raise ValueError("lags must be positive.")
    if ma_q < 0:
        raise ValueError("ma_q must be non-negative.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    acf_vals = sm_acf(vals, nlags=lags, fft=False)
    lag_idx = np.arange(1, lags + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.axhline(0.0, color="black", lw=1)
    ax.vlines(lag_idx, 0.0, acf_vals[1:], color="steelblue", lw=2)
    ax.plot(lag_idx, acf_vals[1:], "o", color="steelblue", ms=4)

    if ci_method.lower() == "bartlett":
        bands = _bartlett_bands(acf_vals, n_obs, lags, ma_q, alpha)
        ax.plot(lag_idx, bands, color="red", lw=1.5, ls="--")
        ax.plot(lag_idx, -bands, color="red", lw=1.5, ls="--")
        ax.fill_between(lag_idx, -bands, bands, color="red", alpha=0.08)
    elif ci_method.lower() != "none":
        raise ValueError("ci_method must be 'bartlett' or 'none'.")

    ax.set_xlim(0.5, lags + 0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    if title:
        ax.set_title(title)
    return fig


def _bartlett_bands(
    acf_vals: np.ndarray,
    n_obs: int,
    lags: int,
    ma_q: int,
    alpha: float,
) -> np.ndarray:
    """Compute Bartlett confidence bands for ACF at each lag."""
    z = float(scipy.stats.norm.ppf(1.0 - alpha / 2.0))
    bands = np.empty(lags, dtype=float)
    for k in range(1, lags + 1):
        if ma_q > 0:
            max_j = min(ma_q, k - 1)
        else:
            max_j = k - 1
        if max_j <= 0:
            var = 1.0 / n_obs
        else:
            rho_sq_sum = float(np.sum(acf_vals[1 : max_j + 1] ** 2))
            var = (1.0 + 2.0 * rho_sq_sum) / n_obs
        bands[k - 1] = z * np.sqrt(var)
    return bands
