"""Residual diagnostic plots.

All functions return a ``matplotlib.figure.Figure``.  They never call
``plt.show()`` or ``plt.savefig()`` — the caller controls display/saving.

Public API
----------
plot_residuals_vs_fitted(result, lowess, figsize) -> Figure
plot_scale_location(result, lowess, figsize)      -> Figure
plot_qq(result, figsize)                          -> Figure
"""

from __future__ import annotations

import numpy as np
import scipy.stats
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from econtools.models._results import RegressionResult


def plot_residuals_vs_fitted(
    result: RegressionResult,
    lowess: bool = True,
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
) -> Figure:
    """Residuals vs Fitted values plot.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    lowess:
        Overlay a LOWESS smoother (default True).
    figsize:
        Figure size in inches.
    ax:
        Existing Axes to plot on.  If None, creates a new figure.

    Returns
    -------
    matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    fitted = result.fitted
    resid = result.resid

    ax.scatter(fitted, resid, alpha=0.5, s=20, color="steelblue")
    ax.axhline(0, color="black", lw=1, ls="--")

    if lowess:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
        smooth = sm_lowess(resid.values, fitted.values, frac=0.3)
        ax.plot(smooth[:, 0], smooth[:, 1], color="red", lw=2)

    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    return fig


def plot_scale_location(
    result: RegressionResult,
    lowess: bool = True,
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
) -> Figure:
    """Scale-Location plot: √|standardised residuals| vs Fitted.

    Useful for detecting heteroskedasticity — a flat red line indicates
    constant variance.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    lowess:
        Overlay a LOWESS smoother (default True).
    figsize:
        Figure size in inches.
    ax:
        Existing Axes to plot on.  If None, creates a new figure.

    Returns
    -------
    matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    fitted = result.fitted
    std_resid = result.resid / result.resid.std()
    sqrt_abs = np.sqrt(np.abs(std_resid))

    ax.scatter(fitted, sqrt_abs, alpha=0.5, s=20, color="steelblue")

    if lowess:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
        smooth = sm_lowess(sqrt_abs.values, fitted.values, frac=0.3)
        ax.plot(smooth[:, 0], smooth[:, 1], color="red", lw=2)

    ax.set_xlabel("Fitted values")
    ax.set_ylabel(r"$\sqrt{|\mathrm{Std.\,Resid.}|}$")
    ax.set_title("Scale-Location")

    return fig


def plot_qq(
    result: RegressionResult,
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
) -> Figure:
    """Normal Q-Q plot of residuals.

    Points lying on the diagonal reference line indicate normality.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    figsize:
        Figure size in inches.
    ax:
        Existing Axes to plot on.  If None, creates a new figure.

    Returns
    -------
    matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    scipy.stats.probplot(result.resid.values, plot=ax)
    ax.set_title("Normal Q-Q Plot")
    return fig
