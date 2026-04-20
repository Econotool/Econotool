"""Figure-returning plot functions — Phase 1+."""

from econtools.plots.residual_plots import (
    plot_residuals_vs_fitted,
    plot_scale_location,
    plot_qq,
)
from econtools.plots.coefficient_plots import plot_coef_forest
from econtools.plots.time_series import (
    plot_time_series,
    plot_correlogram,
    plot_distribution,
    plot_series_with_trend,
    plot_residual_diagnostics,
    plot_series_with_trends,
)
from econtools.plots.binscatter import plot_binscatter
from econtools.plots.panel_figure import panel_figure
from econtools.plots.style import apply_theme, pub_style, PALETTE, PALETTE_LIST

__all__ = [
    "plot_residuals_vs_fitted",
    "plot_scale_location",
    "plot_qq",
    "plot_coef_forest",
    "plot_time_series",
    "plot_correlogram",
    "plot_distribution",
    "plot_series_with_trend",
    "plot_residual_diagnostics",
    "plot_series_with_trends",
    "plot_binscatter",
    "panel_figure",
    "apply_theme",
    "pub_style",
    "PALETTE",
    "PALETTE_LIST",
]
