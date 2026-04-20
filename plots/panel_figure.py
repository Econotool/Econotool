"""Multi-panel figure compositor for publication-quality output.

Composes multiple plot functions into a single figure with labeled panels,
suitable for journal submissions where page space is constrained.

Public API
----------
panel_figure(panels, ...) -> Figure
"""

from __future__ import annotations

import math
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from econtools.plots.style import apply_theme


def panel_figure(
    panels: list[Callable | dict],
    *,
    layout: tuple[int, int] | None = None,
    figsize: tuple[float, float] | None = None,
    panel_labels: str = "lower_alpha",
    label_fontsize: int = 12,
    label_fontweight: str = "bold",
    shared_x: bool = False,
    shared_y: bool = False,
    wspace: float | None = None,
    hspace: float | None = None,
    suptitle: str | None = None,
    group_frame: bool = False,
    frame_linewidth: float = 0.6,
    frame_color: str = "#444444",
) -> Figure:
    """Compose multiple plots into a single multi-panel figure.

    Parameters
    ----------
    panels:
        List of panel specifications.  Each element is either:
        - A callable ``fn(ax)`` that draws on the provided Axes, or
        - A dict ``{"func": fn, "kwargs": {...}}`` where ``fn(ax, **kwargs)``
          is called.
    layout:
        Grid layout as ``(nrows, ncols)``.  If None, auto-computed to be
        roughly square.
    figsize:
        Figure size in inches.  If None, auto-computed from layout.
    panel_labels:
        Label style: ``"lower_alpha"`` for ``(a), (b), ...``,
        ``"upper_alpha"`` for ``(A), (B), ...``,
        ``"panel"`` for ``Panel A, Panel B, ...``,
        ``"none"`` to suppress labels.
    label_fontsize:
        Font size for panel labels.
    label_fontweight:
        Font weight for panel labels.
    shared_x:
        Share x-axis across columns.
    shared_y:
        Share y-axis across rows.
    wspace:
        Horizontal space between subplots (passed to ``subplots_adjust``).
    hspace:
        Vertical space between subplots (passed to ``subplots_adjust``).
    suptitle:
        Optional super-title for the entire figure.

    Returns
    -------
    matplotlib Figure with all panels rendered.
    """
    n = len(panels)
    if n == 0:
        raise ValueError("panels must be a non-empty list.")

    # Auto layout
    if layout is None:
        ncols = min(n, 2) if n > 1 else 1
        nrows = math.ceil(n / ncols)
    else:
        nrows, ncols = layout
        if nrows * ncols < n:
            raise ValueError(
                f"layout ({nrows}x{ncols}={nrows * ncols}) is too small "
                f"for {n} panels."
            )

    if figsize is None:
        figsize = (6.5 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=shared_x,
        sharey=shared_y,
        squeeze=False,
    )

    # Flatten axes for iteration
    flat_axes = axes.flatten()

    # Label generator
    labels = _make_labels(panel_labels, n)

    for i, panel_spec in enumerate(panels):
        ax = flat_axes[i]

        if callable(panel_spec):
            panel_spec(ax)
        elif isinstance(panel_spec, dict):
            func = panel_spec["func"]
            kwargs = panel_spec.get("kwargs", {})
            func(ax, **kwargs)
        else:
            raise TypeError(
                f"Panel {i} must be a callable or dict, got {type(panel_spec)}"
            )

        # Add panel label
        if labels is not None:
            ax.text(
                -0.1, 1.05, labels[i],
                transform=ax.transAxes,
                fontsize=label_fontsize,
                fontweight=label_fontweight,
                va="bottom",
                ha="right",
            )

    # Hide unused axes
    for j in range(n, len(flat_axes)):
        flat_axes[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=label_fontsize + 2, fontweight="bold")

    apply_theme(fig)

    # Apply spacing AFTER apply_theme — apply_theme calls tight_layout()
    # which would otherwise clobber these values.
    kwargs_adjust: dict[str, float] = {}
    if wspace is not None:
        kwargs_adjust["wspace"] = wspace
    if hspace is not None:
        kwargs_adjust["hspace"] = hspace
    if kwargs_adjust:
        fig.subplots_adjust(**kwargs_adjust)

    if group_frame:
        # Draw a thin box around the composite area so the panels read as a
        # single grouped unit.  Uses figure coordinates with a small inset
        # so the frame does not overlap the outermost tick labels.
        from matplotlib.patches import Rectangle
        pad = 0.005
        frame = Rectangle(
            (pad, pad), 1.0 - 2 * pad, 1.0 - 2 * pad,
            fill=False, linewidth=frame_linewidth, edgecolor=frame_color,
            transform=fig.transFigure, figure=fig, zorder=100,
        )
        fig.patches.append(frame)

    return fig


def _make_labels(style: str, n: int) -> list[str] | None:
    """Generate panel labels based on style."""
    if style == "none":
        return None
    if style == "lower_alpha":
        return [f"({chr(ord('a') + i)})" for i in range(n)]
    if style == "upper_alpha":
        return [f"({chr(ord('A') + i)})" for i in range(n)]
    if style == "panel":
        return [f"Panel {chr(ord('A') + i)}" for i in range(n)]
    raise ValueError(
        f"Unknown panel_labels style '{style}'. "
        "Use 'lower_alpha', 'upper_alpha', 'panel', or 'none'."
    )
