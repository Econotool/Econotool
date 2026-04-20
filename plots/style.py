"""Publication-quality plot styling.

Provides a shared colorblind-friendly palette, matplotlib rcParams preset,
a context manager for scoped styling, and a post-hoc theme applicator.

Public API
----------
PALETTE         — dict of named colors
PALETTE_LIST    — ordered list for sequential use
PUB_RC          — matplotlib rcParams dict
pub_style()     — context manager wrapping plt.rc_context(PUB_RC)
apply_theme(fig) — post-hoc styling (remove spines, set fonts)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011, adapted)
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "primary": "#2C5F8A",
    "secondary": "#D14B4B",
    "accent1": "#E69F00",
    "accent2": "#56B4E9",
    "accent3": "#009E73",
    "accent4": "#F0E442",
    "accent5": "#CC79A7",
    "dark": "#333333",
    "gray": "#999999",
}

PALETTE_LIST: list[str] = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["accent1"],
    PALETTE["accent3"],
    PALETTE["accent2"],
    PALETTE["accent5"],
    PALETTE["accent4"],
]

# ---------------------------------------------------------------------------
# rcParams preset
# ---------------------------------------------------------------------------

PUB_RC: dict[str, object] = {
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    # Spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Lines
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    # Legend
    "legend.frameon": False,
    "legend.loc": "best",
    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    # Color cycle
    "axes.prop_cycle": plt.cycler("color", PALETTE_LIST),
}


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def pub_style() -> Generator[None, None, None]:
    """Context manager that temporarily applies publication rcParams.

    Usage::

        with pub_style():
            fig = plot_binscatter(df, "y", "x")
    """
    with plt.rc_context(PUB_RC):
        yield


# ---------------------------------------------------------------------------
# Post-hoc theme applicator
# ---------------------------------------------------------------------------


def apply_theme(fig: Figure) -> Figure:
    """Apply publication styling to an existing figure.

    Removes top/right spines, sets serif fonts on labels and titles,
    and calls ``tight_layout()``.

    Parameters
    ----------
    fig:
        Matplotlib Figure to style.

    Returns
    -------
    The same Figure, modified in-place.
    """
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("serif")
        if ax.get_xlabel():
            ax.xaxis.label.set_fontfamily("serif")
        if ax.get_ylabel():
            ax.yaxis.label.set_fontfamily("serif")
        title = ax.get_title()
        if title:
            ax.title.set_fontfamily("serif")
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig
