"""Declarative layout model for a composite multi-panel float."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompositeStyle:
    """Typography and geometry knobs for a stacked composite float.

    The defaults target a **formal** publication style — centred, bolded
    panel labels with generous vertical breathing room and a small right-
    edge pad so numeric content is not flush against the rule endpoint.
    Override individual fields for tighter (Liu-Tsyvinski) or looser
    (slides) conventions; switch ``panel_label_style`` to ``"plain"`` to
    drop the bold.
    """

    # --- panel-label row ------------------------------------------------
    panel_label_style: str = "centered_bold"
    """One of ``"bold"`` (flush-left bold), ``"plain"`` (flush-left
    roman), ``"centered"`` (centred roman), ``"centered_bold"`` (centred
    bold)."""

    # --- caption --------------------------------------------------------
    caption_gap: str = "0.8em"
    """Vertical space between ``\\caption{...}`` and ``\\toprule``.
    Larger values visually separate the title from the table frame."""

    # --- outer tabular geometry -----------------------------------------
    tabcolsep: str = "6pt"
    """Half of the inter-column gap inside the outer tabular."""
    arraystretch: float = 1.15
    """Row-height multiplier for the outer tabular."""
    font_size: str = "footnotesize"
    """LaTeX size command applied to the tabular block.  One of
    ``"scriptsize"``, ``"footnotesize"``, ``"small"``, ``"normalsize"``."""

    # --- edge padding ---------------------------------------------------
    left_pad: str = "4pt"
    """Hspace inserted at the left edge of the outer tabular so the
    leftmost column's content is not flush against the rule start."""
    right_pad: str = "6pt"
    """Hspace inserted at the right edge of the outer tabular so the
    rightmost column's right-aligned content is not flush against the
    rule end (the key fix for the 'rightmost column looks pushed out'
    percept)."""

    # --- between-panel spacing ------------------------------------------
    between_panel_above: str = "0.5em"
    """Extra ``\\addlinespace`` above the midrule that separates panels."""
    between_panel_below: str = "0.25em"
    """Extra ``\\addlinespace`` below the midrule that separates panels."""

    # --- notes paragraph ------------------------------------------------
    notes_gap: str = "0.5em"
    """Vertical space between the ``\\bottomrule`` and the notes block."""
    notes_font: str = "footnotesize"
    """Size command for the notes minipage."""

    # --- figure-panel geometry -----------------------------------------
    figure_max_height: float = 0.35
    """Cap on ``\\includegraphics`` height inside a :class:`FigurePanel`,
    expressed as a fraction of ``\\textheight``.  Keeps a figure panel
    from pushing other panels off the page."""

    # --- full-page geometry --------------------------------------------
    fill_page: bool = False
    """When ``True``, the float occupies a dedicated page: ``[p]``
    specifier plus ``\\vspace*{\\fill}`` above and below the content so
    the composite is vertically centred on its own page.  Useful when
    each float should claim one full page in a figure-driven report."""


_DEFAULT_STYLE = CompositeStyle()


@dataclass
class TablePanel:
    """A panel whose body is a table rendered as a bare tabular fragment."""

    panel_id: str
    label: str            # e.g. "Panel A. Descriptive statistics"
    tabular_src: str      # full ``\begin{tabular}...\end{tabular}`` block


@dataclass
class FigurePanel:
    """A panel whose body is a pre-rendered figure image."""

    panel_id: str
    label: str
    image_path: str       # relative path used inside \includegraphics


@dataclass
class Composite:
    """A composite float ready for LaTeX emission."""

    composite_id: str
    panels: list[TablePanel | FigurePanel]
    layout: tuple[int, int] | None = None   # (nrows, ncols); None => auto
    caption: str | None = None
    notes: str | None = None
    float_kind: str = "table"                # "table" or "figure"
    # ``column_fraction`` caps the natural width of a grid column as a
    # fraction of ``\textwidth``.  Defaults to an even split.
    column_fraction: float | None = None
    # Legacy shortcut for the most common override.  Kept for backward
    # compatibility; when ``style`` is provided it takes precedence.
    panel_label_style: str | None = None
    # Full style dataclass.  ``None`` -> use :data:`_DEFAULT_STYLE`.
    style: CompositeStyle | None = None

    def resolved_style(self) -> CompositeStyle:
        """Return the effective :class:`CompositeStyle`, merging the
        legacy ``panel_label_style`` shortcut into the full style."""
        base = self.style if self.style is not None else CompositeStyle()
        if self.panel_label_style is not None and self.style is None:
            # Legacy path — clone the default but swap in the shortcut.
            from dataclasses import replace
            base = replace(base, panel_label_style=self.panel_label_style)
        return base
