"""Full-page panelled table compositor.

Stacks multiple table objects (ResultsTable, SummaryTable, DiagnosticsTable)
or raw LaTeX strings under labeled panels within a single ``\\begin{table}``
float, producing a space-efficient one-page layout.

Public API
----------
PanelEntry     — (label, content) pair
PanelledPage   — compositor producing full LaTeX output
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from econtools.output.tables.pub_latex import (
    DiagnosticsTable,
    ResultsTable,
    SummaryTable,
    _font_cmd,
)

_TableType = Union[ResultsTable, SummaryTable, DiagnosticsTable]

# Roman numeral labels
_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


@dataclass
class PanelEntry:
    """A single panel within a PanelledPage.

    Parameters
    ----------
    label:
        Human-readable panel label (e.g. "Baseline Results").
    content:
        A table object with a ``to_tabular()`` method, or a raw LaTeX string.
    """

    label: str
    content: _TableType | str


class PanelledPage:
    """Compose multiple table panels into a single full-page table float.

    Parameters
    ----------
    panels:
        List of :class:`PanelEntry` objects.
    caption:
        Table caption.
    label:
        LaTeX ``\\label`` key.
    notes:
        List of note strings for the ``tablenotes`` block.
    add_star_note:
        Append a significance-level note.
    landscape:
        Wrap in ``\\begin{landscape}...\\end{landscape}``.
    font_size:
        Font size command (``"footnotesize"``, ``"small"``, etc.) or ``None``.
    fit_to_page:
        Wrap each tabular in ``\\resizebox{\\textwidth}{!}{}``.
    float_position:
        Float specifier (e.g. ``"p"`` for page float).
    panel_label_style:
        ``"alpha"`` for A/B/C, ``"roman"`` for I/II/III.
    panel_spacing:
        Vertical space between panels (LaTeX length string).
    """

    def __init__(
        self,
        panels: list[PanelEntry],
        *,
        caption: str | None = None,
        label: str | None = None,
        notes: list[str] | None = None,
        add_star_note: bool = True,
        landscape: bool = False,
        font_size: str | None = "footnotesize",
        fit_to_page: bool = True,
        float_position: str = "p",
        panel_label_style: str = "alpha",
        panel_spacing: str = "8pt",
    ) -> None:
        if not panels:
            raise ValueError("panels must be a non-empty list.")
        self.panels = panels
        self.caption = caption
        self.label = label
        self.notes = list(notes or [])
        if add_star_note:
            self.notes.append(
                r"$^{*}$ $p<0.10$,\ $^{**}$ $p<0.05$,\ $^{***}$ $p<0.01$."
            )
        self.landscape = landscape
        self.font_size = font_size
        self.fit_to_page = fit_to_page
        self.float_position = float_position
        self.panel_label_style = panel_label_style
        self.panel_spacing = panel_spacing

    def _panel_label(self, index: int) -> str:
        if self.panel_label_style == "roman":
            tag = _ROMAN[index] if index < len(_ROMAN) else str(index + 1)
        else:
            tag = chr(ord("A") + index)
        return tag

    def _tabular_content(self, entry: PanelEntry) -> str:
        if isinstance(entry.content, str):
            return entry.content
        return entry.content.to_tabular()

    def to_latex(self) -> str:
        """Return the complete LaTeX string for the panelled table."""
        lines: list[str] = []

        if self.landscape:
            lines.append(r"\begin{landscape}")

        fp = f"[{self.float_position}]" if self.float_position else ""
        lines.append(rf"\begin{{table}}{fp}")
        lines.append(r"\centering")

        if self.caption:
            lines.append(rf"\caption{{{self.caption}}}")
        if self.label:
            lines.append(rf"\label{{{self.label}}}")

        has_notes = bool(self.notes)
        if has_notes:
            lines.append(r"\begin{threeparttable}")

        font_open, font_close = _font_cmd(self.font_size)
        if font_open:
            lines.append(font_open)

        for i, entry in enumerate(self.panels):
            if i > 0:
                lines.append(rf"\vspace{{{self.panel_spacing}}}")

            tag = self._panel_label(i)
            lines.append(
                rf"\textit{{Panel {tag}: {entry.label}}}"
            )
            lines.append("")

            tabular = self._tabular_content(entry)

            if self.fit_to_page:
                lines.append(r"\resizebox{\textwidth}{!}{%")
                lines.append(tabular)
                lines.append(r"}")
            else:
                lines.append(tabular)

        if font_open:
            lines.append(font_close)

        if has_notes:
            lines.append(r"\begin{tablenotes}")
            lines.append(r"\footnotesize")
            for note in self.notes:
                lines.append(rf"\item {note}")
            lines.append(r"\end{tablenotes}")
            lines.append(r"\end{threeparttable}")

        lines.append(r"\end{table}")

        if self.landscape:
            lines.append(r"\end{landscape}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_latex()
