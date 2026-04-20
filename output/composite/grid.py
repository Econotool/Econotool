"""Grid composite: rigid ``p x q`` tiling inside one outer tabular.

Panel bodies are kept intact (figures as ``\\includegraphics``, tables as
their native bare ``tabular``), but the outer geometry is owned by a
top-level tabular whose column spec fixes cell widths and whose
``\\midrule`` / ``\\bottomrule`` fix the row boundaries.

To keep rows visually flush, panel bodies that are tabulars have their
internal ``\\toprule`` and ``\\bottomrule`` suppressed before being
placed into a cell — the outer tabular's rules carry the framing.
"""

from __future__ import annotations

import re

from econtools.output.composite.model import Composite, FigurePanel, TablePanel


_RULE_TO_STRIP = re.compile(r"\\toprule|\\bottomrule|\\hline\\hline|\\hline")


def _strip_outer_rules(tabular_src: str) -> str:
    """Remove the single ``\\toprule`` / ``\\bottomrule`` framing the body.

    Only the first ``\\toprule`` and the last ``\\bottomrule`` are
    stripped — any interior ``\\midrule`` is preserved so the panel's
    own header/body separation stays visible.
    """
    m = re.search(r"\\toprule|\\hline\\hline", tabular_src)
    if m is None:
        m = re.search(r"\\hline", tabular_src)
    if m is not None:
        tabular_src = tabular_src[:m.start()] + tabular_src[m.end():]

    # Find last bottom rule by scanning from the right.
    for token in (r"\bottomrule", r"\hline\hline", r"\hline"):
        idx = tabular_src.rfind(token)
        if idx >= 0:
            tabular_src = tabular_src[:idx] + tabular_src[idx + len(token):]
            break
    return tabular_src


def _cell_body(panel: TablePanel | FigurePanel, body_h: float) -> str:
    if isinstance(panel, TablePanel):
        stripped = _strip_outer_rules(panel.tabular_src)
        # ``max width`` only: shrink-to-fit horizontally if the tabular
        # exceeds the cell width, but do not reserve a fixed height —
        # doing so caused adjustbox to centre short tabulars vertically
        # inside the box, breaking row-top alignment.
        return (
            r"\adjustbox{max width=\linewidth}{%" + "\n" + stripped + "\n}"
        )
    # FigurePanel — cap the figure height so a large matplotlib export
    # cannot overshoot the float page; keepaspectratio leaves it
    # naturally sized when the cap isn't binding.
    return (
        rf"\includegraphics[width=\linewidth,"
        rf"height={body_h}\textheight,keepaspectratio]{{{panel.image_path}}}"
    )


def render_grid(
    composite: Composite,
    *,
    nrows: int,
    ncols: int,
) -> str:
    """Render *composite* as a rigid ``nrows x ncols`` grid."""
    n = len(composite.panels)
    if nrows * ncols < n:
        raise ValueError(
            f"layout {nrows}x{ncols} cannot hold {n} panels"
        )

    gutter = 0.02  # fraction of \textwidth between columns
    col_width = round((1.0 - gutter * (ncols - 1)) / ncols, 4)
    if composite.column_fraction is not None:
        col_width = composite.column_fraction

    # Cap per-cell body height so a large figure cannot overshoot the
    # float page.  The row itself will size to the tallest cell.
    body_h_cap = round(0.80 / nrows, 3)

    col_spec = "@{}" + f"@{{\\hspace{{{gutter}\\textwidth}}}}".join(["c"] * ncols) + "@{}"

    lines: list[str] = [r"\begin{figure}[htbp]", r"\centering"]
    lines.append(
        r"{\footnotesize\renewcommand{\arraystretch}{1.0}%"
    )
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # ---- Header row: panel labels, one per column, aligned horizontally.
    # We split the header and body into *separate* tabular rows so the
    # panel labels across a row land at the same y-coordinate (they are
    # literally one tabular row).

    def _cell_label(i: int) -> str:
        if i >= n:
            return ""
        return rf"\textbf{{{composite.panels[i].label}}}"

    def _cell_minipage(i: int) -> str:
        if i >= n:
            return r"\strut"
        panel = composite.panels[i]
        inner = _cell_body(panel, body_h_cap)
        return (
            rf"\begin{{minipage}}[t]{{{col_width}\textwidth}}\centering"
            "\n" + inner + "\n"
            r"\end{minipage}"
        )

    for row_idx in range(nrows):
        if row_idx > 0:
            lines.append(r"\midrule")

        # Label row: plain tabular cells (no minipage), centered.  Each
        # cell sits in the outer tabular row so labels across the row
        # share a single baseline.
        label_cells = [_cell_label(row_idx * ncols + c) for c in range(ncols)]
        lines.append(" & ".join(label_cells) + r" \\")
        lines.append(r"\addlinespace[0.2em]")

        # Body row: one minipage per cell.  [t] alignment means tops of
        # minipages share the row's top edge.
        body_cells = [_cell_minipage(row_idx * ncols + c) for c in range(ncols)]
        lines.append(" & ".join(body_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    if composite.caption:
        lines.append(rf"\caption{{{composite.caption}}}")
    lines.append(rf"\label{{fig:{composite.composite_id}}}")
    if composite.notes:
        lines.append(r"\begin{minipage}{\linewidth}\footnotesize")
        lines.append(r"\vspace{0.3em}")
        lines.append(rf"\textit{{Notes:}} {composite.notes}")
        lines.append(r"\end{minipage}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)
