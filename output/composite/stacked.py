"""Stacked composite: one outer tabular, all panels share a column grid.

This is the canonical "parent table with section-break panels" form.

**Layout algorithm.**  Each panel is parsed into ``PanelParts`` (header /
body / footer rows + per-column alignment).  The outer tabular's column
count is the least-common-multiple of the panels' native column counts,
so every panel divides the outer grid evenly.  A panel with
``ncols_native = n_p`` and ``ncols_outer = N`` emits each native cell as
``\\multicolumn{N/n_p}{<alignment>}{cell}``, so narrower panels produce
wider outer cells.

**Rule discipline.**  The outer tabular carries exactly one ``\\toprule``,
a ``\\midrule`` between every adjacent pair of section elements (panel
label / header block / body), and one ``\\bottomrule``.  Panel bodies
contribute **zero** internal booktabs rules — all horizontal demarcation
comes from the outer tabular.  This is what keeps the outer contour
strictly rectangular.
"""

from __future__ import annotations

from math import gcd
from functools import reduce

from econtools.output.composite.model import (
    Composite,
    CompositeStyle,
    FigurePanel,
    TablePanel,
)
from econtools.output.composite.parse import PanelParts, parse_tabular, split_row_cells


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _lcm_many(values: list[int]) -> int:
    return reduce(_lcm, values, 1)


def _expand_row(row: str, src_specs: list[str], scale: int) -> str:
    """Rewrite a tabular row to span ``scale`` outer columns per native cell.

    ``row`` is expected to contain exactly ``len(src_specs)`` ``&``-
    separated cells (or fewer if it already uses ``\\multicolumn``).  If
    *scale* is 1 the row is returned unchanged.  If the row already
    contains a ``\\multicolumn{K}{...}{...}`` directive we multiply ``K``
    by *scale* in-place; otherwise every plain cell is wrapped in a
    fresh ``\\multicolumn{scale}{<alignment>}{cell}``.
    """
    if scale == 1:
        return row
    cells = split_row_cells(row)
    if not cells:
        return row
    out_cells: list[str] = []
    for idx, cell in enumerate(cells):
        align = src_specs[idx] if idx < len(src_specs) else "c"
        stripped = cell.strip()
        if stripped.startswith(r"\multicolumn"):
            # Scale the existing span.  Parse ``\multicolumn{N}{A}{C}``.
            parsed = _parse_multicolumn(stripped)
            if parsed is None:
                # Fall back: wrap the whole thing as-is.
                out_cells.append(rf"\multicolumn{{{scale}}}{{{align}}}{{{stripped}}}")
                continue
            span, mc_align, mc_body = parsed
            out_cells.append(rf"\multicolumn{{{span * scale}}}{{{mc_align}}}{{{mc_body}}}")
        else:
            out_cells.append(rf"\multicolumn{{{scale}}}{{{align}}}{{{stripped}}}")
    return " & ".join(out_cells)


def _parse_multicolumn(src: str) -> tuple[int, str, str] | None:
    """Parse a leading ``\\multicolumn{N}{A}{C}`` from *src*.

    Returns ``(N, A, C)`` or ``None`` if *src* does not start with a
    well-formed multicolumn.
    """
    prefix = r"\multicolumn"
    if not src.startswith(prefix):
        return None
    i = len(prefix)
    args: list[str] = []
    for _ in range(3):
        # Skip whitespace
        while i < len(src) and src[i] == " ":
            i += 1
        if i >= len(src) or src[i] != "{":
            return None
        # Match balanced braces
        depth = 0
        j = i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= len(src):
            return None
        args.append(src[i + 1: j])
        i = j + 1
    try:
        span = int(args[0])
    except ValueError:
        return None
    return span, args[1], args[2]


def _panel_label_row(label: str, ncols_outer: int, *, style: str = "bold") -> str:
    """A single span-all row rendering the panel label.

    Supported styles:

    * ``"bold"``          — ``\\textbf{...}`` flush-left;
    * ``"plain"``         — flush-left roman text;
    * ``"centered"``      — centred roman text across the full grid;
    * ``"centered_bold"`` — centred bold (the formal default).
    """
    if style == "bold":
        return rf"\multicolumn{{{ncols_outer}}}{{@{{}}l}}{{\textbf{{{label}}}}}"
    if style == "centered":
        return rf"\multicolumn{{{ncols_outer}}}{{c}}{{{label}}}"
    if style == "centered_bold":
        return rf"\multicolumn{{{ncols_outer}}}{{c}}{{\textbf{{{label}}}}}"
    return rf"\multicolumn{{{ncols_outer}}}{{@{{}}l}}{{{label}}}"


def _emit_panel_rows(
    parts: PanelParts,
    label: str,
    ncols_outer: int,
    *,
    label_style: str = "bold",
    label_gap_below: str = "0.15em",
) -> list[str]:
    """Emit the LaTeX rows for one panel inside the outer tabular.

    Output shape:

    .. code-block:: text

        <panel-label row>
        \\cmidrule(l{2pt}r{2pt}){1-N}
        <header rows, expanded>
        \\midrule
        <body rows, expanded>
        <footer rows, expanded — may be empty>
    """
    scale = ncols_outer // parts.ncols

    lines: list[str] = []
    lines.append(_panel_label_row(label, ncols_outer, style=label_style) + r" \\")
    if label_gap_below and label_gap_below != "0pt":
        lines.append(rf"\addlinespace[{label_gap_below}]")
    # Full-width cmidrule (no ``(l{..}r{..})`` trim) — a trimmed rule
    # would shorten the rule by a couple of points on each side,
    # breaking the "one rectangular outer contour" invariant.
    lines.append(rf"\cmidrule{{1-{ncols_outer}}}")

    for h in parts.header_rows:
        lines.append(_expand_row(h, parts.col_specs, scale) + r" \\")
    if parts.header_rows:
        lines.append(r"\midrule")

    for b in parts.body_rows:
        lines.append(_expand_row(b, parts.col_specs, scale) + r" \\")

    if parts.footer_rows:
        lines.append(r"\midrule")
        for f in parts.footer_rows:
            lines.append(_expand_row(f, parts.col_specs, scale) + r" \\")

    return lines


def _outer_col_spec(ncols: int, *, left_pad: str = "0pt", right_pad: str = "0pt") -> str:
    """Outer column spec: left-aligned first column, right-aligned rest.

    ``left_pad`` / ``right_pad`` insert horizontal space at the edges of
    the outer tabular via ``@{\\hspace{..}}``.  The right pad is the key
    fix for the 'rightmost column looks pushed out' percept — with
    ``0pt`` the right-aligned numerics end exactly at the rule endpoint,
    which reads as visually inconsistent with the inner columns.
    """
    left = rf"@{{\hspace{{{left_pad}}}}}" if left_pad and left_pad != "0pt" else "@{}"
    right = rf"@{{\hspace{{{right_pad}}}}}" if right_pad and right_pad != "0pt" else "@{}"
    return f"{left}l " + " ".join(["r"] * (ncols - 1)) + f" {right}"


def render_stacked(composite: Composite) -> str:
    """Render *composite* as a single stacked tabular inside a LaTeX float.

    Supports mixed :class:`TablePanel` and :class:`FigurePanel` entries.
    Figure panels are rendered as a span-all ``\\includegraphics`` row
    inside the outer tabular, so the one-rectangular-contour invariant
    is preserved for mixed tables-and-figures composites.
    """
    panels: list[tuple[TablePanel | FigurePanel, PanelParts | None]] = []
    for panel in composite.panels:
        if isinstance(panel, TablePanel):
            panels.append((panel, parse_tabular(panel.tabular_src)))
        elif isinstance(panel, FigurePanel):
            panels.append((panel, None))
        else:
            raise TypeError(
                f"stacked composite requires TablePanel or FigurePanel; "
                f"got {type(panel).__name__} for panel {panel.panel_id!r}"
            )

    if not panels:
        raise ValueError("composite has no panels")

    table_ncols = [parts.ncols for _, parts in panels if parts is not None]
    ncols_outer = _lcm_many(table_ncols) if table_ncols else 1
    style = composite.resolved_style()

    float_kind = composite.float_kind
    label_prefix = "tab" if float_kind == "table" else "fig"

    float_position = "p" if style.fill_page else "htbp"
    lines: list[str] = [rf"\begin{{{float_kind}}}[{float_position}]", r"\centering"]
    if style.fill_page:
        lines.append(r"\vspace*{\fill}")
    if composite.caption:
        lines.append(rf"\caption{{{composite.caption}}}")
    lines.append(rf"\label{{{label_prefix}:{composite.composite_id}}}")
    if style.caption_gap and style.caption_gap != "0pt":
        lines.append(rf"\vspace{{{style.caption_gap}}}")
    lines.append(
        rf"{{\{style.font_size}"
        rf"\setlength{{\tabcolsep}}{{{style.tabcolsep}}}"
        rf"\renewcommand{{\arraystretch}}{{{style.arraystretch}}}%"
    )
    lines.append(
        rf"\begin{{tabular}}{{{_outer_col_spec(ncols_outer, left_pad=style.left_pad, right_pad=style.right_pad)}}}"
    )
    lines.append(r"\toprule")

    for i, (panel, parts) in enumerate(panels):
        if i > 0:
            # A single rule is enough between panels; the panel-label row
            # below it carries the name of the next section.
            if style.between_panel_above and style.between_panel_above != "0pt":
                lines.append(rf"\addlinespace[{style.between_panel_above}]")
            lines.append(r"\midrule")
            if style.between_panel_below and style.between_panel_below != "0pt":
                lines.append(rf"\addlinespace[{style.between_panel_below}]")
        if parts is None:
            # FigurePanel — label row, then a span-all includegraphics row.
            lines.append(_panel_label_row(panel.label, ncols_outer, style=style.panel_label_style) + r" \\")
            lines.append(rf"\cmidrule{{1-{ncols_outer}}}")
            lines.append(
                rf"\multicolumn{{{ncols_outer}}}{{c}}{{"
                rf"\includegraphics[width=0.9\linewidth,"
                rf"height={style.figure_max_height}\textheight,"
                rf"keepaspectratio]{{{panel.image_path}}}}} \\"
            )
        else:
            lines.extend(_emit_panel_rows(
                parts, panel.label, ncols_outer,
                label_style=style.panel_label_style,
            ))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    if composite.notes:
        lines.append(rf"\begin{{minipage}}{{\linewidth}}\{style.notes_font}")
        if style.notes_gap and style.notes_gap != "0pt":
            lines.append(rf"\vspace{{{style.notes_gap}}}")
        lines.append(rf"\textit{{Notes:}} {composite.notes}")
        lines.append(r"\end{minipage}")
    if style.fill_page:
        lines.append(r"\vspace*{\fill}")
    lines.append(rf"\end{{{float_kind}}}")
    return "\n".join(lines)
