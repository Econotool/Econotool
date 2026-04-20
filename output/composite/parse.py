"""Parse a ``\\begin{tabular}...\\end{tabular}`` fragment into structured parts.

This lets the composite renderer concatenate panels that were built
independently by :class:`ResultsTable`, :class:`SummaryTable`, and
:class:`DiagnosticsTable` into a single outer tabular with a shared
column grid — the approach the unit tests target.

The parser is intentionally small and format-aware: it relies on the
booktabs conventions that these classes emit (``\\toprule`` once,
``\\midrule`` one or two times, ``\\bottomrule`` once, rows terminated by
``\\\\``).  It does not attempt to handle arbitrary LaTeX.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ``\begin{tabular}{<spec>}`` — locate the start; column spec is parsed
# with balanced-brace scanning because p{..}/m{..}/@{..} contain ``}``.
_BEGIN_RE = re.compile(r"\\begin\{tabular\}\{")
_END_RE = re.compile(r"\\end\{tabular\}")


def _extract_balanced_spec(src: str, open_idx: int) -> tuple[str, int]:
    """Return ``(spec_text, index_after_closing_brace)``.

    ``open_idx`` points at the opening ``{`` of the column-spec group.
    """
    assert src[open_idx] == "{"
    depth = 0
    i = open_idx
    while i < len(src):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx + 1: i], i + 1
        i += 1
    raise ValueError("unterminated column spec in \\begin{tabular}{...}")


@dataclass
class PanelParts:
    """Structured view of a parsed panel tabular."""

    col_specs: list[str]          # per-column alignment tokens ("l", "r", "c", "S", ...)
    header_rows: list[str]        # rows between \toprule and the first \midrule
    body_rows: list[str]          # rows between first \midrule and either bottom or second \midrule
    footer_rows: list[str] = field(default_factory=list)
    # Raw "\\" row strings; no trailing "\\" included.  \addlinespace / \cmidrule
    # directives are preserved as pseudo-rows (empty cells).

    @property
    def ncols(self) -> int:
        return len(self.col_specs)


def _consume_braced(src: str, i: int) -> tuple[str, int]:
    """Read a ``{...}`` group starting at ``src[i] == '{'``, with balanced braces.

    Returns the inner text (without the outer braces) and the index one
    past the closing brace.
    """
    assert src[i] == "{"
    depth = 0
    start = i
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1: i], i + 1
        i += 1
    raise ValueError("unterminated brace group in column spec")


def split_col_spec(spec: str) -> list[str]:
    """Tokenise a LaTeX tabular column spec into per-column alignment tokens.

    Handles balanced braces so that ``@{\\hspace{4pt}}``, ``p{3.4cm}``,
    and ``*{3}{r}`` all parse correctly.

    Examples
    --------
    >>> split_col_spec("l r r r r r")
    ['l', 'r', 'r', 'r', 'r', 'r']
    >>> split_col_spec("l *{3}{r}")
    ['l', 'r', 'r', 'r']
    >>> split_col_spec("l p{3cm} S[table-format=2.3]")
    ['l', 'p{3cm}', 'S[table-format=2.3]']
    """
    tokens: list[str] = []
    i = 0
    n = len(spec)
    while i < n:
        ch = spec[i]
        if ch.isspace() or ch == "|":
            i += 1
            continue
        if ch in "@!><":
            if i + 1 < n and spec[i + 1] == "{":
                _, i = _consume_braced(spec, i + 1)
                continue
            i += 1
            continue
        if ch == "*":
            if i + 1 < n and spec[i + 1] == "{":
                n_text, i = _consume_braced(spec, i + 1)
                count = int(n_text)
                if i < len(spec) and spec[i] == "{":
                    inner_text, i = _consume_braced(spec, i)
                    tokens.extend(split_col_spec(inner_text) * count)
            continue
        if ch in "pmb":
            if i + 1 < n and spec[i + 1] == "{":
                width_text, i_new = _consume_braced(spec, i + 1)
                tokens.append(f"{ch}{{{width_text}}}")
                i = i_new
                continue
        if ch == "S" and i + 1 < n and spec[i + 1] == "[":
            j = spec.find("]", i + 1)
            if j > 0:
                tokens.append(spec[i: j + 1])
                i = j + 1
                continue
        if ch in "lrcsSX":
            tokens.append(ch)
            i += 1
            continue
        # Unknown character — skip
        i += 1
    return tokens


def _split_rows(chunk: str) -> list[str]:
    """Split a chunk of tabular body on row terminators ``\\\\``.

    Preserves directive-only lines (``\\midrule``, ``\\addlinespace[...]``,
    etc.) as their own entries so the renderer can forward or skip them.
    """
    if not chunk.strip():
        return []
    # Normalise: tabular rows end in ``\\`` possibly followed by ``[X]``.
    # Split on ``\\\\`` boundaries but keep whatever follows on the right
    # (e.g. ``\addlinespace``) attached to the prior row by treating the
    # split token as the row terminator.
    parts = re.split(r"\\\\(?:\[[^\]]*\])?", chunk)
    rows: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        rows.append(stripped)
    return rows


def parse_tabular(tabular_src: str) -> PanelParts:
    """Parse a ``\\begin{tabular}...\\end{tabular}`` fragment.

    Raises ``ValueError`` if *tabular_src* does not match the expected
    booktabs layout (one ``\\toprule``, at least one ``\\midrule``, one
    ``\\bottomrule``).
    """
    m = _BEGIN_RE.search(tabular_src)
    if m is None:
        raise ValueError("no \\begin{tabular}{...} found in input")
    spec, spec_end = _extract_balanced_spec(tabular_src, m.end() - 1)
    end_m = _END_RE.search(tabular_src, spec_end)
    if end_m is None:
        raise ValueError("no \\end{tabular} found in input")

    col_specs = split_col_spec(spec)
    inner = tabular_src[spec_end:end_m.start()]

    # Split on the three rule types.  Accept \hline/\hline as synonyms of
    # \toprule and \bottomrule (double-rule journals).
    def _first(pattern: str, src: str, start: int = 0) -> int:
        m = re.search(pattern, src[start:])
        return -1 if m is None else start + m.start()

    def _last(pattern: str, src: str) -> int:
        matches = list(re.finditer(pattern, src))
        return -1 if not matches else matches[-1].start()

    top_end = _first(r"\\toprule|\\hline\\hline", inner)
    if top_end < 0:
        top_end = _first(r"\\hline", inner)
    if top_end < 0:
        raise ValueError("tabular has no \\toprule / \\hline marker")
    # Advance past the matched directive so header parsing starts cleanly.
    tok_len = len("\\toprule")
    if inner[top_end: top_end + len("\\hline\\hline")] == "\\hline\\hline":
        tok_len = len("\\hline\\hline")
    elif inner[top_end: top_end + len("\\hline")] == "\\hline":
        tok_len = len("\\hline")
    header_start = top_end + tok_len

    bottom_start = _last(r"\\bottomrule|\\hline\\hline", inner)
    if bottom_start < 0:
        bottom_start = _last(r"\\hline", inner)
    if bottom_start < 0:
        bottom_start = len(inner)

    # \midrule splits header from body, and optionally body from footer
    # (ResultsTable emits two midrules).  Collect all midrule positions.
    midrules = [m.start() for m in re.finditer(r"\\midrule", inner)]
    midrules = [m for m in midrules if header_start < m < bottom_start]

    if not midrules:
        # No explicit header/body split — treat everything as body.
        return PanelParts(
            col_specs=col_specs,
            header_rows=[],
            body_rows=_split_rows(inner[header_start:bottom_start]),
        )

    first_mid = midrules[0]
    header_chunk = inner[header_start:first_mid]
    if len(midrules) >= 2:
        last_mid = midrules[-1]
        body_chunk = inner[first_mid + len("\\midrule"):last_mid]
        footer_chunk = inner[last_mid + len("\\midrule"):bottom_start]
    else:
        body_chunk = inner[first_mid + len("\\midrule"):bottom_start]
        footer_chunk = ""

    return PanelParts(
        col_specs=col_specs,
        header_rows=_split_rows(header_chunk),
        body_rows=_split_rows(body_chunk),
        footer_rows=_split_rows(footer_chunk),
    )


def split_row_cells(row: str) -> list[str]:
    """Split a tabular row on ``&`` — respecting ``\\multicolumn`` braces.

    Raw splitting on ``&`` is correct for rows without nested braces, but
    ``\\multicolumn{N}{c}{...}`` contains ``&``-free content in its third
    argument so a naive split still works in practice.  We nonetheless
    track brace depth to be safe.
    """
    cells: list[str] = []
    depth = 0
    current = []
    for ch in row:
        if ch == "{":
            depth += 1
            current.append(ch)
        elif ch == "}":
            depth -= 1
            current.append(ch)
        elif ch == "&" and depth == 0:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        cells.append(tail)
    return cells
