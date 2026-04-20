"""Publication-quality LaTeX table builder.

Produces journal-style tables using booktabs, threeparttable, and optional
siunitx.  Output is designed to fit a standard A4 page cleanly.

Required LaTeX packages in the document preamble::

    \\usepackage{booktabs}
    \\usepackage{threeparttable}
    \\usepackage{float}
    % optional for number alignment:
    \\usepackage{siunitx}

Public API
----------
ResultsTable      – multi-model regression results with optional panels
SummaryTable      – descriptive statistics (mean/std/min/max/N …)
DiagnosticsTable  – statistical test results with optional grouping

Typical usage::

    from econtools.output.tables.pub_latex import ResultsTable
    t = ResultsTable(
        results=[res1, res2, res3],
        labels=["(1)", "(2)", "(3)"],
        estimator_labels=["OLS", "2SLS", "2SLS"],
        panels=[("Baseline", ["educ", "exper"]), ("Extended", ["educ", "exper", "tenure"])],
        footer_stats=["N", "r_squared", "f_stat"],
        notes=["Standard errors in parentheses."],
        caption="Returns to Schooling",
        label="tab:returns",
    )
    print(t.to_latex())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from econtools._core.formatting import _fmt, _latex_escape, _latex_star, _star
from econtools._core.types import Estimate, TestResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Characters that indicate a string has already been LaTeX-escaped or contains
# intentional LaTeX markup.  If any of these sequences appear in a label we
# treat it as already-escaped and pass it through verbatim — double-escaping
# would turn "log\_gdp" into "log\textbackslash{}\_gdp" or "$R^2$" into a
# literal dollar sign.
_LATEX_MARKUP_MARKERS: tuple[str, ...] = (
    r"\_",
    r"\&",
    r"\%",
    r"\#",
    r"\$",
    r"\{",
    r"\}",
    r"\textbackslash",
    "$",  # math mode
    "{",  # grouping
    "}",
    "\\",  # any remaining backslash command
)


def _esc(text: str) -> str:
    """LaTeX-escape a variable / column name for use in a table cell.

    Thin wrapper over :func:`econtools._core.formatting._latex_escape` that
    skips the escape pass when the string already contains LaTeX markup
    (backslash commands, math delimiters, braces, or pre-escaped special
    characters).  This prevents double-escaping when callers pass in
    strings like ``r"log\\_gdp"`` or ``r"$\\hat\\alpha$"``.
    """
    if not isinstance(text, str):
        text = str(text)
    # If the string contains any LaTeX markup sequence we assume the caller
    # has already escaped specials or is injecting intentional LaTeX.
    for marker in _LATEX_MARKUP_MARKERS:
        if marker in text:
            return text
    return _latex_escape(text)


def _normalise_label(text: str) -> str:
    """Strip whitespace, hyphens, underscores; lowercase.  Used to decide
    whether two header strings are effectively the same.
    """
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _labels_are_equivalent(a: list[str], b: list[str]) -> bool:
    """True iff two label lists are pairwise equivalent after normalisation.

    Also returns True when every entry in *b* is already embedded in the
    corresponding entry of *a* — e.g. ``labels=["(1) OLS"]`` paired with
    ``estimator_labels=["OLS"]`` would duplicate the estimator name in two
    adjacent header rows, so the second row is suppressed.
    """
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        nx, ny = _normalise_label(x), _normalise_label(y)
        if nx == ny:
            continue
        if ny and ny in nx:
            continue
        return False
    return True


# Estimators for which R² is not a meaningful quantity and should be
# suppressed (rendered as a long-dash) in the footer.  2SLS/LIML produce
# numerically-valid R² but convention suppresses it; GMM/AB-GMM can
# return negative "R²" which is never meaningful.
_NO_RSQUARED_ESTIMATORS: set[str] = {
    "gmm",
    "ivgmm",
    "cue_gmm",
    "ivgmmcue",
    "arellano_bond",
    "ab_gmm",
    "dynamic_panel",
    "2sls",
    "iv2sls",
    "liml",
    # Panel estimators: within-R² is the meaningful quantity; plain R²
    # is misleading once fixed effects are absorbed.  Users who want
    # within-R² should explicitly request ``"r_squared_within"``.
    "fe",
    "panelols",
    "re",
    "randomeffects",
    "fd",
    "firstdifferenceols",
}

# Placeholder rendered for a suppressed R² / adjusted R² row.
_RSQUARED_DASH = r"---"


def _estimator_name(result: Estimate) -> str:
    """Best-effort lookup of the estimator key from an Estimate.

    Looks at ``model_type`` (set by the fit adapters) and normalises.
    Returns ``""`` if nothing helpful can be found.
    """
    raw_name = getattr(result, "model_type", "") or ""
    return str(raw_name).strip().lower()


def _suppress_rsquared(result: Estimate) -> bool:
    """Return True if R² should be rendered as ``---`` for this result."""
    name = _estimator_name(result)
    if name in _NO_RSQUARED_ESTIMATORS:
        return True
    rsq = getattr(result.fit, "r_squared", float("nan"))
    try:
        if not math.isnan(float(rsq)) and float(rsq) < 0.0:
            return True
    except (TypeError, ValueError):
        pass
    return False


# Cap on how far decimal precision is auto-widened before giving up and
# rendering in scientific notation.  Values requiring more than this many
# decimal places to show a leading significant figure always render in
# math-mode scientific notation regardless of the ``scientific`` flag,
# because ``0.00000`` in a publication table is worse than ``$3\times 10^{-6}$``.
_DECIMAL_WIDEN_CAP = 5


def _fmt_coef(value: float, digits: int, *, scientific: bool = False) -> str:
    """Format a coefficient or SE to *digits* decimals.

    Default behaviour (``scientific=False``) is publication style: plain
    decimal output, auto-widened to up to ``_DECIMAL_WIDEN_CAP`` dp so
    small-but-non-zero values still show at least one significant figure
    (``0.0005`` rather than ``0.000``).  Values still invisible at the
    cap fall back to math-mode scientific notation so the reader never
    sees a spurious zero.  When ``scientific=True`` any value with
    ``|x| < 10^{-digits}`` renders as ``$m\\times 10^{e}$`` directly.
    """
    if value is None:
        return ""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(fv) or math.isinf(fv):
        return _fmt(fv, digits)
    if fv == 0.0:
        return _fmt(fv, digits)
    threshold = 10.0 ** (-digits)
    if abs(fv) < threshold:
        if scientific:
            return _scientific(fv)
        widened = digits
        while widened < _DECIMAL_WIDEN_CAP and abs(fv) < 10.0 ** (-widened):
            widened += 1
        if abs(fv) < 10.0 ** (-widened):
            return _scientific(fv)
        return _fmt(fv, widened)
    return _fmt(fv, digits)


def _scientific(fv: float) -> str:
    mantissa, exponent = f"{fv:.2e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def _coef_cell(
    coef: float,
    pval: float,
    digits: int,
    stars: bool,
    *,
    scientific: bool = False,
) -> str:
    body = _fmt_coef(coef, digits, scientific=scientific)
    s = _star(pval) if stars else ""
    if not s:
        return body
    # If the formatted coefficient is in math mode (scientific notation),
    # merge the stars inside the math span so they attach to the number
    # rather than rendering as a separate adjacent math atom.
    if body.startswith("$") and body.endswith("$"):
        count = s.count("*")
        return body[:-1] + r"^{" + r"\ast" * count + r"}" + body[-1:]
    return body + _latex_star(s)


def _se_cell(
    se: float,
    digits: int,
    *,
    scientific: bool = False,
    delimiter: str = "parens",
) -> str:
    """Format SE cell with chosen delimiters.

    ``delimiter='parens'`` (default) renders ``(se)``.  ``delimiter='brackets'``
    renders ``[se]`` — used e.g. in Bertrand & Mullainathan (2004).
    """
    formatted = _fmt_coef(se, digits, scientific=scientific)
    open_d, close_d = ("(", ")") if delimiter == "parens" else ("[", "]")
    # If the formatter returned math-mode scientific notation the dollar
    # signs already wrap the expression — we must not nest $...$ inside.
    if formatted.startswith("$") and formatted.endswith("$"):
        return rf"{open_d}{formatted}{close_d}"
    return rf"${open_d}{formatted}{close_d}$"


def _nan_or(val: float, digits: int) -> str:
    if math.isnan(val):
        return ""
    return _fmt(val, digits)


def _first_stage_f(result: Estimate) -> float:
    """Extract first-stage F from a linearmodels IV result, or return nan.

    ``linearmodels`` exposes a ``first_stage`` attribute on 2SLS/LIML/GMM
    results with a ``diagnostics`` DataFrame indexed by endogenous
    regressor and containing a ``f.stat`` column.  With multiple
    endogenous regressors we report the *minimum* first-stage F — the
    relevant statistic for weak-instrument diagnosis.  OLS / other
    models simply return ``nan`` so the footer cell renders blank.
    """
    raw = result.raw
    for attr in ("first_stage", "_first_stage"):
        fs = getattr(raw, attr, None)
        if fs is None:
            continue
        diag = getattr(fs, "diagnostics", None)
        if diag is None:
            continue
        try:
            cols = getattr(diag, "columns", [])
            if "f.stat" in cols:
                vals = [float(v) for v in diag["f.stat"].tolist()]
                vals = [v for v in vals if not math.isnan(v)]
                if vals:
                    return float(min(vals))
            # Fallback: legacy layout where stat sits on a row named "f.stat".
            if hasattr(diag, "loc"):
                try:
                    return float(diag.loc["f.stat", "stat"])
                except Exception:
                    pass
        except Exception:
            continue
    return float("nan")


def _tabular_spec(n_cols: int, use_siunitx: bool) -> str:
    if use_siunitx:
        return "l" + " S[table-format=3.3]" * n_cols
    return "l" + " c" * n_cols


def _font_cmd(font_size: str | None) -> tuple[str, str]:
    """Return (open_cmd, close_cmd) for font size wrapping."""
    if font_size is None:
        return ("", "")
    valid = {"tiny", "scriptsize", "footnotesize", "small", "normalsize", "large"}
    if font_size not in valid:
        raise ValueError(f"font_size must be one of {valid}")
    return (rf"\{font_size}{{%", r"}")


def _build_table_env(
    body_lines: list[str],
    caption: str | None,
    label: str | None,
    float_position: str,
    fit_to_page: bool,
    font_size: str | None,
    notes: list[str],
    *,
    landscape: bool = False,
    fill_page: bool = False,
) -> str:
    """Wrap body_lines in table / threeparttable / resizebox environment.

    ``landscape=True`` wraps the entire table float in ``\\begin{landscape}``
    (from ``pdflscape``) so wide tables can be rotated onto their own page.

    ``fill_page=True`` vertically centres the table on a dedicated float
    page (``[p]`` specifier, ``\\vspace*{\\fill}`` above and below).  Used
    when each float should occupy one page of a figure-driven report.
    """
    effective_position = "p" if fill_page else float_position
    fp = f"[{effective_position}]" if effective_position else ""
    lines: list[str] = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.extend([rf"\begin{{table}}{fp}", r"\centering"])
    if fill_page:
        lines.append(r"\vspace*{\fill}")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")

    if notes:
        lines.append(r"\begin{threeparttable}")

    if fit_to_page:
        # Use \linewidth rather than \textwidth so the resize target
        # tracks the current context — crucial inside \begin{landscape},
        # where \textwidth keeps the portrait value while \linewidth is
        # the actual landscape-page width.
        lines.append(r"\resizebox{\linewidth}{!}{%")

    font_open, font_close = _font_cmd(font_size)
    if font_open:
        lines.append(font_open)

    lines.extend(body_lines)

    if font_open:
        lines.append(font_close)
    if fit_to_page:
        lines.append(r"}")  # closes resizebox

    if notes:
        lines.append(r"\begin{tablenotes}")
        lines.append(r"\footnotesize")
        for note in notes:
            lines.append(rf"\item {note}")
        lines.append(r"\end{tablenotes}")
        lines.append(r"\end{threeparttable}")

    if fill_page:
        lines.append(r"\vspace*{\fill}")
    lines.append(r"\end{table}")
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ResultsTable
# ---------------------------------------------------------------------------

# Footer stat label lookup
_FOOTER_LABELS: dict[str, str] = {
    "N": r"$N$",
    "r_squared": r"$R^2$",
    "r_squared_adj": r"Adj.\ $R^2$",
    "r_squared_within": r"Within $R^2$",
    "f_stat": r"$F$-statistic",
    "rmse": "RMSE",
    "aic": "AIC",
    "bic": "BIC",
    "first_stage_f": r"First-stage $F$",
    "cov_type": "SE type",
    "log_likelihood": "Log-likelihood",
}


def _footer_value(stat: str, result: Estimate, digits: int) -> str:
    if stat == "N":
        return str(int(result.fit.nobs))
    if stat == "first_stage_f":
        v = _first_stage_f(result)
        return _nan_or(v, digits)
    if stat == "cov_type":
        return _esc(result.cov_type)
    attr_map = {
        "r_squared": "r_squared",
        "r_squared_adj": "r_squared_adj",
        "r_squared_within": "r_squared_within",
        "f_stat": "f_stat",
        "rmse": "rmse",
        "aic": "aic",
        "bic": "bic",
        "log_likelihood": "log_likelihood",
    }
    if stat in attr_map:
        # Suppress R² / Adj R² for estimators where the quantity is not
        # meaningful (GMM, Arellano-Bond, 2SLS, LIML) or where the fit
        # would report a negative value — render a dash instead.
        if stat in {"r_squared", "r_squared_adj"} and _suppress_rsquared(result):
            return _RSQUARED_DASH
        v = getattr(result.fit, attr_map[stat], float("nan"))
        return _nan_or(float(v), digits)
    return ""


class ResultsTable:
    """Publication-quality multi-model results table.

    Parameters
    ----------
    results:
        List of fitted :class:`~econtools._core.types.Estimate` objects.
    labels:
        Column header labels e.g. ``["(1)", "(2)", "(3)"]``.  Defaults to
        ``(1)``, ``(2)``, …
    estimator_labels:
        Optional second header row with estimator names (e.g. "OLS", "2SLS").
    variable_names:
        Mapping from parameter name to display name.
    omit_vars:
        Parameter names to suppress from the table body (e.g. ``["const"]``).
    panels:
        List of ``(panel_label, [var_names])`` tuples.  Variables are shown
        under their panel header.  Variables not in any panel are ignored.
        If ``None`` all variables are shown in a flat list.
    column_groups:
        Spanning-header specification.  Two accepted shapes:

        * **Flat** (single level): ``[(label, [cols]), ...]`` — one row of
          spanners across the data columns.
        * **Nested** (multi-level): ``[[(outer_label, [cols]), ...],
          [(inner_label, [cols]), ...], ...]`` — each inner list is one
          level of spanners rendered top-to-bottom.  Useful for tables like
          AK91 Table IV (OLS/2SLS outer groups with sub-group headers
          underneath) or CMR 2008 Table 3 (three-deep header nesting).

        Indices are 1-based relative to the data columns (not including
        the label column).  Gaps between groups on a level render as empty
        cells.
    footer_stats:
        Statistics to show below the separator.  Supports built-in keys
        (``"N"``, ``"r_squared"``, ``"r_squared_adj"``, ``"r_squared_within"``,
        ``"f_stat"``, ``"rmse"``, ``"aic"``, ``"bic"``, ``"first_stage_f"``,
        ``"cov_type"``) and custom ``(label, [values])`` tuples.
    notes:
        List of note strings rendered in the ``tablenotes`` block.
    add_star_note:
        Automatically append a significance-level note.
    stars:
        Whether to append significance stars to coefficients.
    digits:
        Decimal places.
    fit_to_page:
        Wrap the tabular in ``\\resizebox{\\linewidth}{!}{}``.  Uses
        ``\\linewidth`` (not ``\\textwidth``) so the target width tracks
        the enclosing context — critical inside ``\\begin{landscape}``,
        where ``\\textwidth`` remains the portrait value.
    use_siunitx:
        Use ``siunitx`` ``S`` columns for number alignment.
    caption:
        Table caption.
    label:
        LaTeX ``\\label`` key.
    float_position:
        Float specifier e.g. ``"htbp"`` or ``"H"``.
    font_size:
        Font size command (``"small"``, ``"footnotesize"``, etc.) or ``None``.
    row_superscripts:
        Optional mapping from variable name to a footnote key string
        (e.g. ``{"earnings": "a", "hours": "b"}``).  Each keyed variable
        renders its row label followed by ``$^{<key>}$`` so readers can
        cross-reference a matching note in ``notes``.  Useful for
        Card & Krueger (1994) Tables 3–4 style presentation.
    long_table:
        Render as ``\\begin{longtable}`` (from the ``longtable`` package)
        so the table can break across pages.  Not a float — placement is
        wherever the fragment appears in the document.  Incompatible with
        ``fit_to_page`` (longtable resizes itself to text width) and
        ``float_position`` / ``landscape`` (ignored in this mode); notes
        render beneath the table as plain ``\\footnotesize`` paragraphs
        rather than via ``threeparttable``.
    """

    def __init__(
        self,
        results: list[Estimate],
        labels: list[str] | None = None,
        estimator_labels: list[str] | None = None,
        variable_names: dict[str, str] | None = None,
        omit_vars: list[str] | None = None,
        keep_vars: list[str] | None = None,
        panels: list[tuple[str, list[str]]] | None = None,
        column_groups: list[tuple[str, list[int]]] | None = None,
        footer_stats: list[str | tuple] | None = None,
        bootstrap_ses: dict[str, list[float]] | None = None,
        bootstrap_se_label: str = "",
        notes: list[str] | None = None,
        add_star_note: bool = True,
        stars: bool = True,
        digits: int = 3,
        fit_to_page: bool = True,
        use_siunitx: bool = False,
        caption: str | None = None,
        label: str | None = None,
        float_position: str = "htbp",
        font_size: str | None = "small",
        scientific_notation: bool = False,
        se_delimiter: str = "parens",
        bootstrap_se_delimiter: str = "brackets",
        landscape: bool = False,
        fill_page: bool = False,
        row_superscripts: dict[str, str] | None = None,
        long_table: bool = False,
        missing_cell: str = "---",
        double_rule: bool = False,
    ) -> None:
        if not results:
            raise ValueError("results must be a non-empty list.")
        self.results = results
        self.labels = labels or [f"({i + 1})" for i in range(len(results))]
        # Defensive: suppress estimator_labels if they would duplicate the
        # primary label row (case-insensitive, ignoring punctuation such as
        # hyphens/underscores).  Callers that intentionally want both rows
        # should pass materially different strings (e.g. "(1)"/"(2)" as
        # labels and "OLS"/"2SLS" as estimator_labels).
        if estimator_labels is not None and _labels_are_equivalent(
            self.labels, estimator_labels
        ):
            estimator_labels = None
        # Defensive: if every column shares the same estimator (e.g. a SUR
        # table with equations "VAC1".."VAC4" but a single estimator "SUR"),
        # surface that single estimator once in a tablenote rather than
        # repeating it across every column.
        self._estimator_note: str | None = None
        if estimator_labels is not None and len(estimator_labels) > 0:
            unique = {str(e).strip() for e in estimator_labels if str(e).strip()}
            if len(unique) == 1:
                sole = next(iter(unique))
                self._estimator_note = (
                    f"All columns estimated by {sole}."
                )
                estimator_labels = None
        self.estimator_labels = estimator_labels
        self.variable_names = variable_names or {}
        self.omit_vars = set(omit_vars or [])
        self.keep_vars = list(keep_vars) if keep_vars else None
        self.panels = panels
        self.column_groups = column_groups
        self.footer_stats = footer_stats or ["N", "r_squared", "r_squared_adj"]
        self.bootstrap_ses = dict(bootstrap_ses) if bootstrap_ses else {}
        if self.bootstrap_ses:
            for var, ses in self.bootstrap_ses.items():
                if len(ses) != len(results):
                    raise ValueError(
                        f"bootstrap_ses['{var}'] has {len(ses)} entries, "
                        f"expected {len(results)} (one per column)."
                    )
        self.bootstrap_se_label = bootstrap_se_label
        if bootstrap_se_delimiter not in {"parens", "brackets"}:
            raise ValueError("bootstrap_se_delimiter must be 'parens' or 'brackets'")
        self.bootstrap_se_delimiter = bootstrap_se_delimiter
        self.notes = list(notes or [])
        if self._estimator_note is not None:
            # Prepend so it appears above the significance-star note.
            self.notes.insert(0, self._estimator_note)
        if add_star_note and stars:
            self.notes.append(
                r"$^{*}$ $p<0.10$,\ $^{**}$ $p<0.05$,\ $^{***}$ $p<0.01$."
            )
        self.stars = stars
        self.digits = digits
        self.fit_to_page = fit_to_page
        self.use_siunitx = use_siunitx
        self.caption = caption
        self.label = label
        self.float_position = float_position
        self.font_size = font_size
        self.scientific_notation = scientific_notation
        if se_delimiter not in {"parens", "brackets"}:
            raise ValueError("se_delimiter must be 'parens' or 'brackets'")
        self.se_delimiter = se_delimiter
        self.landscape = landscape
        self.fill_page = fill_page
        self.row_superscripts = dict(row_superscripts or {})
        self.long_table = long_table
        self.missing_cell = missing_cell
        self.double_rule = double_rule

    @property
    def _top_rule(self) -> str:
        return r"\hline\hline" if self.double_rule else r"\toprule"

    @property
    def _bottom_rule(self) -> str:
        return r"\hline\hline" if self.double_rule else r"\bottomrule"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_vars(self) -> list[str]:
        """Union of all param names across results, preserving first-seen order.

        When :pyattr:`keep_vars` is set, only those variables (in that
        order) are returned, filtered to names that actually appear in at
        least one result.
        """
        if self.keep_vars is not None:
            available: set[str] = set()
            for res in self.results:
                available.update(res.params.index)
            return [v for v in self.keep_vars
                    if v in available and v not in self.omit_vars]
        seen: set[str] = set()
        out: list[str] = []
        for res in self.results:
            for name in res.params.index:
                if name not in seen:
                    seen.add(name)
                    out.append(name)
        return [v for v in out if v not in self.omit_vars]

    def _display_name(self, var: str) -> str:
        raw = self.variable_names.get(var, var)
        label = _esc(raw)
        sup = self.row_superscripts.get(var)
        if sup:
            # Keyed footnote superscript (e.g. ``^{a}``) — attached to the
            # variable label so the reader can cross-reference a note in
            # the tablenotes block (Card & Krueger 1994 Tables 3/4 style).
            label = rf"{label}$^{{{_esc(sup)}}}$"
        return label

    def _coef_row(self, var: str) -> list[str]:
        """Return the coefficient row cells (one per result)."""
        cells = []
        for res in self.results:
            if var in res.params.index:
                cells.append(
                    _coef_cell(
                        float(res.params[var]),
                        float(res.pvalues[var]),
                        self.digits,
                        self.stars,
                        scientific=self.scientific_notation,
                    )
                )
            else:
                cells.append(self.missing_cell)
        return cells

    def _se_row(self, var: str) -> list[str]:
        cells = []
        for res in self.results:
            if var in res.bse.index:
                cells.append(
                    _se_cell(
                        float(res.bse[var]),
                        self.digits,
                        scientific=self.scientific_notation,
                        delimiter=self.se_delimiter,
                    )
                )
            else:
                cells.append("")
        return cells

    def _bootstrap_se_row(self, var: str) -> list[str] | None:
        """Return a bootstrap-SE row for *var*, or ``None`` if no entry.

        ``NaN`` entries in ``bootstrap_ses[var]`` render as empty cells,
        so a single dict can cover multiple variables with overlapping
        column support (e.g. only the IV column gets a bootstrap SE).
        """
        if var not in self.bootstrap_ses:
            return None
        ses = self.bootstrap_ses[var]
        cells: list[str] = []
        for se in ses:
            try:
                fv = float(se)
            except (TypeError, ValueError):
                cells.append("")
                continue
            if math.isnan(fv):
                cells.append("")
                continue
            cells.append(
                _se_cell(
                    fv,
                    self.digits,
                    scientific=self.scientific_notation,
                    delimiter=self.bootstrap_se_delimiter,
                )
            )
        return cells

    def _tabular_row(self, label: str, cells: list[str]) -> str:
        return " & ".join([label] + cells) + r" \\"

    def _panel_header(self, panel_label: str) -> str:
        n = len(self.results) + 1
        return (
            rf"\multicolumn{{{n}}}{{l}}{{\textit{{{panel_label}}}}} \\"
            + r"[2pt]"
        )

    def _body_lines(self) -> list[str]:
        lines: list[str] = []
        if self.panels:
            for panel_label, var_list in self.panels:
                lines.append(self._panel_header(panel_label))
                for var in var_list:
                    if var in self.omit_vars:
                        continue
                    lines.append(
                        self._tabular_row(self._display_name(var), self._coef_row(var))
                    )
                    lines.append(
                        self._tabular_row("", self._se_row(var))
                    )
                    boot = self._bootstrap_se_row(var)
                    if boot is not None:
                        lines.append(
                            self._tabular_row(self.bootstrap_se_label, boot)
                        )
                lines.append(r"\addlinespace[2pt]")
        else:
            for var in self._all_vars():
                lines.append(
                    self._tabular_row(self._display_name(var), self._coef_row(var))
                )
                lines.append(self._tabular_row("", self._se_row(var)))
                boot = self._bootstrap_se_row(var)
                if boot is not None:
                    lines.append(
                        self._tabular_row(self.bootstrap_se_label, boot)
                    )
        return lines

    def _footer_lines(self) -> list[str]:
        lines: list[str] = []
        for stat in self.footer_stats:
            if isinstance(stat, tuple):
                label, values = stat
                assert len(values) == len(self.results), (
                    f"Custom footer row '{label}' must have {len(self.results)} values."
                )
                lines.append(self._tabular_row(label, [str(v) for v in values]))
            else:
                lbl = _FOOTER_LABELS.get(stat, stat)
                vals = [_footer_value(stat, res, self.digits) for res in self.results]
                lines.append(self._tabular_row(lbl, vals))
        return lines

    def _normalize_column_group_levels(
        self,
    ) -> list[list[tuple[str, list[int]]]]:
        """Return the column groups as a list of levels (outer → inner).

        Accepts two input shapes for :pyattr:`column_groups`:

        * **Flat** — ``[(label, [cols]), ...]`` — treated as a single-level
          spanner row (the historical form).
        * **Nested** — ``[[(outer_label, [cols]), ...], [(inner_label, [cols]), ...], ...]``
          — each inner list is one level of spanners rendered top-to-bottom,
          enabling multi-level headers (e.g. AK91 Table IV's OLS / 2SLS outer
          groups with inner covariate sub-groups).
        """
        if not self.column_groups:
            return []
        first = self.column_groups[0]
        # Flat form: first element is a (label, indices) tuple — (str, sequence).
        if (
            isinstance(first, tuple)
            and len(first) == 2
            and isinstance(first[0], str)
        ):
            return [list(self.column_groups)]
        # Nested form: first element is itself a list of tuples.
        return [list(level) for level in self.column_groups]

    def _render_one_level(
        self, level: list[tuple[str, list[int]]]
    ) -> list[str]:
        """Render one level of spanners (one header row + its cmidrules).

        Handles gaps between groups by inserting empty cells so the column
        count matches ``len(self.results) + 1`` regardless of whether the
        groups cover every data column.
        """
        n_model_cols = len(self.results)
        # Sort groups by starting column so we can walk left-to-right.
        sorted_groups = sorted(level, key=lambda g: min(g[1]))
        header_cells = [""]  # label column (col 1 in LaTeX)
        cmidrules: list[str] = []
        cursor = 1  # next data column (1-based) yet to be accounted for
        for group_label, col_indices in sorted_groups:
            lo = min(col_indices)
            hi = max(col_indices)
            while cursor < lo:
                header_cells.append("")
                cursor += 1
            span = hi - lo + 1
            header_cells.append(
                rf"\multicolumn{{{span}}}{{c}}{{{group_label}}}"
            )
            cmidrules.append(rf"\cmidrule(lr){{{lo + 1}-{hi + 1}}}")
            cursor = hi + 1
        while cursor <= n_model_cols:
            header_cells.append("")
            cursor += 1
        out = [" & ".join(header_cells) + r" \\"]
        if cmidrules:
            out.append(" ".join(cmidrules))
        return out

    def _column_group_lines(self) -> list[str]:
        """Render column group spanning headers with cmidrule.

        Produces one header row + one cmidrule row per level.  For
        multi-level headers the outermost spanners are emitted first.
        """
        levels = self._normalize_column_group_levels()
        if not levels:
            return []
        out: list[str] = []
        for level in levels:
            out.extend(self._render_one_level(level))
        return out

    def to_tabular(self) -> str:
        """Return only the ``\\begin{tabular}...\\end{tabular}`` block."""
        n = len(self.results)
        spec = _tabular_spec(n, self.use_siunitx)

        body: list[str] = [rf"\begin{{tabular}}{{{spec}}}"]
        body.append(self._top_rule)

        # Column group spanning headers (if any)
        if self.column_groups:
            body.extend(self._column_group_lines())

        # Primary header row (model numbers)
        body.append(self._tabular_row("", self.labels))

        # Optional estimator name row
        if self.estimator_labels:
            body.append(self._tabular_row("", self.estimator_labels))

        body.append(r"\midrule")

        # Body: coefficients (with panels or flat)
        body.extend(self._body_lines())

        body.append(r"\midrule")

        # Footer statistics
        body.extend(self._footer_lines())

        body.append(self._bottom_rule)
        body.append(r"\end{tabular}")
        return "\n".join(body)

    def to_latex(self) -> str:
        """Return the full LaTeX table string."""
        if self.long_table:
            return self._to_longtable()

        n = len(self.results)
        spec = _tabular_spec(n, self.use_siunitx)

        body: list[str] = [rf"\begin{{tabular}}{{{spec}}}"]
        body.append(self._top_rule)

        # Column group spanning headers (if any)
        if self.column_groups:
            body.extend(self._column_group_lines())

        # Primary header row (model numbers)
        body.append(self._tabular_row("", self.labels))

        # Optional estimator name row
        if self.estimator_labels:
            body.append(self._tabular_row("", self.estimator_labels))

        body.append(r"\midrule")

        # Body: coefficients (with panels or flat)
        body.extend(self._body_lines())

        body.append(r"\midrule")

        # Footer statistics
        body.extend(self._footer_lines())

        body.append(self._bottom_rule)
        body.append(r"\end{tabular}")

        return _build_table_env(
            body,
            caption=self.caption,
            label=self.label,
            float_position=self.float_position,
            fit_to_page=self.fit_to_page,
            font_size=self.font_size,
            notes=self.notes,
            landscape=self.landscape,
            fill_page=self.fill_page,
        )

    def _to_longtable(self) -> str:
        """Render the table as a ``longtable`` that breaks across pages.

        Longtable cannot live inside ``\\begin{table}`` (it is not a float)
        and does not support ``threeparttable``/``\\resizebox``.  Notes are
        rendered beneath the table as plain ``\\footnotesize`` paragraphs.
        The column header block is repeated on every page via the
        ``\\endfirsthead`` / ``\\endhead`` commands so readers do not lose
        their place at a page break.
        """
        n = len(self.results)
        spec = _tabular_spec(n, self.use_siunitx)
        out: list[str] = []

        font_open, font_close = _font_cmd(self.font_size)
        if font_open:
            out.append(font_open)
        out.append(rf"\begin{{longtable}}{{{spec}}}")

        # Caption on first page only — longtable convention puts the
        # caption on its own row terminated by ``\\`` before the header.
        if self.caption:
            label_part = rf"\label{{{self.label}}}" if self.label else ""
            out.append(rf"\caption{{{self.caption}}}{label_part} \\")
        elif self.label:
            out.append(rf"\label{{{self.label}}} \\")

        # Build the repeating header block once and emit it for both
        # \endfirsthead and \endhead so pages 2+ also show column labels.
        header_block: list[str] = [self._top_rule]
        if self.column_groups:
            header_block.extend(self._column_group_lines())
        header_block.append(self._tabular_row("", self.labels))
        if self.estimator_labels:
            header_block.append(self._tabular_row("", self.estimator_labels))
        header_block.append(r"\midrule")

        out.extend(header_block)
        out.append(r"\endfirsthead")

        # Continuation header: mark the table as continued, then repeat
        # the column block so readers on later pages still see labels.
        cont_label = r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{(continued)}} \\"
        out.append(cont_label)
        out.extend(header_block)
        out.append(r"\endhead")

        # Bottom-of-page footer (every page except last).
        out.append(r"\midrule")
        out.append(
            r"\multicolumn{" + str(n + 1)
            + r"}{r}{\textit{Continued on next page}} \\"
        )
        out.append(r"\endfoot")

        # Bottom of the very last page.
        out.append(r"\midrule")
        out.extend(self._footer_lines())
        out.append(self._bottom_rule)
        out.append(r"\endlastfoot")

        # Body (coefficients, panels, etc.).
        out.extend(self._body_lines())

        out.append(r"\end{longtable}")
        if font_close:
            out.append(font_close)

        # Notes rendered as plain \footnotesize paragraphs BENEATH the
        # longtable (threeparttable does not compose with longtable).
        if self.notes:
            out.append(r"\begin{flushleft}")
            out.append(r"\footnotesize")
            for note in self.notes:
                out.append(note + r" \par")
            out.append(r"\end{flushleft}")

        return "\n".join(out)

    def __str__(self) -> str:
        return self.to_latex()


# ---------------------------------------------------------------------------
# SummaryTable
# ---------------------------------------------------------------------------

_STAT_LABELS: dict[str, str] = {
    "N": r"$N$",
    "mean": "Mean",
    "std": r"Std.\ Dev.",
    "min": "Min",
    "p25": "P25",
    "median": "Median",
    "p75": "P75",
    "max": "Max",
}

_STAT_FUNCS: dict[str, Any] = {
    "N": lambda s: int(s.count()),
    "mean": lambda s: s.mean(),
    "std": lambda s: s.std(),
    "min": lambda s: s.min(),
    "p25": lambda s: s.quantile(0.25),
    "median": lambda s: s.median(),
    "p75": lambda s: s.quantile(0.75),
    "max": lambda s: s.max(),
}


class SummaryTable:
    """Publication-quality descriptive statistics table.

    Parameters
    ----------
    df:
        Source :class:`pandas.DataFrame`.
    vars:
        Columns to include.  ``None`` selects all numeric columns.
    stats:
        Statistics to compute per variable.  Supported: ``"N"``, ``"mean"``,
        ``"std"``, ``"min"``, ``"p25"``, ``"median"``, ``"p75"``, ``"max"``.
    var_names:
        Display name overrides (column name → display name).
    panels:
        ``[(panel_label, [var_names])]`` for grouped rows.
    caption / label / digits / fit_to_page / float_position / font_size:
        Standard table options.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vars: list[str] | None = None,
        stats: list[str] = ("mean", "std", "min", "max", "N"),
        var_names: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
        description_width: str = "5cm",
        description_label: str = "Description",
        panels: list[tuple[str, list[str]]] | None = None,
        notes: list[str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        digits: int = 3,
        fit_to_page: bool = True,
        float_position: str = "htbp",
        font_size: str | None = "small",
        double_rule: bool = False,
        fill_page: bool = False,
    ) -> None:
        self.df = df
        self.vars = vars or list(df.select_dtypes("number").columns)
        self.stats = list(stats)
        self.var_names = var_names or {}
        self.descriptions = dict(descriptions) if descriptions else {}
        self.description_width = description_width
        self.description_label = description_label
        self.panels = panels
        self.notes = list(notes or [])
        self.caption = caption
        self.label = label
        self.digits = digits
        self.fit_to_page = fit_to_page
        self.float_position = float_position
        self.font_size = font_size
        self.double_rule = double_rule
        self.fill_page = fill_page

    @property
    def _top_rule(self) -> str:
        return r"\hline\hline" if self.double_rule else r"\toprule"

    @property
    def _bottom_rule(self) -> str:
        return r"\hline\hline" if self.double_rule else r"\bottomrule"

    def _display_name(self, col: str) -> str:
        return _esc(self.var_names.get(col, col))

    def _stat_val(self, col: str, stat: str) -> str:
        series = self.df[col].dropna()
        fn = _STAT_FUNCS[stat]
        v = fn(series)
        if stat == "N":
            return str(int(v))
        return _fmt(float(v), self.digits)

    def _tabular_row(self, label: str, cells: list[str]) -> str:
        return " & ".join([label] + cells) + r" \\"

    def _var_row(self, col: str) -> str:
        cells = [self._stat_val(col, s) for s in self.stats]
        if self.descriptions:
            desc = _esc(self.descriptions.get(col, ""))
            cells = [desc] + cells
        return self._tabular_row(self._display_name(col), cells)

    @property
    def _has_descriptions(self) -> bool:
        return bool(self.descriptions)

    def _panel_header(self, panel_label: str) -> str:
        n = len(self.stats) + 1 + (1 if self._has_descriptions else 0)
        return rf"\multicolumn{{{n}}}{{l}}{{\textit{{{panel_label}}}}} \\[2pt]"

    def _build_body(self) -> list[str]:
        if self._has_descriptions:
            spec = f"l p{{{self.description_width}}}" + " r" * len(self.stats)
        else:
            spec = "l" + " r" * len(self.stats)
        header_cells = [_STAT_LABELS.get(s, s) for s in self.stats]
        if self._has_descriptions:
            header_cells = [self.description_label] + header_cells

        body: list[str] = [rf"\begin{{tabular}}{{{spec}}}"]
        body.append(self._top_rule)
        body.append(self._tabular_row("Variable", header_cells))
        body.append(r"\midrule")

        if self.panels:
            for panel_label, var_list in self.panels:
                body.append(self._panel_header(panel_label))
                for col in var_list:
                    if col not in self.df.columns:
                        continue
                    body.append(self._var_row(col))
                body.append(r"\addlinespace[2pt]")
        else:
            for col in self.vars:
                if col not in self.df.columns:
                    continue
                body.append(self._var_row(col))

        body.append(self._bottom_rule)
        body.append(r"\end{tabular}")
        return body

    def to_tabular(self) -> str:
        """Return only the ``\\begin{tabular}...\\end{tabular}`` block."""
        return "\n".join(self._build_body())

    def to_latex(self) -> str:
        return _build_table_env(
            self._build_body(),
            caption=self.caption,
            label=self.label,
            float_position=self.float_position,
            fit_to_page=self.fit_to_page,
            font_size=self.font_size,
            notes=self.notes,
            fill_page=self.fill_page,
        )

    def __str__(self) -> str:
        return self.to_latex()


# ---------------------------------------------------------------------------
# InfoTable
# ---------------------------------------------------------------------------


class InfoTable:
    """Generic info-style table for literature reviews, data sources,
    variable definitions, and similar key / value / description content.

    The table has one row per dict in *rows*; columns are declared
    explicitly so the same class can render:

    * a literature-review table (author, year, sample, estimate, notes)
    * a data-sources table (variable, source, frequency, coverage)
    * a variable-definitions table (name, description, units, construction)
    * a mixed table via ``panels=...`` grouping rows by purpose.

    Parameters
    ----------
    rows:
        List of ``{col_key: cell_text}`` dicts.  Missing keys render as
        empty cells.
    columns:
        Ordered list of column specs.  Each entry is a dict with keys

        * ``key`` — dict-key used to pull cell text from *rows*;
        * ``label`` — header text (LaTeX allowed);
        * ``align`` — one of ``"l"``, ``"c"``, ``"r"`` (default ``"l"``);
        * ``width`` — optional width (e.g. ``"4cm"``) → ``p{width}`` column.
    panels:
        Optional ``[(panel_label, [row_indices])]`` for grouped rows.
        Indices refer to positions in *rows*.
    notes / caption / label / fit_to_page / float_position / font_size /
    double_rule / fill_page:
        Standard table options.
    """

    def __init__(
        self,
        rows: list[dict],
        columns: list[dict],
        panels: list[tuple[str, list[int]]] | None = None,
        notes: list[str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        fit_to_page: bool = False,
        float_position: str = "htbp",
        font_size: str | None = "small",
        double_rule: bool = False,
        fill_page: bool = False,
    ) -> None:
        if not columns:
            raise ValueError("InfoTable requires at least one column spec.")
        self.rows = list(rows)
        self.columns = list(columns)
        self.panels = panels
        self.notes = list(notes or [])
        self.caption = caption
        self.label = label
        self.fit_to_page = fit_to_page
        self.float_position = float_position
        self.font_size = font_size
        self.double_rule = double_rule
        self.fill_page = fill_page

    @property
    def _top_rule(self) -> str:
        return r"\hline\hline" if self.double_rule else r"\toprule"

    @property
    def _bottom_rule(self) -> str:
        return r"\hline\hline" if self.double_rule else r"\bottomrule"

    def _col_spec_fragment(self, col: dict) -> str:
        if col.get("width"):
            return f"p{{{col['width']}}}"
        align = col.get("align", "l")
        if align not in {"l", "c", "r"}:
            raise ValueError(f"InfoTable column align must be l|c|r, got {align!r}")
        return align

    def _cell(self, row: dict, col: dict) -> str:
        value = row.get(col["key"], "")
        if value is None:
            return ""
        return _esc(str(value))

    def _row_line(self, row: dict) -> str:
        cells = [self._cell(row, c) for c in self.columns]
        return " & ".join(cells) + r" \\"

    def _header_line(self) -> str:
        labels: list[str] = []
        for c in self.columns:
            lbl = c.get("label", c["key"])
            if c.get("align") == "c" and not c.get("width"):
                lbl = rf"\multicolumn{{1}}{{c}}{{{lbl}}}"
            elif c.get("align") == "r" and not c.get("width"):
                lbl = rf"\multicolumn{{1}}{{r}}{{{lbl}}}"
            labels.append(lbl)
        return " & ".join(labels) + r" \\"

    def _panel_header(self, panel_label: str) -> str:
        n = len(self.columns)
        return rf"\multicolumn{{{n}}}{{l}}{{\textit{{{panel_label}}}}} \\[2pt]"

    def _build_body(self) -> list[str]:
        spec = " ".join(self._col_spec_fragment(c) for c in self.columns)
        body: list[str] = [rf"\begin{{tabular}}{{{spec}}}"]
        body.append(self._top_rule)
        body.append(self._header_line())
        body.append(r"\midrule")

        if self.panels:
            for panel_label, indices in self.panels:
                body.append(self._panel_header(panel_label))
                for i in indices:
                    if 0 <= i < len(self.rows):
                        body.append(self._row_line(self.rows[i]))
                body.append(r"\addlinespace[2pt]")
        else:
            for row in self.rows:
                body.append(self._row_line(row))

        body.append(self._bottom_rule)
        body.append(r"\end{tabular}")
        return body

    def to_tabular(self) -> str:
        """Return only the ``\\begin{tabular}...\\end{tabular}`` block."""
        return "\n".join(self._build_body())

    def to_latex(self) -> str:
        return _build_table_env(
            self._build_body(),
            caption=self.caption,
            label=self.label,
            float_position=self.float_position,
            fit_to_page=self.fit_to_page,
            font_size=self.font_size,
            notes=self.notes,
            fill_page=self.fill_page,
        )

    def __str__(self) -> str:
        return self.to_latex()


# ---------------------------------------------------------------------------
# DiagnosticsTable
# ---------------------------------------------------------------------------


class DiagnosticsTable:
    """Publication-quality diagnostic test results table.

    Parameters
    ----------
    tests:
        List of :class:`~econtools._core.types.TestResult` objects.
    groups:
        ``[(group_label, [indices])]`` where indices are 0-based into ``tests``.
        If provided, tests are rendered under group headers.
    show_h0:
        Include H₀ text as a sub-row below each test.
    notes / caption / label / digits / fit_to_page / float_position / font_size:
        Standard table options.
    """

    def __init__(
        self,
        tests: list[TestResult],
        groups: list[tuple[str, list[int]]] | None = None,
        show_h0: bool = False,
        notes: list[str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        digits: int = 3,
        fit_to_page: bool = False,
        float_position: str = "htbp",
        font_size: str | None = None,
        double_rule: bool = False,
    ) -> None:
        self.tests = tests
        self.groups = groups
        self.show_h0 = show_h0
        self.notes = list(notes or [])
        self.caption = caption
        self.label = label
        self.digits = digits
        self.fit_to_page = fit_to_page
        self.float_position = float_position
        self.font_size = font_size
        self.double_rule = double_rule

    @property
    def _top_rule(self) -> str:
        return r"\hline\hline" if self.double_rule else r"\toprule"

    @property
    def _bottom_rule(self) -> str:
        return r"\hline\hline" if self.double_rule else r"\bottomrule"

    def _df_str(self, df: float | tuple | None) -> str:
        if df is None:
            return ""
        if isinstance(df, tuple):
            return f"({df[0]:.0f}, {df[1]:.0f})"
        try:
            if math.isnan(float(df)):
                return ""
        except (TypeError, ValueError):
            pass
        return f"{df:.0f}"

    @staticmethod
    def _nan_dash(value: float | None, digits: int) -> str:
        """Render *value* with *digits* precision; dash if NaN / None."""
        if value is None:
            return r"---"
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return r"---"
        if math.isnan(fv):
            return r"---"
        return _fmt(fv, digits)

    def _test_rows(self, test_indices: list[int]) -> list[str]:
        rows = []
        for i in test_indices:
            r = self.tests[i]
            reject = "Yes" if r.reject else "No"
            rows.append(
                " & ".join([
                    _esc(r.test_name),
                    self._nan_dash(r.statistic, self.digits),
                    self._nan_dash(r.pvalue, self.digits),
                    self._df_str(r.df),
                    r.distribution,
                    reject,
                ]) + r" \\"
            )
            if self.show_h0:
                rows.append(
                    rf"\multicolumn{{6}}{{l}}{{\footnotesize{{H$_0$: {_esc(r.null_hypothesis)}}}}} \\"
                )
        return rows

    def _group_header(self, label: str) -> str:
        return rf"\multicolumn{{6}}{{l}}{{\textit{{{label}}}}} \\[2pt]"

    def _build_body(self) -> list[str]:
        spec = "l r r r l c"
        header = r"Test & Stat & $p$-value & df & Dist. & Reject \\"

        body: list[str] = [rf"\begin{{tabular}}{{{spec}}}"]
        body.append(self._top_rule)
        body.append(header)
        body.append(r"\midrule")

        if self.groups:
            for group_label, indices in self.groups:
                body.append(self._group_header(group_label))
                body.extend(self._test_rows(indices))
                body.append(r"\addlinespace[2pt]")
        else:
            body.extend(self._test_rows(list(range(len(self.tests)))))

        body.append(self._bottom_rule)
        body.append(r"\end{tabular}")
        return body

    def to_tabular(self) -> str:
        """Return only the ``\\begin{tabular}...\\end{tabular}`` block."""
        return "\n".join(self._build_body())

    def to_latex(self) -> str:
        # Put H0s in notes if not showing inline
        all_notes = list(self.notes)
        if not self.show_h0:
            for r in self.tests:
                all_notes.append(rf"\textit{{{_esc(r.test_name)}}}: H$_0$: {_esc(r.null_hypothesis)}")

        return _build_table_env(
            self._build_body(),
            caption=self.caption,
            label=self.label,
            float_position=self.float_position,
            fit_to_page=self.fit_to_page,
            font_size=self.font_size,
            notes=all_notes,
        )

    def __str__(self) -> str:
        return self.to_latex()
