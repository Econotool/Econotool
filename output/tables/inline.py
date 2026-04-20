"""Thin helpers that produce bare tabulars ready to slot into a composite.

The classes in :mod:`econtools.output.tables.pub_latex` expose a full
``to_latex()`` method that wraps the body in a floating table
environment.  The helpers here return only the ``\\begin{tabular}...
\\end{tabular}`` block so the caller can feed it to a stacked
:class:`~econtools.output.composite.model.Composite`.

The goal is to let a pipeline script stay short and declarative — no
ad-hoc formatting helpers in the runner — while still exposing the
full knobs of the underlying classes.

Public helpers
--------------
``results_panel``       — ``ResultsTable.to_tabular()`` façade
``summary_panel``       — ``SummaryTable.to_tabular()`` façade
``info_panel``          — ``InfoTable.to_tabular()`` façade
``diagnostics_panel``   — runs one or more named diagnostic tests on a
                          fitted model and returns a formatted tabular
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from econtools._core.types import Estimate, TestResult
from econtools.output.tables.pub_latex import (
    InfoTable,
    ResultsTable,
    SummaryTable,
)


# ---------------------------------------------------------------------------
# Thin wrappers over the three table classes
# ---------------------------------------------------------------------------


def results_panel(
    results: list[Estimate],
    *,
    labels: list[str] | None = None,
    estimator_labels: list[str] | None = None,
    variable_names: dict[str, str] | None = None,
    keep_vars: list[str] | None = None,
    omit_vars: list[str] | None = None,
    panels: list[tuple[str, list[str]]] | None = None,
    footer_stats: list[str | tuple] | None = None,
    bootstrap_ses: dict[str, list[float]] | None = None,
    bootstrap_se_label: str = "",
    bootstrap_se_delimiter: str = "brackets",
    digits: int = 3,
    stars: bool = True,
    se_delimiter: str = "parens",
    missing_cell: str = "---",
) -> str:
    """Return a bare results tabular suitable for a composite panel.

    Thin wrapper over :class:`ResultsTable.to_tabular` that forwards
    every knob the pipeline is likely to override.  When the pipeline
    wants stars suppressed or different footer stats it should pass
    those explicitly — the wrapper adds no defaults of its own.
    """
    table = ResultsTable(
        results=results,
        labels=labels,
        estimator_labels=estimator_labels,
        variable_names=variable_names,
        keep_vars=keep_vars,
        omit_vars=omit_vars,
        panels=panels,
        footer_stats=footer_stats,
        bootstrap_ses=bootstrap_ses,
        bootstrap_se_label=bootstrap_se_label,
        bootstrap_se_delimiter=bootstrap_se_delimiter,
        digits=digits,
        stars=stars,
        se_delimiter=se_delimiter,
        missing_cell=missing_cell,
        add_star_note=False,  # notes live on the composite, not the panel
    )
    return table.to_tabular()


def summary_panel(
    df: pd.DataFrame,
    *,
    vars: list[str],
    stats: list[str] = ("mean", "std", "min", "median", "max", "N"),
    var_names: dict[str, str] | None = None,
    descriptions: dict[str, str] | None = None,
    description_width: str = "5cm",
    description_label: str = "Description",
    panels: list[tuple[str, list[str]]] | None = None,
    digits: int = 3,
) -> str:
    """Return a bare summary-statistics tabular for a composite panel."""
    table = SummaryTable(
        df=df,
        vars=vars,
        stats=list(stats),
        var_names=var_names,
        descriptions=descriptions,
        description_width=description_width,
        description_label=description_label,
        panels=panels,
        digits=digits,
    )
    return table.to_tabular()


def info_panel(
    rows: list[dict],
    *,
    columns: list[dict],
    panels: list[tuple[str, list[int]]] | None = None,
) -> str:
    """Return a bare info-table tabular for a composite panel."""
    table = InfoTable(rows=rows, columns=columns, panels=panels)
    return table.to_tabular()


# ---------------------------------------------------------------------------
# Diagnostics panel
# ---------------------------------------------------------------------------


def _fmt_p(pval: float) -> str:
    if pval is None or math.isnan(float(pval)):
        return "---"
    return f"{pval:.3f}" if pval >= 0.001 else r"$<$0.001"


def _decision(tr: TestResult) -> str:
    """Rejection decision string for a TestResult at the test's own alpha."""
    if tr.pvalue is None or math.isnan(float(tr.pvalue)):
        return "---"
    return r"Reject $H_0$" if tr.reject else r"Fail to reject"


def _strong_weak_decision(f_stat: float, threshold: float = 10.0) -> str:
    """Stock-Yogo rule of thumb for first-stage F."""
    if f_stat is None or math.isnan(float(f_stat)):
        return "---"
    return "Strong" if f_stat > threshold else "Weak"


def _format_stat(stat: float, digits: int) -> str:
    if stat is None:
        return "---"
    try:
        fv = float(stat)
    except (TypeError, ValueError):
        return "---"
    if math.isnan(fv):
        return "---"
    return f"{fv:.{digits}f}"


# Test keys → ``(display name, callable)``.  The callable takes whatever
# the panel harness passes in (either a single Estimate or a mapping of
# helper models) and returns a :class:`TestResult` — except the weak-
# instrument entry which returns a list and is handled specially.
def _noarg(fn: Callable) -> Callable[..., Any]:
    return lambda result, **_: fn(result)


def _with_kwargs(fn: Callable) -> Callable[..., Any]:
    return lambda result, **kwargs: fn(result, **kwargs)


def _weak_instr(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.iv import weak_instrument_tests
    rows = weak_instrument_tests(result)
    # Return the minimum-F row (most conservative weak-instrument reading).
    return min(rows, key=lambda r: r.statistic)


def _sargan(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.iv import sargan_test
    return sargan_test(result)


def _basmann(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.iv import basmann_test
    return basmann_test(result)


def _wu_hausman(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.iv import wu_hausman_test
    return wu_hausman_test(result)


def _reset(result: Estimate, **kwargs) -> TestResult:
    from econtools.diagnostics.specification import reset_test
    return reset_test(result, **kwargs)


def _breusch_pagan(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.heteroskedasticity import breusch_pagan
    return breusch_pagan(result)


def _white(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.heteroskedasticity import white_test
    return white_test(result)


def _jarque_bera(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.normality import jarque_bera
    return jarque_bera(result)


def _durbin_watson(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.serial_correlation import durbin_watson
    return durbin_watson(result)


def _breusch_godfrey(result: Estimate, **kwargs) -> TestResult:
    from econtools.diagnostics.serial_correlation import breusch_godfrey
    return breusch_godfrey(result, **kwargs)


def _harvey_collier(result: Estimate, **_) -> TestResult:
    from econtools.diagnostics.specification import harvey_collier
    return harvey_collier(result)


_TEST_REGISTRY: dict[str, tuple[str, Callable[..., TestResult]]] = {
    "weak_instruments": (r"Weak instrument $F$ (min.)", _weak_instr),
    "sargan":           (r"Sargan overid.\ ($H_0$: IVs valid)", _sargan),
    "basmann":          (r"Basmann overid.\ ($H_0$: IVs valid)", _basmann),
    "wu_hausman":       (r"Wu-Hausman ($H_0$: endog. exog.)", _wu_hausman),
    "reset":            (r"Ramsey RESET ($H_0$: linear)", _reset),
    "breusch_pagan":    (r"Breusch-Pagan ($H_0$: homosk.)", _breusch_pagan),
    "white":            (r"White ($H_0$: homosk.)", _white),
    "jarque_bera":      (r"Jarque-Bera ($H_0$: normal)", _jarque_bera),
    "durbin_watson":    (r"Durbin-Watson", _durbin_watson),
    "breusch_godfrey":  (r"Breusch-Godfrey ($H_0$: no SC)", _breusch_godfrey),
    "harvey_collier":   (r"Harvey-Collier ($H_0$: linear)", _harvey_collier),
}


def diagnostics_panel(
    model: Estimate,
    *,
    tests: list[str | tuple[str, Any]],
    digits: int = 2,
    include_decision: bool = True,
    weak_instr_threshold: float = 10.0,
    kwargs_per_test: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Run a list of named diagnostics on *model* and return a bare tabular.

    Parameters
    ----------
    model:
        The fitted :class:`~econtools._core.types.Estimate`.
    tests:
        Ordered list of test keys from :data:`_TEST_REGISTRY` (or tuples
        ``(display_name, callable)`` for ad-hoc rows, where the callable
        takes *model* and returns a :class:`TestResult`).
    digits:
        Decimal places for the statistic column.
    include_decision:
        When True, adds a "Decision" column (reject/fail-to-reject; the
        ``weak_instruments`` row uses Strong/Weak at ``F > 10``).
    weak_instr_threshold:
        Stock-Yogo rule-of-thumb threshold for the Strong/Weak call.
    kwargs_per_test:
        Optional ``{test_key: {kwarg: value}}`` — forwarded to the test
        callable (e.g. ``{"reset": {"power": 3}}``).
    """
    kwargs_per_test = kwargs_per_test or {}

    if include_decision:
        header = (
            r"Test & \multicolumn{1}{c}{Statistic} "
            r"& \multicolumn{1}{c}{$p$-value} "
            r"& \multicolumn{1}{c}{Decision} \\"
        )
        col_spec = "l rrl"
    else:
        header = (
            r"Test & \multicolumn{1}{c}{Statistic} "
            r"& \multicolumn{1}{c}{$p$-value} \\"
        )
        col_spec = "l rr"

    rows: list[str] = []
    for entry in tests:
        if isinstance(entry, tuple):
            display_name, callable_ = entry
            tr = callable_(model)
            key = None
        else:
            key = entry
            if key not in _TEST_REGISTRY:
                raise ValueError(
                    f"Unknown diagnostic test '{key}'.  "
                    f"Available: {sorted(_TEST_REGISTRY)}"
                )
            display_name, callable_ = _TEST_REGISTRY[key]
            extra = kwargs_per_test.get(key, {})
            tr = callable_(model, **extra)

        stat_str = _format_stat(tr.statistic, digits)
        pval_str = _fmt_p(tr.pvalue)

        row_cells = [display_name, stat_str, pval_str]
        if include_decision:
            if key == "weak_instruments":
                decision = _strong_weak_decision(
                    tr.statistic, threshold=weak_instr_threshold
                )
            else:
                decision = _decision(tr)
            row_cells.append(decision)

        rows.append(" & ".join(row_cells) + r" \\")

    return (
        rf"\begin{{tabular}}{{{col_spec}}}" + "\n"
        r"\toprule" + "\n"
        + header + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}"
    )
