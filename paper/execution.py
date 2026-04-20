"""Execute a :class:`Paper` plan: fit, render, assemble, optionally compile.

This is the engine behind :meth:`Paper.build`.  It intentionally contains
no statistical logic — all computations go through
:func:`econtools.fit.fit_model` and the :mod:`econtools.evaluation` helpers.

Public API
----------
execute_plan(paper, output_dir, compile_pdf, provenance_log) -> BuildResult
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from econtools._core.types import Estimate, TestResult
from econtools.fit import fit_model
from econtools.model.spec import ModelSpec
from econtools.output.latex.document import (
    PageSetup,
    assemble_appendix,
    assemble_document,
    write_document,
)
from econtools.output.latex.journal_profiles import JournalProfile
from econtools.output.tables.pub_latex import (
    DiagnosticsTable,
    ResultsTable,
    SummaryTable,
)
from econtools.paper.builder import (
    BuildResult,
    CompositeNode,
    FigureNode,
    Paper,
    TableNode,
)
from econtools.paper.composition import compose_appendix, compose_body
from econtools.replication.spec import ColumnSpec


# ---------------------------------------------------------------------------
# Test registry — name → callable(Estimate) → TestResult | list[TestResult]
# ---------------------------------------------------------------------------


def _test_registry() -> dict[str, Any]:
    """Return a lazy-imported dict mapping test name → function.

    Each function accepts a fitted :class:`Estimate` and returns either
    a :class:`TestResult` or a ``list[TestResult]``.  The dispatcher in
    :func:`execute_plan` flattens list returns so IV weak-instrument
    diagnostics (one row per endogenous regressor) appear as separate
    rows in the :class:`DiagnosticsTable`.
    """
    from econtools.evaluation.heteroskedasticity import (
        breusch_pagan,
        goldfeld_quandt,
        white_test,
    )
    from econtools.evaluation.iv_checks import (
        basmann_f_test,
        basmann_test,
        j_test,
        sargan_test,
        weak_instrument_tests,
        wu_hausman_test,
    )
    from econtools.evaluation.normality import (
        jarque_bera,
        omnibus_test,
        shapiro_wilk,
    )
    from econtools.evaluation.specification import reset_test
    from econtools.evaluation.serial_correlation import (
        breusch_godfrey,
        durbin_watson,
        ljung_box_q,
    )

    return {
        # Heteroskedasticity
        "breusch_pagan": breusch_pagan,
        "bp": breusch_pagan,
        "white": white_test,
        "white_test": white_test,
        "goldfeld_quandt": goldfeld_quandt,
        # Normality
        "jarque_bera": jarque_bera,
        "jb": jarque_bera,
        "omnibus": omnibus_test,
        "shapiro_wilk": shapiro_wilk,
        # Specification
        "reset": reset_test,
        "reset_test": reset_test,
        # Serial correlation
        "breusch_godfrey": breusch_godfrey,
        "durbin_watson": durbin_watson,
        "ljung_box": ljung_box_q,
        # IV diagnostics
        "sargan": sargan_test,
        "sargan_test": sargan_test,
        "basmann": basmann_test,
        "basmann_test": basmann_test,
        "basmann_f": basmann_f_test,
        "wu_hausman": wu_hausman_test,
        "wu-hausman": wu_hausman_test,
        "hausman": wu_hausman_test,
        "j_test": j_test,
        "hansen_j": j_test,
        "weak_instruments": weak_instrument_tests,
        "weak_instrument": weak_instrument_tests,
        "weak_f": weak_instrument_tests,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataframe(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix in (".feather", ".fth"):
        return pd.read_feather(path)
    raise ValueError(
        f"Unsupported data file extension '{suffix}' for {path}. "
        "Supply a .parquet, .csv, .dta, or .feather file."
    )


def _fingerprint(df: pd.DataFrame) -> str:
    """Small, stable hash of a DataFrame for the manifest."""
    try:
        buf = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    except Exception:
        # Fall back on a description for unhashable edge cases
        buf = str(df.shape).encode() + str(list(df.columns)).encode()
    return hashlib.sha1(buf).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Column → ModelSpec
# ---------------------------------------------------------------------------


_PANEL_ESTIMATORS = {"fe", "re", "fd", "pooled", "between", "first_difference"}
_DYNAMIC_PANEL_ESTIMATORS = {"arellano_bond", "ab_gmm", "dynamic_panel"}
_IV_ESTIMATORS = {"2sls", "iv2sls", "liml", "gmm", "ivgmm", "cue_gmm", "ivgmmcue"}


def _column_to_spec(col: ColumnSpec, df: pd.DataFrame) -> ModelSpec:
    """Build a :class:`ModelSpec` from a :class:`ColumnSpec`.

    When a ``cluster_var`` is supplied, the corresponding array is
    extracted from *df* and placed in ``cov_kwargs['groups']`` — both
    backends expect array-like groups, not column names.
    """
    cov_kwargs: dict[str, Any] = {}
    cluster = col.cluster_var
    estimator = col.estimator.lower()
    needs_panel = (
        estimator in _PANEL_ESTIMATORS or estimator in _DYNAMIC_PANEL_ESTIMATORS
    )

    if cluster and col.cov_type == "classical":
        cov_type = "cluster"
    else:
        cov_type = col.cov_type

    if cluster and cov_type == "cluster":
        if needs_panel:
            # For panel models, linearmodels needs a cluster Series aligned
            # with the (entity, time) MultiIndex that the adapter will build.
            needed = (
                [col.dep_var] + list(col.exog_vars) +
                [col.entity_col, col.time_col]
            )
            if cluster not in needed:
                needed.append(cluster)
            sub = df[needed].dropna()
            panel = sub.set_index([col.entity_col, col.time_col]).sort_index()
            if cluster in panel.columns:
                cluster_series = panel[cluster]
            elif cluster in panel.index.names:
                cluster_series = pd.Series(
                    panel.index.get_level_values(cluster),
                    index=panel.index,
                    name=cluster,
                )
            else:
                raise KeyError(
                    f"cluster_var '{cluster}' not found in DataFrame."
                )
            cov_kwargs["groups"] = cluster_series
        else:
            if cluster in df.columns:
                cov_kwargs["groups"] = df[cluster].values
            else:
                cov_kwargs["groups"] = cluster  # leave resolution to adapter

    # Effects default for panel FE
    effects: str | None = None
    if estimator == "fe":
        effects = "entity"

    return ModelSpec(
        dep_var=col.dep_var,
        exog_vars=list(col.exog_vars),
        endog_vars=list(col.endog_vars),
        instruments=list(col.instruments),
        entity_col=col.entity_col,
        time_col=col.time_col,
        weights_col=col.weights_var,
        effects=effects,
        estimator=estimator,
        cov_type=cov_type,
        cov_kwargs=cov_kwargs,
        add_constant=col.add_constant,
    )


def _prepare_panel_df(df: pd.DataFrame, col: ColumnSpec) -> pd.DataFrame:
    """Ensure the DataFrame is suitable for the adapter of the given column.

    The panel and dynamic-panel adapters expect entity/time to be
    **columns**, not a MultiIndex — the adapters do their own
    ``set_index`` call.  If the incoming frame was indexed by the panel
    identifiers (e.g. because a previous fit set the index), we reset it
    so the adapter can proceed.
    """
    est = col.estimator.lower()
    needs_panel_cols = (
        est in _PANEL_ESTIMATORS
        or est in _DYNAMIC_PANEL_ESTIMATORS
    )
    if not needs_panel_cols:
        return df
    if not col.entity_col or not col.time_col:
        return df
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if col.entity_col in names and col.time_col in names:
            return df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _split_system_estimate(
    system_est: Estimate,
) -> tuple[list[Estimate], list[str]]:
    """Split a SUR :class:`Estimate` into one pseudo-Estimate per equation.

    ``linearmodels.system.SUR`` returns a single flat result object with
    params indexed by ``"{equation}_{coef}"``.  Downstream rendering via
    :class:`ResultsTable` expects one :class:`Estimate` per column, so we
    splice the per-equation slice out and wrap it in an
    :class:`Estimate` whose ``raw`` points at the per-equation
    ``linearmodels`` result (preserving ``rsquared`` etc. for footers).
    """
    from econtools._core.types import FitMetrics

    raw = system_est.raw
    eq_map = raw.equations  # dict[label -> linearmodels EquationResult]

    pseudo: list[Estimate] = []
    labels: list[str] = []
    for eq_label, eq_res in eq_map.items():
        prefix = f"{eq_label}_"
        mask = [idx.startswith(prefix) for idx in system_est.params.index]
        sub_idx = [
            idx[len(prefix):]
            for idx, m in zip(system_est.params.index, mask) if m
        ]
        sub = Estimate(
            model_type="SUR",
            dep_var=str(getattr(eq_res, "dependent", eq_label)),
            params=pd.Series(system_est.params.values[mask], index=sub_idx),
            bse=pd.Series(system_est.bse.values[mask], index=sub_idx),
            tvalues=pd.Series(system_est.tvalues.values[mask], index=sub_idx),
            pvalues=pd.Series(system_est.pvalues.values[mask], index=sub_idx),
            conf_int_lower=pd.Series(
                system_est.conf_int_lower.values[mask], index=sub_idx
            ),
            conf_int_upper=pd.Series(
                system_est.conf_int_upper.values[mask], index=sub_idx
            ),
            resid=pd.Series(dtype=float),
            fitted=pd.Series(dtype=float),
            cov_params=pd.DataFrame(),
            cov_type=system_est.cov_type,
            fit=FitMetrics(
                nobs=int(getattr(eq_res, "nobs", 0)),
                df_model=float(getattr(eq_res, "df_model", float("nan"))),
                df_resid=float(getattr(eq_res, "df_resid", float("nan"))),
                r_squared=float(getattr(eq_res, "rsquared", float("nan"))),
            ),
            raw=eq_res,
        )
        pseudo.append(sub)
        labels.append(eq_label)
    return pseudo, labels


def _fit_system(
    node: TableNode,
    df: pd.DataFrame,
) -> tuple[list[Estimate], list[str], list[str] | None, Estimate]:
    """Fit the system attached to *node* and split into per-equation columns.

    Returns the list of per-equation pseudo-Estimates plus the full
    system :class:`Estimate` (used to populate cross-system footers and
    power :meth:`Paper.table_diagnostics` references keyed on
    ``"<table_id>:SYSTEM"``).
    """
    assert node.system is not None, "_fit_system requires node.system to be set"
    system_est = fit_model(node.system, df)
    per_eq, labels = _split_system_estimate(system_est)
    estimator_labels = [node.system.estimator.upper()] * len(per_eq)
    return per_eq, labels, estimator_labels, system_est


def _fit_table(
    node: TableNode,
    df: pd.DataFrame,
) -> tuple[list[Estimate], list[str], list[str] | None]:
    """Fit every column in a results-table node.

    Returns
    -------
    results, labels, estimator_labels

    If the user supplied an explicit ``label`` for every column, the
    estimator-name row is suppressed (returned as ``None``) so the LaTeX
    output does not duplicate the header.
    """
    if node.system is not None:
        per_eq, labels, est_labels, _ = _fit_system(node, df)
        return per_eq, labels, est_labels

    results: list[Estimate] = []
    labels: list[str] = []
    estimator_labels: list[str] = []
    any_label_supplied = False
    for i, col in enumerate(node.columns):
        prepared_df = _prepare_panel_df(df, col)
        spec = _column_to_spec(col, prepared_df)
        est = fit_model(spec, prepared_df)
        results.append(est)
        user_label = col.label or ""
        if user_label:
            any_label_supplied = True
            labels.append(user_label)
        else:
            labels.append(col.column_id or f"({i + 1})")
        estimator_labels.append(col.estimator.upper())

    # Suppress estimator row if the user supplied any label — the first
    # row already carries meaningful column names.  When no labels are
    # supplied at all we keep the estimator row (useful second-tier info).
    if any_label_supplied:
        return results, labels, None
    return results, labels, estimator_labels


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_summary(node: TableNode, df: pd.DataFrame) -> tuple[str, str]:
    tbl = SummaryTable(
        df=df,
        vars=node.vars,
        stats=node.stats,
        var_names=node.var_names,
        caption=node.caption or node.title or None,
        label=f"tab:{node.table_id}",
        notes=node.notes,
        font_size=None,
        fit_to_page=False,
    )
    return tbl.to_latex(), tbl.to_tabular()


def _render_results(
    node: TableNode,
    results: list[Estimate],
    labels: list[str],
    estimator_labels: list[str] | None,
    profile: "JournalProfile | None" = None,
) -> tuple[str, str]:
    footer_stats = node.footer_stats or [
        "N", "r_squared", "r_squared_adj", "cov_type"
    ]
    node_double = getattr(node, "double_rule", None)
    if node_double is None:
        double_rule = bool(getattr(profile, "table_double_rule", False)) if profile else False
    else:
        double_rule = bool(node_double)
    tbl = ResultsTable(
        results=results,
        labels=labels,
        estimator_labels=estimator_labels,
        omit_vars=node.omit_vars,
        footer_stats=footer_stats,
        caption=node.caption or node.title or None,
        label=f"tab:{node.table_id}",
        notes=node.notes,
        font_size=None,
        fit_to_page=False,
        double_rule=double_rule,
        missing_cell=getattr(node, "missing_cell", "---"),
    )
    return tbl.to_latex(), tbl.to_tabular()


def _pretty_model_ref(model: str | None) -> str | None:
    """Strip ``"table_id:"`` prefix from a model reference for user-facing text.

    ``"tab_main:OLS"`` -> ``"OLS"``; ``"OLS"`` -> ``"OLS"``; ``None`` -> ``None``.
    """
    if not model:
        return None
    return model.split(":", 1)[1] if ":" in model else model


def _render_diagnostics(
    node: TableNode,
    test_results: list[TestResult],
) -> tuple[str, str]:
    caption = node.caption or node.title
    if not caption:
        pretty = _pretty_model_ref(node.model)
        if pretty:
            caption = f"Specification diagnostics for the {pretty} regression."
        else:
            caption = "Specification diagnostics."
    tbl = DiagnosticsTable(
        tests=test_results,
        caption=caption,
        label=f"tab:{node.table_id}",
        notes=node.notes,
    )
    return tbl.to_latex(), tbl.to_tabular()


def default_figure_caption(
    figure_id: str,
    kind: str,
    *,
    model: str | None = None,
    cols: list[str] | None = None,
    groupby: str | None = None,
) -> str:
    """Return a sensible default caption for a figure with no user caption.

    The heuristic is informative enough to survive a journal-submission pass
    on a canary paper while staying short and stylistically neutral.  Model
    references, column names and groupby keys that flow in are LaTeX-escaped
    before interpolation so underscores and special characters render
    correctly.
    """
    from econtools._core.formatting import _latex_escape

    def _esc_or(value: str | None) -> str | None:
        return _latex_escape(value) if value else None

    model_clean = _esc_or(_pretty_model_ref(model))
    cols_clean = [_latex_escape(c) for c in (cols or []) if c]
    group_clean = _esc_or(groupby)

    if kind == "residuals":
        if model_clean:
            return f"Residual diagnostics for the {model_clean} regression."
        return "Residual diagnostics for the baseline regression."
    if kind == "qq":
        if model_clean:
            return f"Normal Q--Q plot of residuals from the {model_clean} regression."
        return "Normal Q--Q plot of regression residuals."
    if kind == "coef":
        if model_clean:
            return f"Coefficient forest plot for the {model_clean} regression."
        return "Coefficient forest plot for the regression."
    if kind == "series":
        if cols_clean and group_clean:
            cols_str = ", ".join(cols_clean)
            return f"Time series of {cols_str} by {group_clean}."
        if cols_clean:
            cols_str = ", ".join(cols_clean)
            return f"Time series of {cols_str}."
        return "Time series plot."
    if kind == "distribution":
        if cols_clean:
            return f"Distribution of {cols_clean[0]}."
        return "Distribution plot."
    # Fallback — keep a readable caption so LaTeX doesn't render an empty box.
    return f"Figure {_latex_escape(figure_id)}."


def _figure_fragment(
    figure_id: str,
    rel_path: str,
    caption: str | None,
    *,
    kind: str | None = None,
    title: str | None = None,
    model: str | None = None,
    cols: list[str] | None = None,
    groupby: str | None = None,
    notes: str | None = None,
) -> str:
    """Build a ``\\begin{figure}`` fragment referencing a saved image.

    Caption resolution (in priority order):
      1. Explicit ``caption`` supplied by the user — used verbatim.
      2. Explicit ``title`` supplied by the user — used verbatim.
      3. Generated default based on ``kind`` / ``model`` / ``cols``.

    Always emits a ``\\caption{...}`` so LaTeX numbers the float correctly.
    A ``\\label`` must follow the caption for ``\\ref{fig:...}`` to resolve
    to the expected number.
    """
    if caption:
        caption_text = caption
    elif title:
        caption_text = title
    else:
        caption_text = default_figure_caption(
            figure_id,
            kind or "",
            model=model,
            cols=cols,
            groupby=groupby,
        )
    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        rf"\includegraphics[width=0.9\textwidth]{{{rel_path}}}",
        rf"\caption{{{caption_text}}}",
        rf"\label{{fig:{figure_id}}}",
    ]
    if notes:
        lines.extend([
            r"\begin{minipage}{0.9\textwidth}\footnotesize",
            r"\vspace{0.4em}",
            rf"\textit{{Notes:}} {notes}",
            r"\end{minipage}",
        ])
    lines.append(r"\end{figure}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Composite float rendering
# ---------------------------------------------------------------------------


def _render_composite(
    cnode: CompositeNode,
    paper: Paper,
    bare_fragments: dict[str, str],
    figure_paths: dict[str, Path],
) -> str:
    """Pack registered table/figure panels into a single multi-panel float.

    Thin delegator to :mod:`econtools.output.composite`.  The composite
    module owns the geometry: stacked mode renders one outer tabular
    with a shared LCM column grid, grid mode renders a rigid ``p x q``
    lattice whose horizontal rules are all owned by the outer tabular
    (panel bodies contribute no booktabs rules of their own).
    """
    from econtools.output.composite import render_composite as _impl
    return _impl(cnode, paper, bare_fragments, figure_paths)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def _residual_panel_fig(result: Estimate) -> Figure:
    """Two-panel figure: residuals-vs-fitted and Q-Q."""
    from econtools.plots.residual_plots import (
        plot_qq,
        plot_residuals_vs_fitted,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_residuals_vs_fitted(result, ax=axes[0])
    plot_qq(result, ax=axes[1])
    fig.tight_layout()
    return fig


def _coef_fig(result: Estimate) -> Figure:
    from econtools.plots.coefficient_plots import plot_coef_forest
    return plot_coef_forest(result)


def _series_fig(
    df: pd.DataFrame,
    cols: list[str],
    *,
    groupby: str | None = None,
    time_col: str | None = None,
    title: str | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    working = df
    if isinstance(df.index, pd.MultiIndex) and groupby in df.index.names:
        working = df.reset_index()
    if groupby and groupby in working.columns:
        for key, group in working.groupby(groupby):
            x = group[time_col].values if time_col and time_col in group.columns else range(len(group))
            for col in cols:
                if col in group.columns:
                    ax.plot(x, group[col].values, alpha=0.4, linewidth=0.8)
        # Overlay mean across groups by time if time_col known
        if time_col and time_col in working.columns and cols:
            mean_by_time = working.groupby(time_col)[cols[0]].mean()
            ax.plot(
                mean_by_time.index,
                mean_by_time.values,
                color="black",
                linewidth=2.2,
                label=f"Mean {cols[0]}",
            )
            ax.legend(frameon=False, loc="best")
    else:
        x = working[time_col].values if time_col and time_col in working.columns else range(len(working))
        for col in cols:
            if col in working.columns:
                ax.plot(x, working[col].values, label=col, linewidth=1.5)
        ax.legend(frameon=False, loc="best")
    ax.set_xlabel(time_col or "Time")
    ax.set_ylabel(", ".join(cols))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _distribution_fig(df: pd.DataFrame, col: str, title: str | None) -> Figure:
    from econtools.plots.time_series import plot_distribution
    return plot_distribution(df[col].dropna(), title=title, xlabel=col)


def _scatter_fit_fig(
    df: pd.DataFrame,
    panels: list[dict[str, Any]],
    *,
    layout: tuple[int, int] | None,
    weight_col: str | None,
    cov_type: str,
    hspace: float | None,
    wspace: float | None,
    figsize: tuple[float, float] | None,
    panel_style=None,
) -> Figure:
    """Multi-panel scatter + fitted regression line figure."""
    from econtools.output.style import DEFAULT_STYLE
    from econtools.plots.panel_figure import panel_figure
    from econtools.plots.regression_scatter import scatter_with_fit

    ps = panel_style or DEFAULT_STYLE.panel

    panel_callables = []
    labels: list[str] = []
    for panel in panels:
        p = dict(panel)
        labels.append(p.get("label", ""))

        def _draw(ax, _p=p):
            scatter_with_fit(
                ax, df,
                x=_p["x"], y=_p["y"],
                weight_col=weight_col,
                cov_type=cov_type,
                annotate=_p.get("annotate", True),
                annotate_loc=_p.get("annotate_loc", "lower right"),
                xlabel=_p.get("xlabel"),
                ylabel=_p.get("ylabel"),
                title=_p.get("subtitle"),
            )

        panel_callables.append(_draw)

    n = len(panel_callables)
    if layout is None:
        layout = (n, 1) if n <= 3 else (2, (n + 1) // 2)
    nrows, ncols = layout
    if figsize is None:
        figsize = (
            ps.per_panel_width * ncols,
            ps.per_panel_height * nrows + ps.figsize_row_pad,
        )
    if hspace is None and nrows > 1:
        hspace = ps.hspace
    if wspace is None and ncols > 1:
        wspace = ps.wspace

    fig = panel_figure(
        panel_callables,
        layout=layout,
        figsize=figsize,
        panel_labels=ps.panel_labels,
        hspace=hspace,
        wspace=wspace,
        group_frame=ps.group_frame,
        frame_linewidth=ps.frame_linewidth,
        frame_color=ps.frame_color,
    )
    # Prepend "Panel A." etc. labels above each subplot title.
    for ax, label in zip(fig.axes, labels):
        if label:
            prev = ax.get_title()
            ax.set_title("")
            ax.set_title(
                f"{label}\n{prev}" if prev else label,
                fontsize=9.5, pad=10, loc="center",
            )
    return fig


# ---------------------------------------------------------------------------
# LaTeX compilation helpers
# ---------------------------------------------------------------------------


def _try_compile(tex_path: Path) -> tuple[Path | None, str]:
    """Compile *tex_path*; return ``(pdf_path_or_None, failure_reason)``.

    ``failure_reason`` is empty on success, ``"no_engine"`` when no LaTeX
    engine is installed, and ``"compile_failed"`` when the engine ran but
    did not produce a PDF (missing class, syntax error, etc.).
    """
    try:
        from econtools.tables.latex_utils import compile_tex_to_pdf
        return compile_tex_to_pdf(tex_path), ""
    except RuntimeError as err:
        msg = str(err)
        if "engine" in msg.lower() and "found" in msg.lower():
            print(f"[paper] LaTeX compilation skipped: {err}")
            return None, "no_engine"
        print(f"[paper] LaTeX compilation failed: {err}")
        return None, "compile_failed"
    except Exception as err:  # pragma: no cover — defensive
        print(f"[paper] LaTeX compilation failed: {err}")
        return None, "compile_failed"


def _looks_like_missing_class(tex_path: Path, class_name: str) -> bool:
    """Heuristic: inspect the LaTeX log for a missing-``.cls`` error.

    Returns ``True`` when the log references ``<class_name>.cls`` in a
    "not found" context. Used to distinguish a missing journal class
    from genuine user syntax errors.
    """
    log_path = tex_path.with_suffix(".log")
    if not log_path.exists():
        return False
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover — defensive
        return False
    needle = f"{class_name}.cls"
    if needle not in log:
        return False
    markers = [
        f"{class_name}.cls' not found",
        f"File `{class_name}.cls'",
        f"LaTeX Error: File `{class_name}.cls'",
        "Can't find file",
    ]
    return any(m in log for m in markers)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def execute_plan(
    paper: Paper,
    output_dir: Path,
    *,
    compile_pdf: bool = True,
    provenance_log: Path | None = None,
) -> BuildResult:
    """Execute *paper* and return a :class:`BuildResult`.

    See :meth:`Paper.build` for the user-facing entry point.
    """
    # Make sure any unsnapshotted transforms get attached to the current dataset
    paper._snapshot_transform()

    fig_dir = output_dir / "figures"
    tab_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- #
    # Step 1: load datasets                                          #
    # -------------------------------------------------------------- #
    datasets: dict[str, pd.DataFrame] = {}
    data_info: dict[str, dict[str, Any]] = {}
    for name, node in paper.loads.items():
        if node.df is not None:
            df = node.df.copy()
            source = "in-memory"
        else:
            df = _load_dataframe(node.path)
            source = str(node.path)
        datasets[name] = df
        data_info[name] = {
            "source": source,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "fingerprint": _fingerprint(df),
            "entity_col": node.entity_col,
            "time_col": node.time_col,
        }

    # -------------------------------------------------------------- #
    # Step 2: apply transforms                                       #
    # -------------------------------------------------------------- #
    for tnode in paper.transforms:
        if tnode.dataset not in datasets:
            raise KeyError(
                f"Transform targets unknown dataset '{tnode.dataset}'"
            )
        entity = paper.loads[tnode.dataset].entity_col
        datasets[tnode.dataset] = tnode.transformer.apply(
            datasets[tnode.dataset],
            entity=entity,
            log_path=provenance_log,
        )

    # -------------------------------------------------------------- #
    # Step 3: fit models needed by results tables                    #
    # -------------------------------------------------------------- #
    fitted: dict[tuple[str, str], Estimate] = {}  # (table_id, col_label) → Estimate
    fitted_metadata: dict[str, dict[str, Any]] = {}

    for table_id in paper.table_ids():
        node = paper.get_table(table_id)
        if node.kind != "results":
            continue
        df = datasets[node.dataset]
        if node.system is not None:
            per_eq, labels, _, system_est = _fit_system(node, df)
            # Per-equation Estimates are keyed like column Estimates.
            for label, est in zip(labels, per_eq):
                fitted[(table_id, label)] = est
                eq = node.system.equations[label]
                fitted_metadata[f"{table_id}:{label}"] = {
                    "estimator": node.system.estimator,
                    "dep_var": eq.dep_var,
                    "exog_vars": list(eq.exog_vars),
                    "nobs": int(est.fit.nobs),
                    "cov_type": est.cov_type,
                }
            # Expose the full system result under "SYSTEM" for cross-equation
            # tests (Breusch-Pagan LM, system Wald, etc.).
            fitted[(table_id, "SYSTEM")] = system_est
            fitted_metadata[f"{table_id}:SYSTEM"] = {
                "estimator": node.system.estimator,
                "dep_var": system_est.dep_var,
                "exog_vars": [],
                "nobs": int(system_est.fit.nobs),
                "cov_type": system_est.cov_type,
            }
        else:
            results, labels, _ = _fit_table(node, df)
            for col, label, est in zip(node.columns, labels, results):
                fitted[(table_id, label)] = est
                fitted_metadata[f"{table_id}:{label}"] = {
                    "estimator": col.estimator,
                    "dep_var": col.dep_var,
                    "exog_vars": list(col.exog_vars),
                    "nobs": int(est.fit.nobs),
                    "cov_type": est.cov_type,
                }

    # -------------------------------------------------------------- #
    # Step 4: render tables                                          #
    # -------------------------------------------------------------- #
    fragments: dict[str, str] = {}
    bare_fragments: dict[str, str] = {}
    table_paths: dict[str, Path] = {}

    for table_id in paper.table_ids():
        node = paper.get_table(table_id)
        bare: str | None = None
        if node.kind == "summary":
            df = datasets[node.dataset]
            frag, bare = _render_summary(node, df)
        elif node.kind == "results":
            df = datasets[node.dataset]
            results, labels, estlabels = _fit_table(node, df)
            frag, bare = _render_results(node, results, labels, estlabels, paper.profile)
        elif node.kind == "diagnostics":
            if not node.model:
                raise ValueError(
                    f"Diagnostics table '{table_id}' requires model ref"
                )
            tid, lbl = paper.resolve_model_ref(node.model)
            if (tid, lbl) not in fitted:
                raise KeyError(
                    f"Model {node.model} has not been fitted. Diagnostics "
                    "tables must reference columns of an earlier results table."
                )
            est = fitted[(tid, lbl)]
            registry = _test_registry()
            test_results: list[TestResult] = []
            for tname in node.tests:
                key = tname.lower()
                fn = registry.get(key)
                if fn is None:
                    # Skip silently rather than crash the build
                    continue
                try:
                    out = fn(est)
                except Exception as err:  # pragma: no cover — test-specific
                    print(f"[paper] Test '{tname}' failed on {node.model}: {err}")
                    continue
                # Allow callables that return list[TestResult] (e.g. the
                # per-endogenous-regressor weak-instrument fan-out).
                if isinstance(out, TestResult):
                    test_results.append(out)
                elif isinstance(out, list):
                    test_results.extend(
                        x for x in out if isinstance(x, TestResult)
                    )
                else:  # pragma: no cover — defensive
                    print(
                        f"[paper] Test '{tname}' returned unexpected type "
                        f"{type(out).__name__}; skipping."
                    )
            if not test_results:
                # Emit a placeholder fragment so downstream references still work
                frag = rf"% Diagnostics table '{table_id}' has no applicable tests."
                bare = None
            else:
                frag, bare = _render_diagnostics(node, test_results)
        else:  # pragma: no cover — guarded at registration
            raise ValueError(f"Unknown table kind '{node.kind}'")

        fragments[table_id] = frag
        if bare is not None:
            bare_fragments[table_id] = bare
        tab_path = tab_dir / f"{table_id}.tex"
        tab_path.write_text(frag, encoding="utf-8")
        table_paths[table_id] = tab_path

    # -------------------------------------------------------------- #
    # Step 5: render figures                                         #
    # -------------------------------------------------------------- #
    figure_paths: dict[str, Path] = {}

    for figure_id in paper.figure_ids():
        node = paper.get_figure(figure_id)
        fig_path = fig_dir / f"{figure_id}.pdf"
        try:
            if node.kind == "residuals":
                if not node.model:
                    raise ValueError("Residuals figure requires model ref")
                tid, lbl = paper.resolve_model_ref(node.model)
                est = fitted[(tid, lbl)]
                fig = _residual_panel_fig(est)
            elif node.kind == "coef":
                if not node.model:
                    raise ValueError("Coef figure requires model ref")
                tid, lbl = paper.resolve_model_ref(node.model)
                est = fitted[(tid, lbl)]
                fig = _coef_fig(est)
            elif node.kind == "series":
                df = datasets[node.dataset]
                time_col = node.time_col or paper.loads[node.dataset].time_col
                fig = _series_fig(
                    df,
                    node.cols,
                    groupby=node.groupby,
                    time_col=time_col,
                    title=node.title or None,
                )
            elif node.kind == "distribution":
                df = datasets[node.dataset]
                fig = _distribution_fig(df, node.cols[0], node.title or None)
            elif node.kind == "scatter_fit":
                df = datasets[node.dataset]
                fig = _scatter_fit_fig(
                    df,
                    node.panels,
                    layout=node.layout,
                    weight_col=node.weight_col,
                    cov_type=node.cov_type,
                    hspace=node.hspace,
                    wspace=node.wspace,
                    figsize=node.figsize,
                    panel_style=getattr(paper, "style", None) and paper.style.panel,
                )
            else:  # pragma: no cover — guarded at registration
                raise ValueError(f"Unknown figure kind '{node.kind}'")

            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            figure_paths[figure_id] = fig_path
            rel = (Path("figures") / f"{figure_id}.pdf").as_posix()
            fragments[figure_id] = _figure_fragment(
                figure_id,
                rel,
                node.caption or None,
                kind=node.kind,
                title=node.title or None,
                model=node.model,
                cols=list(node.cols) if node.cols else None,
                groupby=node.groupby,
                notes=node.notes,
            )
        except Exception as err:
            print(f"[paper] Figure '{figure_id}' failed: {err}")
            # Emit a comment fragment so section anchors remain valid
            fragments[figure_id] = (
                rf"% Figure '{figure_id}' could not be rendered: {err}"
            )

    # -------------------------------------------------------------- #
    # Step 5b: composite multi-panel floats                          #
    # -------------------------------------------------------------- #
    for cid in paper.composite_ids():
        cnode = paper.get_composite(cid)
        try:
            frag = _render_composite(cnode, paper, bare_fragments, figure_paths)
        except Exception as err:
            print(f"[paper] Composite '{cid}' failed: {err}")
            fragments[cid] = rf"% Composite '{cid}' could not be rendered: {err}"
            continue
        fragments[cid] = frag
        # Suppress individual panels so they aren't emitted standalone too
        for pid in cnode.panels:
            fragments.pop(pid, None)

    # -------------------------------------------------------------- #
    # Step 6: compose and write .tex                                 #
    # -------------------------------------------------------------- #
    body_fragments = compose_body(paper, fragments)
    appendix_fragments = compose_appendix(paper, fragments)
    if appendix_fragments:
        body_fragments = body_fragments + [
            r"\appendix",
            _make_appendix_block(appendix_fragments),
        ]

    author_str = ", ".join(paper.authors) if paper.authors else None
    page_setup = PageSetup(paper="a4paper", margin="1in")

    tex_path = output_dir / "paper.tex"
    write_document(
        body_fragments,
        tex_path,
        paper.profile,
        title=paper.title,
        author=author_str,
        abstract=paper.abstract or None,
        page_setup=page_setup,
    )

    # -------------------------------------------------------------- #
    # Step 7: compile to PDF (optional, with graceful fallback)      #
    # -------------------------------------------------------------- #
    pdf_path: Path | None = None
    used_fallback = False
    if compile_pdf:
        pdf_path, reason = _try_compile(tex_path)
        if pdf_path is None and reason == "compile_failed":
            # Retry with the profile's fallback class iff the failure looks
            # like a missing-.cls error (common for journal styles).  For
            # other failures we leave the original .tex in place so users
            # can inspect the log.
            primary_cls = paper.profile.document_class
            if (
                primary_cls not in {"article", "report", "book"}
                and _looks_like_missing_class(tex_path, primary_cls)
            ):
                print(
                    f"[paper] Class '{primary_cls}.cls' not found — "
                    f"retrying with fallback '{paper.profile.fallback_class}'."
                )
                fallback = _fallback_profile(paper.profile)
                write_document(
                    body_fragments,
                    tex_path,
                    fallback,
                    title=paper.title,
                    author=author_str,
                    abstract=paper.abstract or None,
                    page_setup=page_setup,
                )
                pdf_path, _ = _try_compile(tex_path)
                used_fallback = True

    # -------------------------------------------------------------- #
    # Step 8: manifest                                               #
    # -------------------------------------------------------------- #
    manifest = {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "profile": paper.profile.name,
        "document_class": paper.profile.document_class,
        "used_fallback": used_fallback,
        "datasets": data_info,
        "tables": {
            tid: {
                "kind": paper.get_table(tid).kind,
                "tex": str(table_paths[tid]),
            }
            for tid in paper.table_ids()
        },
        "figures": {
            fid: str(figure_paths[fid])
            for fid in paper.figure_ids()
            if fid in figure_paths
        },
        "models": fitted_metadata,
        "appendix_ids": paper.appendix_ids,
        "tex": str(tex_path),
        "pdf": str(pdf_path) if pdf_path else None,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return BuildResult(
        tex_path=tex_path.resolve(),
        pdf_path=pdf_path.resolve() if pdf_path else None,
        figures={fid: p.resolve() for fid, p in figure_paths.items()},
        tables={tid: p.resolve() for tid, p in table_paths.items()},
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_appendix_block(fragments: list[str]) -> str:
    """Pack appendix fragments with page breaks between them."""
    out: list[str] = []
    for i, frag in enumerate(fragments):
        if i > 0:
            out.append(r"\clearpage")
        out.append(frag)
    return "\n".join(out)


def _fallback_profile(profile):
    """Return a variant of *profile* using its declared fallback class.

    Honours ``profile.fallback_class`` / ``profile.fallback_options`` when
    present (new-style profiles); older profiles without those fields
    collapse to ``article`` with ``11pt``.  Packages and preamble hooks
    that only make sense with the original class are stripped to
    minimise the chance the fallback itself fails to compile.
    """
    from econtools.output.latex.journal_profiles import JournalProfile
    fallback_class = getattr(profile, "fallback_class", "article") or "article"
    fallback_options = list(getattr(profile, "fallback_options", ["11pt"]) or ["11pt"])
    return JournalProfile(
        name=profile.name,
        document_class=fallback_class,
        class_options=fallback_options,
        table_numbering=profile.table_numbering,
        caption_position=profile.caption_position,
        notes_command=profile.notes_command,
        float_specifier=profile.float_specifier,
        star_symbols=list(profile.star_symbols),
        significance_levels=list(profile.significance_levels),
        bst_file=profile.bst_file,
        packages=[],            # drop journal-specific preamble
        fallback_class=fallback_class,
        fallback_options=fallback_options,
        preamble_hooks=[],      # drop journal-specific hooks
        notes=getattr(profile, "notes", ""),
        table_double_rule=getattr(profile, "table_double_rule", False),
        caption_labelfont=getattr(profile, "caption_labelfont", "bf"),
        caption_textfont=getattr(profile, "caption_textfont", ""),
    )
