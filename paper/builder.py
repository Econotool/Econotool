"""Fluent-builder ``Paper`` orchestrator.

A :class:`Paper` composes a DAG of plan nodes (data loads, transforms,
fits, tables, figures, sections) that is executed lazily by
:func:`econtools.paper.execution.execute_plan` when :meth:`Paper.build`
is called.

The API is intentionally lock-stepped with the spec in the project brief
— deviating callers should use the lower-level ``econtools.fit`` and
``econtools.output`` layers directly.

Public API
----------
Paper                   — the orchestrator
DataLoadNode, TransformNode, TableNode, FigureNode, SectionNode — plan nodes
BuildResult             — value returned by :meth:`Paper.build`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from econtools.output.style import StyleConfig

import pandas as pd

from econtools.model.spec import EquationSpec, SystemSpec
from econtools.output.latex.journal_profiles import (
    AER,
    DEFAULT,
    ECONOMETRICA,
    ECTA,
    JournalProfile,
    QE,
    TE,
    resolve_profile as _resolve_profile_registry,
)
from econtools.paper.transforms import DataTransformer
from econtools.replication.spec import ColumnSpec


# ---------------------------------------------------------------------------
# Plan nodes
# ---------------------------------------------------------------------------


@dataclass
class DataLoadNode:
    """Load a dataset from disk (parquet / csv / dta) or use a raw DataFrame."""

    name: str                       # dataset key (e.g. "main")
    path: str | None = None         # path to parquet/csv/dta; None ⇒ use ``df``
    df: pd.DataFrame | None = None  # pre-loaded DataFrame
    entity_col: str | None = None   # default panel entity column
    time_col: str | None = None     # default panel time column


@dataclass
class TransformNode:
    """Apply a :class:`DataTransformer` to a named dataset."""

    dataset: str
    transformer: DataTransformer


@dataclass
class TableNode:
    """A table node.  ``kind`` controls which renderer is used."""

    table_id: str
    kind: str                        # "summary" | "results" | "diagnostics"
    title: str = ""
    caption: str | None = None
    dataset: str = "main"
    notes: list[str] = field(default_factory=list)

    # Summary-table payload
    vars: list[str] | None = None
    stats: list[str] = field(
        default_factory=lambda: ["mean", "std", "min", "max", "N"]
    )
    var_names: dict[str, str] | None = None

    # Results-table payload
    columns: list[ColumnSpec] = field(default_factory=list)
    footer_stats: list[str] | None = None
    omit_vars: list[str] | None = None
    # System-of-equations payload (SUR / 3SLS).  When non-None, the
    # ``columns`` list is ignored and the table is rendered one column
    # per equation.
    system: SystemSpec | None = None

    # Diagnostics-table payload
    model: str | None = None          # "<table_id>:<col_label>"
    tests: list[str] = field(default_factory=list)


@dataclass
class FigureNode:
    """A figure node.  ``kind`` controls which plot helper is used."""

    figure_id: str
    kind: str                         # "residuals" | "qq" | "coef" | "series" | "distribution" | "scatter_fit"
    title: str = ""
    caption: str | None = None
    notes: str | None = None

    # Common: model reference "<table_id>:<col_label>" (used by residuals, qq, coef)
    model: str | None = None

    # Series plot
    dataset: str = "main"
    cols: list[str] = field(default_factory=list)
    groupby: str | None = None
    time_col: str | None = None

    # scatter_fit plot: list of {x, y, label, xlabel, ylabel, subtitle, annotate_loc}
    panels: list[dict[str, Any]] = field(default_factory=list)
    layout: tuple[int, int] | None = None
    weight_col: str | None = None
    cov_type: str = "HC1"
    hspace: float | None = None
    wspace: float | None = None
    figsize: tuple[float, float] | None = None


@dataclass
class CompositeNode:
    """A composite float that packs multiple tables / figures into one float.

    Each entry in ``panels`` is the id of an already-registered
    :class:`TableNode` or :class:`FigureNode`.  The execution layer
    renders each panel in "bare" form (tabular body / includegraphics)
    and wraps the set in a single ``figure`` float with a master
    caption.  Panels keep their own sub-captions via
    ``\\subcaption{...}``.
    """

    composite_id: str
    panels: list[str]
    layout: tuple[int, int] | None = None
    caption: str | None = None
    notes: str | None = None
    subcaptions: list[str] | None = None
    float_kind: str = "figure"            # "figure" | "table"
    column_fraction: float | None = None  # override default panel width


@dataclass
class SectionNode:
    """Narrative section with anchor hooks for placing tables and figures."""

    title: str
    body: str = ""
    after: list[str] = field(default_factory=list)   # ids of table/figure nodes
    appendix: bool = False                            # put in appendix doc


# ---------------------------------------------------------------------------
# Build result
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    """Summary of :meth:`Paper.build`.

    Attributes
    ----------
    tex_path:
        Absolute path to the generated ``paper.tex``.
    pdf_path:
        Absolute path to ``paper.pdf`` when compilation succeeded,
        otherwise ``None``.
    figures:
        Mapping ``figure_id -> absolute path`` for every emitted figure.
    tables:
        Mapping ``table_id -> .tex file path`` for standalone fragments.
    manifest_path:
        Absolute path to ``manifest.json``.
    manifest:
        The manifest contents as a plain dict.
    """

    tex_path: Path
    pdf_path: Path | None
    figures: dict[str, Path]
    tables: dict[str, Path]
    manifest_path: Path
    manifest: dict[str, Any]


# ---------------------------------------------------------------------------
# Journal profile resolution
# ---------------------------------------------------------------------------

# Extra builder-local aliases on top of the canonical registry in
# econtools.output.latex.journal_profiles.  The canonical registry is
# the source of truth; the map below only adds backward-compat entries
# that predate the new profiles.
_PROFILE_MAP: dict[str, JournalProfile] = {
    "ecta": ECTA,
    "econometrica": ECONOMETRICA,
    "qe": QE,
    "quantitative_economics": QE,
    "te": TE,
    "theoretical_economics": TE,
    "aer": AER,
    "american_economic_review": AER,
    "article": DEFAULT,
    "default": DEFAULT,
}


def _resolve_profile(profile: str | JournalProfile) -> JournalProfile:
    """Case-insensitive lookup delegating to the canonical registry.

    Accepts either a :class:`JournalProfile` instance or one of the
    string keys ``"qe"``, ``"te"``, ``"ecta"``, ``"econometrica"``,
    ``"aer"``, ``"american_economic_review"``, ``"article"``,
    ``"default"``.
    """
    if isinstance(profile, JournalProfile):
        return profile
    key = str(profile).strip().lower()
    if key in _PROFILE_MAP:
        return _PROFILE_MAP[key]
    try:
        return _resolve_profile_registry(profile)
    except ValueError:
        raise ValueError(
            f"Unknown journal profile '{profile}'. "
            f"Available: {sorted(_PROFILE_MAP)}"
        ) from None


# ---------------------------------------------------------------------------
# Paper orchestrator
# ---------------------------------------------------------------------------


class Paper:
    """Fluent-builder orchestrator producing a publication-grade LaTeX paper.

    Parameters
    ----------
    id:
        Short slug used for file names and logs.
    title:
        Paper title.
    authors:
        List of author name strings.
    abstract:
        Abstract text (plain text, may contain LaTeX).
    profile:
        Journal profile key (``"econometrica"``, ``"aer"``, ``"default"``)
        or a :class:`JournalProfile` instance.
    """

    def __init__(
        self,
        id: str,
        title: str,
        authors: list[str],
        abstract: str = "",
        profile: str | JournalProfile = "default",
        numbered_sections: bool = True,
        style: "str | Path | Mapping[str, Any] | StyleConfig | None" = None,
    ) -> None:
        from econtools.output.style import DEFAULT_STYLE, StyleConfig, load_style
        self.id = id
        self.title = title
        self.authors = list(authors)
        self.abstract = abstract
        self.profile = _resolve_profile(profile)
        self.numbered_sections = numbered_sections
        self.style: StyleConfig = (
            style if isinstance(style, StyleConfig)
            else load_style(style)
        )

        # Plan state
        self._loads: dict[str, DataLoadNode] = {}
        self._transforms: list[TransformNode] = []
        self._tables: dict[str, TableNode] = {}
        self._figures: dict[str, FigureNode] = {}
        self._composites: dict[str, "CompositeNode"] = {}
        self._composite_order: list[str] = []
        self._sections: list[SectionNode] = []
        self._appendix_ids: list[str] = []
        self._table_order: list[str] = []
        self._figure_order: list[str] = []

        # Fluent transform recorder (always targets last loaded dataset by default)
        self.transform: DataTransformer = DataTransformer()
        self._transform_target: str | None = None

    # ------------------------------------------------------------------ #
    # Data loading                                                       #
    # ------------------------------------------------------------------ #

    def load(
        self,
        source: str | pd.DataFrame,
        as_: str = "main",
        *,
        entity_col: str | None = None,
        time_col: str | None = None,
    ) -> "Paper":
        """Register a dataset load.

        *source* is either a path (parquet/csv/dta) or a pre-loaded
        :class:`pandas.DataFrame`.  The dataset is keyed by *as_* and
        becomes the default target for subsequent transforms and tables.
        """
        if isinstance(source, pd.DataFrame):
            node = DataLoadNode(
                name=as_, df=source, entity_col=entity_col, time_col=time_col
            )
        else:
            node = DataLoadNode(
                name=as_, path=str(source), entity_col=entity_col, time_col=time_col
            )
        self._loads[as_] = node
        self._transform_target = as_
        # Bind the current transformer to this dataset — subsequent transform
        # calls populate it; ``_transforms`` list is built on build().
        return self

    # ------------------------------------------------------------------ #
    # Tables                                                             #
    # ------------------------------------------------------------------ #

    def table_summary(
        self,
        table_id: str,
        *,
        vars: list[str],
        title: str = "",
        caption: str | None = None,
        stats: list[str] | None = None,
        var_names: dict[str, str] | None = None,
        dataset: str = "main",
        notes: list[str] | None = None,
    ) -> "Paper":
        """Register a descriptive-statistics table."""
        node = TableNode(
            table_id=table_id,
            kind="summary",
            title=title,
            caption=caption,
            dataset=dataset,
            vars=list(vars),
            stats=list(stats) if stats else ["mean", "std", "min", "max", "N"],
            var_names=dict(var_names) if var_names else None,
            notes=list(notes) if notes else [],
        )
        self._register_table(node)
        return self

    def table_results(
        self,
        table_id: str,
        *,
        columns: list[dict[str, Any] | ColumnSpec] | None = None,
        system: SystemSpec | None = None,
        title: str = "",
        caption: str | None = None,
        dataset: str = "main",
        footer_stats: list[str] | None = None,
        omit_vars: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> "Paper":
        """Register a multi-model results table.

        Call with either *columns* (one column per single-equation model)
        or *system* (a :class:`SystemSpec` — one column per equation).
        When both are supplied, ``system`` wins and ``columns`` is
        ignored.

        Each entry in *columns* is either a :class:`ColumnSpec` or a plain
        dict keyed by ``ColumnSpec`` fields, with an optional ``"label"``
        field carrying the column header.
        """
        if system is None and not columns:
            raise ValueError(
                "table_results requires either 'columns' or 'system'."
            )

        col_specs: list[ColumnSpec] = []
        if system is None:
            for i, c in enumerate(columns or []):
                if isinstance(c, ColumnSpec):
                    col_specs.append(c)
                    continue
                d = dict(c)
                d.setdefault("column_id", f"({i + 1})")
                col_specs.append(ColumnSpec(**d))
        node = TableNode(
            table_id=table_id,
            kind="results",
            title=title,
            caption=caption,
            dataset=dataset,
            columns=col_specs,
            footer_stats=list(footer_stats) if footer_stats else None,
            omit_vars=list(omit_vars) if omit_vars else None,
            notes=list(notes) if notes else [],
            system=system,
        )
        self._register_table(node)
        return self

    def table_system(
        self,
        table_id: str,
        *,
        equations: dict[str, EquationSpec | dict[str, Any]] | SystemSpec,
        estimator: str = "sur",
        cov_type: str = "robust",
        cov_kwargs: dict[str, Any] | None = None,
        add_constant: bool = True,
        title: str = "",
        caption: str | None = None,
        dataset: str = "main",
        footer_stats: list[str] | None = None,
        omit_vars: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> "Paper":
        """Register a system-of-equations results table (SUR / 3SLS).

        Sibling of :meth:`table_results` for the case where the user has
        a bag of per-equation specs rather than a pre-built
        :class:`SystemSpec`.  Ultimately forwards to ``table_results``
        with a synthesised :class:`SystemSpec`.

        Parameters
        ----------
        equations:
            Either a fully-built :class:`SystemSpec`, or a mapping
            ``label -> EquationSpec | dict`` (dicts are forwarded to
            :class:`EquationSpec`'s constructor).
        estimator, cov_type, cov_kwargs, add_constant:
            Passed through to the constructed :class:`SystemSpec` when
            *equations* is a dict.  Ignored when a :class:`SystemSpec`
            is supplied directly.
        """
        if isinstance(equations, SystemSpec):
            system = equations
        else:
            eqs: dict[str, EquationSpec] = {}
            for lbl, eq in equations.items():
                if isinstance(eq, EquationSpec):
                    eqs[lbl] = eq
                else:
                    eqs[lbl] = EquationSpec(**dict(eq))
            system = SystemSpec(
                equations=eqs,
                estimator=estimator,
                cov_type=cov_type,
                cov_kwargs=dict(cov_kwargs) if cov_kwargs else {},
                add_constant=add_constant,
            )
        return self.table_results(
            table_id,
            system=system,
            title=title,
            caption=caption,
            dataset=dataset,
            footer_stats=footer_stats,
            omit_vars=omit_vars,
            notes=notes,
        )

    def table_diagnostics(
        self,
        table_id: str,
        *,
        model: str,
        tests: list[str],
        title: str = "",
        caption: str | None = None,
        notes: list[str] | None = None,
    ) -> "Paper":
        """Register a diagnostics table for a fitted column of another table.

        ``model`` resolves to a :class:`Estimate` via
        ``"<table_id>:<column_label>"``.
        """
        node = TableNode(
            table_id=table_id,
            kind="diagnostics",
            title=title,
            caption=caption,
            model=model,
            tests=list(tests),
            notes=list(notes) if notes else [],
        )
        self._register_table(node)
        return self

    def _register_table(self, node: TableNode) -> None:
        if node.table_id in self._tables:
            raise ValueError(f"Duplicate table id '{node.table_id}'")
        self._tables[node.table_id] = node
        self._table_order.append(node.table_id)

    # ------------------------------------------------------------------ #
    # Figures                                                            #
    # ------------------------------------------------------------------ #

    def figure_residuals(
        self,
        figure_id: str,
        *,
        model: str,
        title: str = "",
        caption: str | None = None,
    ) -> "Paper":
        """Residuals-vs-fitted + Q-Q panel for a fitted model."""
        node = FigureNode(
            figure_id=figure_id,
            kind="residuals",
            title=title,
            caption=caption,
            model=model,
        )
        self._register_figure(node)
        return self

    def figure_coef(
        self,
        figure_id: str,
        *,
        model: str,
        title: str = "",
        caption: str | None = None,
    ) -> "Paper":
        """Coefficient forest plot for a fitted model."""
        node = FigureNode(
            figure_id=figure_id,
            kind="coef",
            title=title,
            caption=caption,
            model=model,
        )
        self._register_figure(node)
        return self

    def figure_series(
        self,
        figure_id: str,
        *,
        cols: list[str],
        dataset: str = "main",
        groupby: str | None = None,
        time_col: str | None = None,
        title: str = "",
        caption: str | None = None,
    ) -> "Paper":
        """Time-series plot of one or more columns, optionally grouped."""
        node = FigureNode(
            figure_id=figure_id,
            kind="series",
            title=title,
            caption=caption,
            dataset=dataset,
            cols=list(cols),
            groupby=groupby,
            time_col=time_col,
        )
        self._register_figure(node)
        return self

    def figure_distribution(
        self,
        figure_id: str,
        *,
        col: str,
        dataset: str = "main",
        title: str = "",
        caption: str | None = None,
    ) -> "Paper":
        """Histogram of one column."""
        node = FigureNode(
            figure_id=figure_id,
            kind="distribution",
            title=title,
            caption=caption,
            dataset=dataset,
            cols=[col],
        )
        self._register_figure(node)
        return self

    def figure_scatter_fit(
        self,
        figure_id: str,
        *,
        panels: list[dict[str, Any]],
        dataset: str = "main",
        layout: tuple[int, int] | None = None,
        weight_col: str | None = None,
        cov_type: str = "HC1",
        title: str = "",
        caption: str | None = None,
        notes: str | None = None,
        hspace: float | None = None,
        wspace: float | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> "Paper":
        """Multi-panel scatter + fitted regression figure.

        Each entry of ``panels`` is a dict with at least ``x`` and ``y``
        column names, plus optional ``label`` (e.g. ``"Panel A. First
        stage"``), ``xlabel``, ``ylabel``, ``subtitle``, and
        ``annotate_loc``. A single call produces a publication-ready
        added-variable-plot figure with consistent styling, fitted lines,
        and an inset coef/SE/t box per panel.
        """
        if not panels:
            raise ValueError("figure_scatter_fit requires at least one panel")
        node = FigureNode(
            figure_id=figure_id,
            kind="scatter_fit",
            title=title,
            caption=caption,
            notes=notes,
            dataset=dataset,
            panels=[dict(p) for p in panels],
            layout=layout,
            weight_col=weight_col,
            cov_type=cov_type,
            hspace=hspace,
            wspace=wspace,
            figsize=figsize,
        )
        self._register_figure(node)
        return self

    def _register_figure(self, node: FigureNode) -> None:
        if node.figure_id in self._figures:
            raise ValueError(f"Duplicate figure id '{node.figure_id}'")
        self._figures[node.figure_id] = node
        self._figure_order.append(node.figure_id)

    # ------------------------------------------------------------------ #
    # Composite panels (multi-table / multi-figure single float)         #
    # ------------------------------------------------------------------ #

    def composite(
        self,
        composite_id: str,
        *,
        panels: list[str],
        layout: tuple[int, int] | None = None,
        caption: str | None = None,
        notes: str | None = None,
        subcaptions: list[str] | None = None,
        float_kind: str = "figure",
        column_fraction: float | None = None,
    ) -> "Paper":
        """Register a composite float packing multiple tables/figures.

        Each entry in *panels* names an already-registered table or figure id.
        *layout* defaults to ``(1, n)`` for up to 3 panels, ``(2, ceil(n/2))``
        otherwise.  *subcaptions* — when provided — overrides each panel's own
        caption / title inside the composite.
        """
        if not panels:
            raise ValueError("composite requires at least one panel id")
        for pid in panels:
            if pid not in self._tables and pid not in self._figures:
                raise ValueError(
                    f"composite panel '{pid}' is not a registered table or figure"
                )
        node = CompositeNode(
            composite_id=composite_id,
            panels=list(panels),
            layout=layout,
            caption=caption,
            notes=notes,
            subcaptions=list(subcaptions) if subcaptions else None,
            float_kind=float_kind,
            column_fraction=column_fraction,
        )
        if composite_id in self._composites:
            raise ValueError(f"Duplicate composite id '{composite_id}'")
        self._composites[composite_id] = node
        self._composite_order.append(composite_id)
        return self

    # ------------------------------------------------------------------ #
    # Narrative                                                          #
    # ------------------------------------------------------------------ #

    def section(
        self,
        title: str,
        body: str = "",
        after: str | list[str] | None = None,
    ) -> "Paper":
        """Register a narrative section.

        ``after`` names one or more table/figure ids that should appear
        immediately following this section's text.
        """
        if after is None:
            after_list: list[str] = []
        elif isinstance(after, str):
            after_list = [after]
        else:
            after_list = list(after)
        self._sections.append(
            SectionNode(title=title, body=body, after=after_list)
        )
        return self

    def appendix(self, *ids: str) -> "Paper":
        """Route one or more table/figure ids into the appendix document."""
        for i in ids:
            if i not in self._tables and i not in self._figures:
                raise ValueError(
                    f"Appendix references unknown id '{i}'"
                )
            if i not in self._appendix_ids:
                self._appendix_ids.append(i)
        return self

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #

    def table_ids(self) -> list[str]:
        return list(self._table_order)

    def figure_ids(self) -> list[str]:
        return list(self._figure_order)

    def get_table(self, table_id: str) -> TableNode:
        return self._tables[table_id]

    def get_figure(self, figure_id: str) -> FigureNode:
        return self._figures[figure_id]

    def composite_ids(self) -> list[str]:
        return list(self._composite_order)

    def get_composite(self, composite_id: str) -> "CompositeNode":
        return self._composites[composite_id]

    def resolve_model_ref(self, ref: str) -> tuple[str, str]:
        """Parse a ``"<table_id>:<column_label>"`` model reference.

        Returns a ``(table_id, column_label)`` tuple.  Raises
        :class:`KeyError` if the table does not exist or does not contain a
        column with that label.
        """
        if ":" not in ref:
            raise ValueError(
                f"Model reference must be '<table_id>:<column_label>', got '{ref}'"
            )
        table_id, col_label = ref.split(":", 1)
        if table_id not in self._tables:
            raise KeyError(f"Unknown table id '{table_id}' in model ref '{ref}'")
        node = self._tables[table_id]
        if node.system is not None:
            # System-of-equations table: valid labels are equation keys
            # plus a special "SYSTEM" alias for the full fitted result.
            labels = list(node.system.equations.keys()) + ["SYSTEM"]
        else:
            labels = [c.label or c.column_id for c in node.columns]
        if col_label not in labels:
            raise KeyError(
                f"Column label '{col_label}' not in table '{table_id}' "
                f"(available: {labels})"
            )
        return table_id, col_label

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #

    def build(
        self,
        output_dir: str | Path = "build",
        *,
        compile_pdf: bool = True,
        provenance_log: str | Path | None = None,
    ) -> BuildResult:
        """Execute the plan and emit ``paper.tex`` (+ optional PDF).

        Parameters
        ----------
        output_dir:
            Directory where outputs will be written.  Created if absent.
        compile_pdf:
            Attempt LaTeX compilation when an engine is available.  If no
            engine is found, the ``.tex`` file is still written and
            :attr:`BuildResult.pdf_path` is ``None``.
        provenance_log:
            Optional path for a transforms provenance log.  Falls back to
            ``<output_dir>/provenance.json``.
        """
        # Local import to avoid circular dependency at module load time
        from econtools.paper.execution import execute_plan

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(provenance_log) if provenance_log else out_dir / "provenance.json"

        return execute_plan(
            paper=self,
            output_dir=out_dir,
            compile_pdf=compile_pdf,
            provenance_log=log_path,
        )

    # Attach the active transformer to the last loaded dataset at build-time
    def _snapshot_transform(self) -> None:
        """Bind the active DataTransformer to the current dataset target.

        Called internally before execution so the fluent ``paper.transform``
        state is not lost if the user never explicitly committed it.
        """
        if not self._transform_target or not self.transform.steps:
            return
        already = any(
            t.dataset == self._transform_target
            and t.transformer is self.transform
            for t in self._transforms
        )
        if not already:
            self._transforms.append(
                TransformNode(dataset=self._transform_target, transformer=self.transform)
            )

    @property
    def loads(self) -> dict[str, DataLoadNode]:
        """Read-only view of registered dataset loads."""
        return dict(self._loads)

    @property
    def transforms(self) -> list[TransformNode]:
        """Read-only view of registered transform blocks."""
        return list(self._transforms)

    @property
    def sections(self) -> list[SectionNode]:
        return list(self._sections)

    @property
    def appendix_ids(self) -> list[str]:
        return list(self._appendix_ids)
