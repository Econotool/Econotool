"""Declarative replication specification.

A :class:`ReplicationSpec` fully describes what a published paper estimates,
what data it uses, what causal claims it makes, and what coefficients/SEs
were reported — so the paper's analysis can be replicated, diagnosed,
and extended from a single declarative source.

Designed for round-tripping: can be serialised to/from YAML for storage
and editing, and can be partially auto-populated from Stata .do files
via :mod:`econtools.replication.do_parser`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

@dataclass
class DataRef:
    """Pointer to the paper's source data.

    At least one of ``doi``, ``url``, or ``local_path`` should be set.
    """

    source: str = "unknown"          # "dataverse", "icpsr", "openicpsr", "url", "local"
    doi: str | None = None           # e.g. "10.7910/DVN/ENLGZX"
    url: str | None = None
    local_path: str | None = None
    files: list[str] = field(default_factory=list)  # specific files needed
    description: str = ""


@dataclass
class VariableDef:
    """Definition of a variable used in the analysis.

    Captures the mapping from raw data columns to the analysis variable,
    including any transformations applied.
    """

    name: str                         # column name in analysis dataset
    label: str = ""                   # human-readable label
    construction: str | None = None   # e.g. "log(wage)", "educ^2", "age - mean(age)"
    source_vars: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SampleRestriction:
    """A sample selection criterion."""

    description: str                  # e.g. "Married women in labour force"
    expression: str = ""              # pandas query expression: "inlf == 1"
    reduces_n_from: int | None = None
    reduces_n_to: int | None = None


@dataclass
class PublishedValues:
    """Published point estimates and standard errors for one column."""

    coefficients: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    n_obs: int | None = None
    r_squared: float | None = None
    first_stage_f: float | None = None
    other_stats: dict[str, float] = field(default_factory=dict)


@dataclass
class ColumnSpec:
    """One column in a regression table.

    Maps directly to a :class:`econtools.model.spec.ModelSpec` for estimation,
    plus stores published values for comparison.
    """

    column_id: str                    # e.g. "(1)", "(2)", "3a"
    dep_var: str
    exog_vars: list[str]
    endog_vars: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    estimator: str = "ols"            # "ols", "2sls", "fe", "re", "did", "rdd", etc.
    cov_type: str = "classical"
    cluster_var: str | None = None
    cluster_vars: list[str] = field(default_factory=list)  # for two-way
    absorb_vars: list[str] = field(default_factory=list)   # fixed-effect vars
    entity_col: str | None = None
    time_col: str | None = None
    weights_var: str | None = None
    sample_restriction: str | None = None   # pandas query or restriction key
    add_constant: bool = True

    # What the paper reports
    published: PublishedValues = field(default_factory=PublishedValues)

    # Notes / context
    label: str = ""                   # e.g. "Baseline OLS", "IV — QOB instruments"
    notes: str = ""


@dataclass
class TableSpec:
    """One table from a paper."""

    table_id: str                     # "table_2", "table_A1"
    title: str = ""
    columns: list[ColumnSpec] = field(default_factory=list)
    panel: str | None = None          # "Panel A", "Panel B" if applicable
    notes: list[str] = field(default_factory=list)


@dataclass
class CausalClaim:
    """A causal claim made by the paper.

    Captures what the paper is trying to show and anchors the set of
    robustness checks that are relevant to evaluating it.
    """

    treatment: str                    # treatment variable
    outcome: str                      # outcome variable
    effect_direction: str | None = None   # "positive", "negative", "null"
    effect_magnitude: float | None = None
    effect_units: str = ""            # "log points", "percentage points", "years"
    identification: str = ""          # reference to identification strategy
    population: str = ""              # "compliers", "all", "treated"
    mechanism: str = ""               # proposed mechanism if any
    table_ref: str = ""               # "Table 2, Column (3)"


@dataclass
class ComparisonResult:
    """Result of comparing one estimated coefficient to a published value."""

    variable: str
    published: float
    estimated: float
    abs_diff: float
    pct_diff: float                   # |estimated - published| / |published| * 100
    within_tolerance: bool
    tolerance_pct: float              # the tolerance used


# ---------------------------------------------------------------------------
# Top-level specification
# ---------------------------------------------------------------------------

@dataclass
class ReplicationSpec:
    """Full replication specification for an academic paper.

    This is the central artifact produced when parsing a paper, and the
    input that the :class:`ReplicationRunner` consumes.

    Can be serialised to/from YAML via :meth:`to_yaml` / :meth:`from_yaml`.
    """

    # --- Paper metadata ---
    paper_id: str                     # slug: "angrist_krueger_1991"
    title: str
    authors: list[str]
    year: int
    journal: str = ""
    doi: str = ""                     # paper DOI (not data DOI)

    # --- Identification ---
    identification: str = ""          # "iv", "did", "rdd", "selection_on_obs", "rd_did", etc.
    identification_detail: str = ""   # free-text: "QOB instruments for schooling"
    assumptions: list[str] = field(default_factory=list)
    testable_implications: list[str] = field(default_factory=list)
    threats: list[str] = field(default_factory=list)

    # --- Data ---
    data: DataRef = field(default_factory=DataRef)
    variables: list[VariableDef] = field(default_factory=list)
    sample_restrictions: list[SampleRestriction] = field(default_factory=list)

    # --- What to estimate ---
    tables: list[TableSpec] = field(default_factory=list)
    claims: list[CausalClaim] = field(default_factory=list)

    # --- What robustness to run ---
    robustness_reported: list[str] = field(default_factory=list)   # what the paper does
    robustness_suggested: list[str] = field(default_factory=list)  # extensions to add

    # --- Analyst notes ---
    analyst_notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (for YAML/JSON serialisation)."""
        import dataclasses
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write spec to a YAML file."""
        if not _HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=120,
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplicationSpec:
        """Construct from a plain dict."""
        # Reconstruct nested dataclasses
        if "data" in data and isinstance(data["data"], dict):
            data["data"] = DataRef(**data["data"])

        if "variables" in data:
            data["variables"] = [
                VariableDef(**v) if isinstance(v, dict) else v
                for v in data["variables"]
            ]

        if "sample_restrictions" in data:
            data["sample_restrictions"] = [
                SampleRestriction(**s) if isinstance(s, dict) else s
                for s in data["sample_restrictions"]
            ]

        if "claims" in data:
            data["claims"] = [
                CausalClaim(**c) if isinstance(c, dict) else c
                for c in data["claims"]
            ]

        if "tables" in data:
            tables = []
            for t in data["tables"]:
                if isinstance(t, dict):
                    cols = t.get("columns", [])
                    t["columns"] = [
                        _parse_column_spec(c) if isinstance(c, dict) else c
                        for c in cols
                    ]
                    tables.append(TableSpec(**t))
                else:
                    tables.append(t)
            data["tables"] = tables

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReplicationSpec:
        """Load spec from a YAML file."""
        if not _HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def get_all_column_specs(self) -> list[ColumnSpec]:
        """Flatten all columns from all tables."""
        cols = []
        for table in self.tables:
            cols.extend(table.columns)
        return cols


def _parse_column_spec(d: dict) -> ColumnSpec:
    """Parse a ColumnSpec from a dict, handling nested PublishedValues."""
    if "published" in d and isinstance(d["published"], dict):
        d["published"] = PublishedValues(**d["published"])
    return ColumnSpec(**d)
