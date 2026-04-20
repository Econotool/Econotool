"""Compare estimated results to published values.

Provides tolerance-based comparison of point estimates, standard errors,
sample sizes, and fit statistics against values reported in the paper.

Public API
----------
compare_column      — compare one Estimate to one ColumnSpec's published values
compare_all         — compare all columns in a ReplicationSpec
ReplicationReport   — structured comparison results
"""

from __future__ import annotations

from dataclasses import dataclass, field

from econtools._core.types import Estimate
from econtools.replication.spec import ColumnSpec, ComparisonResult


# ---------------------------------------------------------------------------
# Tolerance defaults
# ---------------------------------------------------------------------------

DEFAULT_COEF_TOL_PCT = 1.0      # 1% relative tolerance for coefficients
DEFAULT_SE_TOL_PCT = 5.0        # 5% for standard errors (SEs vary more across packages)
DEFAULT_N_TOL = 0               # exact match for sample size
DEFAULT_R2_TOL = 0.01           # absolute tolerance for R²


# ---------------------------------------------------------------------------
# Single-column comparison
# ---------------------------------------------------------------------------

@dataclass
class ColumnComparison:
    """Comparison results for a single regression column."""

    table_id: str
    column_id: str
    label: str
    n_compared: int = 0
    n_matched: int = 0
    n_failed: int = 0
    coefficient_comparisons: list[ComparisonResult] = field(default_factory=list)
    se_comparisons: list[ComparisonResult] = field(default_factory=list)
    n_obs_match: bool | None = None       # None if not checked
    r_squared_match: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.n_failed == 0 and all(
            c.within_tolerance for c in self.coefficient_comparisons
        )


def compare_column(
    estimate: Estimate,
    col_spec: ColumnSpec,
    *,
    coef_tol_pct: float = DEFAULT_COEF_TOL_PCT,
    se_tol_pct: float = DEFAULT_SE_TOL_PCT,
    n_tol: int = DEFAULT_N_TOL,
    r2_tol: float = DEFAULT_R2_TOL,
    table_id: str = "",
) -> ColumnComparison:
    """Compare an estimated result to published values in a ColumnSpec.

    Parameters
    ----------
    estimate:
        The fitted :class:`Estimate` from ``fit_model``.
    col_spec:
        The :class:`ColumnSpec` with published values to compare against.
    coef_tol_pct:
        Percentage tolerance for coefficient comparison (default 1%).
    se_tol_pct:
        Percentage tolerance for SE comparison (default 5%).
    n_tol:
        Absolute tolerance for sample size (default 0 = exact).
    r2_tol:
        Absolute tolerance for R² (default 0.01).
    table_id:
        Table identifier for labelling.

    Returns
    -------
    ColumnComparison
    """
    comp = ColumnComparison(
        table_id=table_id,
        column_id=col_spec.column_id,
        label=col_spec.label,
    )
    pub = col_spec.published

    # Compare coefficients
    for var, pub_val in pub.coefficients.items():
        if var in estimate.params.index:
            est_val = float(estimate.params[var])
            abs_diff = abs(est_val - pub_val)
            pct_diff = (abs_diff / abs(pub_val) * 100) if pub_val != 0 else float("inf")
            within = pct_diff <= coef_tol_pct

            cr = ComparisonResult(
                variable=var,
                published=pub_val,
                estimated=est_val,
                abs_diff=abs_diff,
                pct_diff=pct_diff,
                within_tolerance=within,
                tolerance_pct=coef_tol_pct,
            )
            comp.coefficient_comparisons.append(cr)
            comp.n_compared += 1
            if within:
                comp.n_matched += 1
            else:
                comp.n_failed += 1
        else:
            comp.notes.append(f"Variable '{var}' not found in estimates")
            comp.n_failed += 1

    # Compare standard errors
    for var, pub_se in pub.std_errors.items():
        if var in estimate.bse.index:
            est_se = float(estimate.bse[var])
            abs_diff = abs(est_se - pub_se)
            pct_diff = (abs_diff / abs(pub_se) * 100) if pub_se != 0 else float("inf")
            within = pct_diff <= se_tol_pct

            comp.se_comparisons.append(ComparisonResult(
                variable=var,
                published=pub_se,
                estimated=est_se,
                abs_diff=abs_diff,
                pct_diff=pct_diff,
                within_tolerance=within,
                tolerance_pct=se_tol_pct,
            ))

    # Compare N
    if pub.n_obs is not None:
        est_n = estimate.fit.nobs
        comp.n_obs_match = abs(est_n - pub.n_obs) <= n_tol
        if not comp.n_obs_match:
            comp.notes.append(
                f"N mismatch: published={pub.n_obs}, estimated={est_n}"
            )

    # Compare R²
    if pub.r_squared is not None:
        import math
        est_r2 = estimate.fit.r_squared
        if not math.isnan(est_r2):
            comp.r_squared_match = abs(est_r2 - pub.r_squared) <= r2_tol
            if not comp.r_squared_match:
                comp.notes.append(
                    f"R² mismatch: published={pub.r_squared:.4f}, "
                    f"estimated={est_r2:.4f}"
                )

    return comp


# ---------------------------------------------------------------------------
# Full replication comparison
# ---------------------------------------------------------------------------

@dataclass
class ReplicationReport:
    """Aggregated comparison across all tables and columns."""

    paper_id: str
    total_columns: int = 0
    columns_passed: int = 0
    columns_failed: int = 0
    column_results: list[ColumnComparison] = field(default_factory=list)
    diagnostics_run: list[str] = field(default_factory=list)
    robustness_results: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def replication_rate(self) -> float:
        """Fraction of columns that replicate within tolerance."""
        if self.total_columns == 0:
            return 0.0
        return self.columns_passed / self.total_columns

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Replication Report: {self.paper_id}",
            f"{'=' * 50}",
            f"Columns tested:  {self.total_columns}",
            f"Columns passed:  {self.columns_passed}",
            f"Columns failed:  {self.columns_failed}",
            f"Replication rate: {self.replication_rate:.0%}",
        ]

        if self.column_results:
            lines.append("")
            lines.append("Column Details:")
            for cr in self.column_results:
                status = "PASS" if cr.all_passed else "FAIL"
                lines.append(f"  {cr.table_id} {cr.column_id} [{status}] "
                           f"— {cr.n_matched}/{cr.n_compared} coefficients match")
                for note in cr.notes:
                    lines.append(f"    ! {note}")
                for cc in cr.coefficient_comparisons:
                    if not cc.within_tolerance:
                        lines.append(
                            f"    {cc.variable}: pub={cc.published:.4f}, "
                            f"est={cc.estimated:.4f} ({cc.pct_diff:.1f}% off)"
                        )

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        if self.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for s in self.suggestions:
                lines.append(f"  → {s}")

        return "\n".join(lines)
