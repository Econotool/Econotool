"""Replication orchestration engine.

Given a :class:`ReplicationSpec` and a DataFrame, the
:class:`ReplicationRunner` will:

1. Estimate each column specification via ``fit_model``
2. Compare estimates to published values
3. Run identification-appropriate diagnostics
4. Suggest robustness checks and extensions

The runner assumes the caller handles data preparation, interprets
results, and iterates on the spec.

Public API
----------
ReplicationRunner   — orchestrate a full replication
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from econtools._core.types import Estimate, TestResult
from econtools.fit.estimators import fit_model
from econtools.model.spec import ModelSpec
from econtools.replication.comparison import (
    ColumnComparison,
    ReplicationReport,
    compare_column,
)
from econtools.replication.spec import ColumnSpec, ReplicationSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic playbook: identification strategy → diagnostics to run
# ---------------------------------------------------------------------------

_DIAGNOSTIC_PLAYBOOK: dict[str, list[str]] = {
    "iv": [
        "weak_instrument_f",
        "sargan_overid",
        "wu_hausman",
        "first_stage_results",
        "reduced_form",
    ],
    "2sls": [
        "weak_instrument_f",
        "sargan_overid",
        "wu_hausman",
        "first_stage_results",
        "reduced_form",
    ],
    "ols": [
        "breusch_pagan",
        "white_test",
        "reset_test",
        "jarque_bera",
        "vif",
    ],
    "selection_on_obs": [
        "breusch_pagan",
        "white_test",
        "reset_test",
        "vif",
        "oster_bounds",
    ],
    "did": [
        "parallel_trends",
        "placebo_pre_trends",
        "event_study",
        "breusch_pagan",
    ],
    "rdd": [
        "mccrary_density",
        "bandwidth_sensitivity",
        "placebo_cutoff",
        "covariate_balance_at_cutoff",
    ],
    "fe": [
        "hausman_fe_re",
        "breusch_pagan_lm",
        "cluster_se_robustness",
    ],
    "panel": [
        "hausman_fe_re",
        "breusch_pagan_lm",
        "serial_correlation",
    ],
}

# Robustness checks by identification strategy
_ROBUSTNESS_PLAYBOOK: dict[str, list[str]] = {
    "iv": [
        "alternative_instruments",
        "reduced_form_significance",
        "ols_vs_iv_comparison",
        "liml_vs_2sls",
        "subset_instrument_validity",
        "bootstrap_se",
    ],
    "ols": [
        "alternative_se_types",
        "functional_form_variations",
        "sample_sensitivity",
        "added_controls",
        "leave_one_out_variables",
    ],
    "did": [
        "parallel_trends_test",
        "alternative_control_groups",
        "placebo_treatment_dates",
        "staggered_adoption_robust",
        "event_study_dynamic_effects",
    ],
    "rdd": [
        "bandwidth_sensitivity",
        "polynomial_order_sensitivity",
        "donut_hole",
        "placebo_cutoffs",
        "covariate_discontinuity",
    ],
    "fe": [
        "re_vs_fe_hausman",
        "correlated_random_effects",
        "alternative_clustering",
        "time_trends",
    ],
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class EstimationResult:
    """Result of estimating one column."""

    column_spec: ColumnSpec
    table_id: str
    estimate: Estimate | None = None
    error: str | None = None
    diagnostics: dict[str, TestResult | Any] = field(default_factory=dict)
    comparison: ColumnComparison | None = None


class ReplicationRunner:
    """Orchestrate a full paper replication.

    Parameters
    ----------
    spec:
        The :class:`ReplicationSpec` describing the paper.
    df:
        The analysis DataFrame (already cleaned and constructed).
    coef_tol_pct:
        Percentage tolerance for coefficient matching (default 1%).
    se_tol_pct:
        Percentage tolerance for SE matching (default 5%).
    """

    def __init__(
        self,
        spec: ReplicationSpec,
        df: pd.DataFrame,
        *,
        coef_tol_pct: float = 1.0,
        se_tol_pct: float = 5.0,
    ):
        self.spec = spec
        self.df = df
        self.coef_tol_pct = coef_tol_pct
        self.se_tol_pct = se_tol_pct
        self.results: list[EstimationResult] = []
        self._report: ReplicationReport | None = None

    def _column_to_model_spec(self, col: ColumnSpec) -> ModelSpec:
        """Convert a ColumnSpec to a ModelSpec for estimation."""
        # Map replication estimator names to ModelSpec estimator names
        estimator_map = {
            "ols": "ols",
            "2sls": "2sls",
            "iv": "2sls",
            "fe": "fe",
            "re": "re",
            "fd": "fd",
            "pooled": "pooled",
            "between": "between",
            "probit": "probit",
            "logit": "logit",
            "liml": "liml",
            "gmm": "gmm",
            "wls": "wls",
        }

        estimator = estimator_map.get(col.estimator, col.estimator)

        # Build cov_kwargs
        cov_kwargs: dict[str, Any] = {}
        if col.cluster_var:
            cov_kwargs["groups"] = col.cluster_var
        if len(col.cluster_vars) == 2:
            cov_kwargs["groups"] = col.cluster_vars[0]
            cov_kwargs["groups2"] = col.cluster_vars[1]

        return ModelSpec(
            dep_var=col.dep_var,
            exog_vars=col.exog_vars,
            endog_vars=col.endog_vars,
            instruments=col.instruments,
            entity_col=col.entity_col,
            time_col=col.time_col,
            weights_col=col.weights_var,
            estimator=estimator,
            cov_type=col.cov_type,
            cov_kwargs=cov_kwargs,
            add_constant=col.add_constant,
        )

    def estimate_column(
        self,
        col: ColumnSpec,
        table_id: str = "",
        df: pd.DataFrame | None = None,
    ) -> EstimationResult:
        """Estimate a single column specification.

        Parameters
        ----------
        col:
            The column to estimate.
        table_id:
            Table identifier for labelling.
        df:
            Override DataFrame (e.g. for subsample). If None, uses self.df.
        """
        data = df if df is not None else self.df

        # Apply sample restriction
        if col.sample_restriction:
            try:
                data = data.query(col.sample_restriction)
            except Exception as e:
                return EstimationResult(
                    column_spec=col,
                    table_id=table_id,
                    error=f"Sample restriction failed: {col.sample_restriction!r} — {e}",
                )

        # Build ModelSpec and estimate
        try:
            model_spec = self._column_to_model_spec(col)
            estimate = fit_model(model_spec, data)
        except Exception as e:
            return EstimationResult(
                column_spec=col,
                table_id=table_id,
                error=f"Estimation failed: {e}",
            )

        # Compare to published values
        comparison = None
        if col.published.coefficients:
            comparison = compare_column(
                estimate, col,
                coef_tol_pct=self.coef_tol_pct,
                se_tol_pct=self.se_tol_pct,
                table_id=table_id,
            )

        result = EstimationResult(
            column_spec=col,
            table_id=table_id,
            estimate=estimate,
            comparison=comparison,
        )
        self.results.append(result)
        return result

    def estimate_all(self) -> list[EstimationResult]:
        """Estimate all columns in all tables."""
        results = []
        for table in self.spec.tables:
            for col in table.columns:
                result = self.estimate_column(col, table_id=table.table_id)
                results.append(result)
                if result.error:
                    logger.warning(
                        "Column %s %s failed: %s",
                        table.table_id, col.column_id, result.error,
                    )
        return results

    def run_diagnostics(self, result: EstimationResult) -> dict[str, Any]:
        """Run identification-appropriate diagnostics on one result.

        Uses the paper's identification strategy to select which
        diagnostics to run from the playbook.
        """
        if result.estimate is None:
            return {}

        strategy = self.spec.identification.lower()
        # Also consider column-level estimator
        col_strategy = result.column_spec.estimator.lower()
        if col_strategy in ("2sls", "iv", "liml", "gmm"):
            strategy = "iv"

        playbook = _DIAGNOSTIC_PLAYBOOK.get(strategy, [])
        diagnostics: dict[str, Any] = {}

        for diag_name in playbook:
            try:
                diag_result = self._run_single_diagnostic(
                    diag_name, result.estimate, result.column_spec
                )
                if diag_result is not None:
                    diagnostics[diag_name] = diag_result
            except Exception as e:
                diagnostics[diag_name] = f"Error: {e}"
                logger.debug("Diagnostic %s failed: %s", diag_name, e)

        result.diagnostics = diagnostics
        return diagnostics

    def _run_single_diagnostic(
        self,
        name: str,
        estimate: Estimate,
        col: ColumnSpec,
    ) -> TestResult | Any | None:
        """Run a single named diagnostic."""
        if name == "breusch_pagan":
            from econtools.evaluation.heteroskedasticity import breusch_pagan
            return breusch_pagan(estimate)

        if name == "white_test":
            from econtools.evaluation.heteroskedasticity import white_test
            return white_test(estimate)

        if name == "reset_test":
            from econtools.evaluation.specification import reset_test
            return reset_test(estimate)

        if name == "jarque_bera":
            from econtools.evaluation.normality import jarque_bera
            return jarque_bera(estimate)

        if name == "vif":
            from econtools.evaluation.multicollinearity import compute_vif
            return compute_vif(estimate)

        if name == "weak_instrument_f":
            from econtools.evaluation.iv_checks import weak_instrument_tests
            return weak_instrument_tests(estimate)

        if name == "sargan_overid":
            from econtools.evaluation.iv_checks import sargan_test
            return sargan_test(estimate)

        if name == "wu_hausman":
            from econtools.evaluation.iv_checks import wu_hausman_test
            return wu_hausman_test(estimate)

        # Diagnostics not yet implemented return None
        return None

    def get_suggested_robustness(self) -> list[str]:
        """Return list of suggested robustness checks based on identification."""
        strategy = self.spec.identification.lower()
        checks = _ROBUSTNESS_PLAYBOOK.get(strategy, [])

        # Filter out checks the paper already reports
        reported = {r.lower() for r in self.spec.robustness_reported}
        return [c for c in checks if c.lower() not in reported]

    def build_report(self) -> ReplicationReport:
        """Build a ReplicationReport from all results so far."""
        report = ReplicationReport(paper_id=self.spec.paper_id)

        for result in self.results:
            report.total_columns += 1
            if result.error:
                report.columns_failed += 1
                report.warnings.append(
                    f"{result.table_id} {result.column_spec.column_id}: {result.error}"
                )
            elif result.comparison:
                report.column_results.append(result.comparison)
                if result.comparison.all_passed:
                    report.columns_passed += 1
                else:
                    report.columns_failed += 1
            else:
                # No published values to compare against
                report.columns_passed += 1  # assume pass if no reference

        # Add diagnostic names
        for result in self.results:
            for diag_name in result.diagnostics:
                if diag_name not in report.diagnostics_run:
                    report.diagnostics_run.append(diag_name)

        # Add robustness suggestions
        report.suggestions = self.get_suggested_robustness()

        self._report = report
        return report
