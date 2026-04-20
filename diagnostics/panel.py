"""Panel diagnostics (lead/lag tests, Hausman FE vs RE, Breusch-Pagan LM)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from econtools.inference.hypothesis import TestResult, wald_test
from econtools.models.ols import fit_ols


def lead_test_strict_exogeneity(
    df: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    *,
    entity: str,
    time: str,
    leads: int = 1,
    cov_type: str = "cluster",
) -> TestResult:
    """Lead test for strict exogeneity in a first-difference model.

    Adds leads of the *levels* of regressors to the FD regression and
    tests whether their coefficients are jointly zero.
    """
    if leads < 1:
        raise ValueError("leads must be >= 1.")

    cols = [entity, time, dep_var] + list(exog_vars)
    data = df[cols].dropna().copy()
    data = data.sort_values([entity, time])

    grp = data.groupby(entity, sort=False)
    data["d_y"] = grp[dep_var].diff()
    for x in exog_vars:
        data[f"d_{x}"] = grp[x].diff()
        for k in range(1, leads + 1):
            data[f"lead{k}_{x}"] = grp[x].shift(-k)

    lead_cols = [f"lead{k}_{x}" for k in range(1, leads + 1) for x in exog_vars]
    diff_cols = [f"d_{x}" for x in exog_vars]
    needed = ["d_y"] + diff_cols + lead_cols
    data = data.dropna(subset=needed)

    result = fit_ols(
        data,
        "d_y",
        diff_cols + lead_cols,
        add_constant=False,
        cov_type=cov_type,
        groups=data[entity] if cov_type == "cluster" else None,
    )

    # Wald test: all lead coefficients = 0
    params = list(result.params.index)
    R = np.zeros((len(lead_cols), len(params)))
    for i, name in enumerate(lead_cols):
        if name not in params:
            raise ValueError(f"Lead term '{name}' not in regression.")
        R[i, params.index(name)] = 1.0

    test = wald_test(result, R, use_f=True)
    return TestResult(
        test_name="Lead test (strict exogeneity)",
        statistic=test.statistic,
        pvalue=test.pvalue,
        df=test.df,
        distribution=test.distribution,
        null_hypothesis="Coefficients on leads are jointly zero",
        reject=test.pvalue < 0.05,
    )


def hausman_fe_re(
    fe_result,
    re_result,
) -> TestResult:
    """Hausman test for FE vs RE consistency.

    Tests whether the RE estimator is consistent by comparing FE and RE
    coefficient estimates.  Under H₀ (RE is consistent), the difference
    ``b_FE - b_RE`` should be close to zero.

    Parameters
    ----------
    fe_result:
        :class:`~econtools._core.types.Estimate` from a fixed-effects model.
    re_result:
        :class:`~econtools._core.types.Estimate` from a random-effects model.

    Returns
    -------
    TestResult
        Chi-squared statistic and p-value.

    Notes
    -----
    Only coefficients that appear in *both* models (excluding the constant)
    are compared.  The test uses ``Var(b_FE - b_RE) = Var(b_FE) - Var(b_RE)``.
    """
    # Find common coefficients (exclude constant)
    fe_idx = set(fe_result.params.index)
    re_idx = set(re_result.params.index)
    common = sorted(fe_idx & re_idx - {"const", "Intercept"})
    if not common:
        raise ValueError("No common coefficients between FE and RE results.")

    b_diff = fe_result.params[common].values - re_result.params[common].values
    V_fe = np.asarray(fe_result.cov_params.loc[common, common])
    V_re = np.asarray(re_result.cov_params.loc[common, common])
    V_diff = V_fe - V_re

    # V_diff should be positive semi-definite under H0; use pseudoinverse
    # to handle near-singular cases
    try:
        V_diff_inv = np.linalg.pinv(V_diff)
    except np.linalg.LinAlgError:
        V_diff_inv = np.linalg.pinv(V_diff)

    chi2_stat = float(b_diff @ V_diff_inv @ b_diff)
    chi2_stat = max(chi2_stat, 0.0)  # guard against numerical negatives
    df = len(common)
    pvalue = float(1.0 - stats.chi2.cdf(chi2_stat, df))

    return TestResult(
        test_name="Hausman (FE vs RE)",
        statistic=chi2_stat,
        pvalue=pvalue,
        df=float(df),
        distribution="Chi2",
        null_hypothesis="RE is consistent (difference in coefficients is not systematic)",
        reject=pvalue < 0.05,
        details={"coefficients_compared": common},
    )


def breusch_pagan_lm_test(
    pooled_result,
) -> TestResult:
    """Breusch-Pagan LM test for random effects.

    Tests whether the variance of the entity-specific error component is
    zero (i.e., whether pooled OLS is adequate vs RE).

    Parameters
    ----------
    pooled_result:
        :class:`~econtools._core.types.Estimate` from a pooled OLS model.
        The underlying data must have a MultiIndex ``(entity, time)``.

    Returns
    -------
    TestResult
        Chi-squared(1) statistic and p-value.
    """
    resid = np.asarray(pooled_result.resid)

    # Get the panel structure from the index
    idx = pooled_result.resid.index
    if not isinstance(idx, pd.MultiIndex):
        raise ValueError(
            "BP LM test requires panel residuals with a MultiIndex (entity, time)."
        )

    entity_ids = idx.get_level_values(0)
    entities = entity_ids.unique()
    N = len(entities)
    T_total = len(resid)

    # Compute sum of (sum_t e_it)^2 and sum of e_it^2
    entity_sums_sq = 0.0
    for ent in entities:
        mask = entity_ids == ent
        entity_sums_sq += resid[mask].sum() ** 2

    sum_sq = float(np.sum(resid ** 2))

    # LM = (N*T / (2*(T-1))) * (sum_i (sum_t e_it)^2 / sum_i sum_t e_it^2  - 1)^2
    # For unbalanced panels, use the general formula
    ratio = entity_sums_sq / sum_sq
    lm_stat = float(T_total / (2.0 * (N - 1)) * (ratio - 1.0) ** 2)
    # Simplified: for balanced T, this equals nT/(2(T-1)) * (ratio - 1)^2
    # General formula: LM = nT/(2(T-1)) * (sum (sum_t e_it)^2 / sum e_it^2 - 1)^2
    lm_stat = max(lm_stat, 0.0)

    pvalue = float(1.0 - stats.chi2.cdf(lm_stat, 1))

    return TestResult(
        test_name="Breusch-Pagan LM (random effects)",
        statistic=lm_stat,
        pvalue=pvalue,
        df=1.0,
        distribution="Chi2",
        null_hypothesis="Var(entity effect) = 0 (pooled OLS is adequate)",
        reject=pvalue < 0.05,
        details={"N_entities": N, "T_total": T_total},
    )


def run_panel_diagnostics(
    df: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    *,
    entity: str,
    time: str,
    tests: Iterable[str] | None = None,
    leads: int = 1,
) -> list[TestResult]:
    """Run panel diagnostics; filter by test names if provided."""
    tests = [t.strip().lower() for t in tests] if tests else None
    outputs: list[TestResult] = []

    if tests is None or "lead" in tests or "lead_test" in tests or "lead-test" in tests:
        outputs.append(
            lead_test_strict_exogeneity(
                df,
                dep_var,
                exog_vars,
                entity=entity,
                time=time,
                leads=leads,
            )
        )

    return outputs
