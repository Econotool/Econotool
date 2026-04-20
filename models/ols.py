"""OLS, WLS, and formula-based estimators.

All estimators return a normalised :class:`RegressionResult` which wraps
the underlying statsmodels result via ``.raw``.

Public API
----------
fit_ols(df, dep_var, exog_vars, *, add_constant, cov_type, ...) -> RegressionResult
fit_wls(df, dep_var, exog_vars, weights, *, ...) -> RegressionResult
fit_ols_formula(df, formula, *, cov_type, ...) -> RegressionResult
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from econtools.inference.se_types import resolve_cov_args
from econtools.models._results import FitMetrics, RegressionResult


def fit_ols(
    df: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    *,
    add_constant: bool = True,
    cov_type: str = "classical",
    maxlags: int | None = None,
    groups: object = None,
    alpha: float = 0.05,
) -> RegressionResult:
    """Fit OLS and return a :class:`RegressionResult`.

    Parameters
    ----------
    df:
        Input DataFrame (rows with NaN in relevant columns are dropped).
    dep_var:
        Name of the dependent variable column.
    exog_vars:
        List of regressor column names (intercept added automatically
        unless ``add_constant=False``).
    add_constant:
        Prepend a constant column (named ``'const'``) to the design matrix.
    cov_type:
        Covariance estimator label — one of ``VALID_COV_TYPES``.
    maxlags:
        Maximum lags for HAC/Newey-West SEs.
    groups:
        Cluster variable for clustered SEs.
    alpha:
        Significance level for confidence intervals (default 0.05 → 95% CI).

    Returns
    -------
    RegressionResult
    """
    cols = [dep_var] + list(exog_vars)
    data = df[cols].dropna()

    y: pd.Series = data[dep_var]
    X: pd.DataFrame = data[list(exog_vars)]

    if add_constant:
        X = sm.add_constant(X)

    cov_args = resolve_cov_args(cov_type, maxlags=maxlags, groups=groups)
    sm_result = sm.OLS(y, X).fit(**cov_args)

    return _build_result(sm_result, "OLS", dep_var, cov_type, alpha)


def fit_wls(
    df: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    weights: str,
    *,
    add_constant: bool = True,
    cov_type: str = "classical",
    maxlags: int | None = None,
    groups: object = None,
    alpha: float = 0.05,
) -> RegressionResult:
    """Fit WLS and return a :class:`RegressionResult`.

    Parameters
    ----------
    df:
        Input DataFrame.
    dep_var:
        Dependent variable column name.
    exog_vars:
        Regressor column names.
    weights:
        Column name containing observation weights.
    add_constant:
        Prepend a constant column.
    cov_type:
        Covariance estimator label.
    maxlags:
        Max lags for HAC SEs.
    groups:
        Cluster variable for clustered SEs.
    alpha:
        Significance level for confidence intervals.

    Returns
    -------
    RegressionResult
    """
    cols = [dep_var] + list(exog_vars) + [weights]
    data = df[cols].dropna()

    y: pd.Series = data[dep_var]
    X: pd.DataFrame = data[list(exog_vars)]
    w: pd.Series = data[weights]

    if add_constant:
        X = sm.add_constant(X)

    cov_args = resolve_cov_args(cov_type, maxlags=maxlags, groups=groups)
    sm_result = sm.WLS(y, X, weights=w).fit(**cov_args)

    return _build_result(sm_result, "WLS", dep_var, cov_type, alpha)


def fit_ols_formula(
    df: pd.DataFrame,
    formula: str,
    *,
    cov_type: str = "classical",
    maxlags: int | None = None,
    groups: object = None,
    alpha: float = 0.05,
) -> RegressionResult:
    """Fit OLS via Patsy formula and return a :class:`RegressionResult`.

    The constant is controlled by the formula (patsy names it ``'Intercept'``).

    Parameters
    ----------
    df:
        Input DataFrame.
    formula:
        Patsy formula string, e.g. ``'y ~ x1 + x2'``.
    cov_type:
        Covariance estimator label.
    maxlags:
        Max lags for HAC SEs.
    groups:
        Cluster variable for clustered SEs.
    alpha:
        Significance level for confidence intervals.

    Returns
    -------
    RegressionResult
    """
    dep_var = formula.split("~")[0].strip()
    cov_args = resolve_cov_args(cov_type, maxlags=maxlags, groups=groups)
    sm_result = smf.ols(formula, data=df).fit(**cov_args)
    return _build_result(sm_result, "OLS", dep_var, cov_type, alpha, formula=formula)


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------


def _build_result(
    sm_result: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
    formula: str | None = None,
) -> RegressionResult:
    """Populate a :class:`RegressionResult` from a fitted statsmodels result."""
    ci = sm_result.conf_int(alpha=alpha)

    fvalue = sm_result.fvalue
    f_pvalue = sm_result.f_pvalue
    f_stat = float(np.squeeze(fvalue)) if fvalue is not None else float("nan")
    f_pval = float(np.squeeze(f_pvalue)) if f_pvalue is not None else float("nan")

    fit = FitMetrics(
        r_squared=float(sm_result.rsquared),
        r_squared_adj=float(sm_result.rsquared_adj),
        aic=float(sm_result.aic),
        bic=float(sm_result.bic),
        log_likelihood=float(sm_result.llf),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=float(np.sqrt(sm_result.mse_resid)),
        ssr=float(sm_result.ssr),
        ess=float(sm_result.ess),
        tss=float(sm_result.centered_tss),
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
    )

    return RegressionResult(
        model_type=model_type,
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.bse,
        tvalues=sm_result.tvalues,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=sm_result.resid,
        fitted=sm_result.fittedvalues,
        cov_params=sm_result.cov_params(),
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=formula,
    )
