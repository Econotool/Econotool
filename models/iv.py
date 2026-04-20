"""Instrumental variables estimators (2SLS)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from econtools.inference.se_types import VALID_COV_TYPES
from econtools.models._results import FitMetrics, RegressionResult


def fit_iv_2sls(
    df: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    endog_vars: list[str],
    instruments: list[str],
    *,
    add_constant: bool = True,
    cov_type: str = "classical",
    maxlags: int | None = None,
    groups: object = None,
    alpha: float = 0.05,
) -> RegressionResult:
    """Fit 2SLS and return a :class:`RegressionResult`."""
    cols = [dep_var] + list(exog_vars) + list(endog_vars) + list(instruments)
    data = df[cols].dropna()

    y: pd.Series = data[dep_var]
    exog: pd.DataFrame = data[list(exog_vars)]
    endog: pd.DataFrame = data[list(endog_vars)]
    instr: pd.DataFrame = data[list(instruments)]

    if add_constant:
        exog = sm.add_constant(exog)

    cov_args = _resolve_iv_cov_args(cov_type, maxlags=maxlags, groups=groups)
    sm_result = IV2SLS(y, exog, endog, instr).fit(**cov_args)

    return _build_iv_result(sm_result, "2SLS", dep_var, cov_type, alpha, y)


def _resolve_iv_cov_args(
    cov_type: str,
    maxlags: int | None = None,
    groups: Any = None,
) -> dict[str, Any]:
    if cov_type not in VALID_COV_TYPES:
        raise ValueError(
            f"Unknown cov_type '{cov_type}'. "
            f"Choose from: {VALID_COV_TYPES}."
        )

    if cov_type == "classical":
        return {"cov_type": "unadjusted"}

    if cov_type in ("HC0", "HC1", "HC2", "HC3"):
        return {"cov_type": "robust"}

    if cov_type in ("HAC", "newey_west"):
        cov: dict[str, Any] = {"cov_type": "kernel", "kernel": "bartlett"}
        if maxlags is not None:
            cov["bandwidth"] = maxlags
        return cov

    if groups is None:
        raise ValueError("cov_type='cluster' requires the ``groups`` argument.")
    return {"cov_type": "clustered", "clusters": groups}


def _build_iv_result(
    sm_result: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
    y: pd.Series,
) -> RegressionResult:
    try:
        ci = sm_result.conf_int(level=1 - alpha)
    except Exception:
        ci = sm_result.conf_int()

    f_stat = float("nan")
    f_pval = float("nan")
    f_obj = getattr(sm_result, "f_statistic", None)
    if f_obj is not None:
        f_stat = float(getattr(f_obj, "stat", f_stat))
        f_pval = float(getattr(f_obj, "pval", f_pval))

    resid = sm_result.resids
    ssr = float(np.sum(np.asarray(resid) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    ess = float(tss - ssr)
    rmse = math.sqrt(ssr / len(y))

    fit = FitMetrics(
        r_squared=float(sm_result.rsquared),
        r_squared_adj=float(sm_result.rsquared_adj),
        aic=float(getattr(sm_result, "aic", float("nan"))),
        bic=float(getattr(sm_result, "bic", float("nan"))),
        log_likelihood=float(getattr(sm_result, "loglik", float("nan"))),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=rmse,
        ssr=ssr,
        ess=ess,
        tss=tss,
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
    )

    return RegressionResult(
        model_type=model_type,
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.std_errors,
        tvalues=sm_result.tstats,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=sm_result.resids,
        fitted=sm_result.fitted_values,
        cov_params=sm_result.cov,
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=None,
    )
