"""Panel estimators (first differences, pooled OLS, FE/RE later)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from linearmodels.panel import FirstDifferenceOLS

from econtools.models._results import FitMetrics, RegressionResult


def fit_first_difference(
    df: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    *,
    entity: str,
    time: str,
    add_constant: bool = False,
    cov_type: str = "unadjusted",
    alpha: float = 0.05,
) -> RegressionResult:
    """Fit a first-differenced panel model using linearmodels."""
    cols = [dep_var] + list(exog_vars) + [entity, time]
    data = df[cols].dropna()
    panel = data.set_index([entity, time]).sort_index()

    y: pd.Series = panel[dep_var]
    X: pd.DataFrame = panel[list(exog_vars)]
    if add_constant:
        X = X.assign(const=1.0)

    res = FirstDifferenceOLS(y, X).fit(cov_type=cov_type)
    return _build_panel_result(res, "FirstDifference", dep_var, cov_type, alpha)


def _build_panel_result(
    res: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
) -> RegressionResult:
    ci = res.conf_int(level=1 - alpha)

    f_stat = float("nan")
    f_pval = float("nan")
    f_obj = getattr(res, "f_statistic", None)
    if f_obj is not None:
        f_stat = float(getattr(f_obj, "stat", f_stat))
        f_pval = float(getattr(f_obj, "pval", f_pval))

    resid = res.resids
    ssr = float(np.sum(np.asarray(resid) ** 2))
    y = np.asarray(res.model.dependent.values2d).squeeze()
    tss = float(np.sum((y - y.mean()) ** 2))
    ess = float(tss - ssr)
    rmse = math.sqrt(ssr / len(y))

    fit = FitMetrics(
        r_squared=float(res.rsquared),
        r_squared_adj=float("nan"),
        aic=float("nan"),
        bic=float("nan"),
        log_likelihood=float("nan"),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=rmse,
        ssr=ssr,
        ess=ess,
        tss=tss,
        nobs=int(res.nobs),
        df_model=float(res.df_model),
        df_resid=float(res.df_resid),
    )

    cov = res.cov

    return RegressionResult(
        model_type=model_type,
        dep_var=dep_var,
        params=res.params,
        bse=res.std_errors,
        tvalues=res.tstats,
        pvalues=res.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=res.resids,
        fitted=res.fitted_values,
        cov_params=cov,
        cov_type=cov_type_label,
        fit=fit,
        raw=res,
        formula=None,
    )
