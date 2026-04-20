"""Probit estimator wrapper.

Public API
----------
fit_probit(df, dep_var, exog_vars, *, add_constant, cov_type, ...) -> RegressionResult
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm

from econtools.inference.se_types import resolve_cov_args
from econtools.models._results import FitMetrics, RegressionResult


def fit_probit(
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
    """Fit a Probit model and return a :class:`RegressionResult`."""
    cols = [dep_var] + list(exog_vars)
    data = df[cols].dropna()

    y: pd.Series = data[dep_var]
    X: pd.DataFrame = data[list(exog_vars)]

    if add_constant:
        X = sm.add_constant(X)

    sm_result = sm.Probit(y, X).fit(disp=False)

    if cov_type != "classical":
        cov_args = resolve_cov_args(cov_type, maxlags=maxlags, groups=groups)
        cov_type_sm = cov_args.get("cov_type", "nonrobust")
        cov_kwds = cov_args.get("cov_kwds", {})
        sm_result = sm_result.get_robustcov_results(
            cov_type=cov_type_sm, **cov_kwds
        )

    return _build_probit_result(sm_result, dep_var, cov_type, alpha)


def _build_probit_result(
    sm_result: object,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
) -> RegressionResult:
    ci = sm_result.conf_int(alpha=alpha)

    resid = getattr(sm_result, "resid_response", None)
    if resid is None:
        resid = getattr(sm_result, "resid", None)
    if resid is None:
        resid = pd.Series(dtype=float)

    fitted = sm_result.predict()

    fit = FitMetrics(
        r_squared=float(getattr(sm_result, "prsquared", math.nan)),
        r_squared_adj=float("nan"),
        aic=float(sm_result.aic),
        bic=float(sm_result.bic),
        log_likelihood=float(sm_result.llf),
        f_stat=float("nan"),
        f_pvalue=float("nan"),
        rmse=float("nan"),
        ssr=float("nan"),
        ess=float("nan"),
        tss=float("nan"),
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
    )

    return RegressionResult(
        model_type="Probit (LDP)",
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.bse,
        tvalues=sm_result.tvalues,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=resid,
        fitted=fitted,
        cov_params=sm_result.cov_params(),
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=None,
    )
