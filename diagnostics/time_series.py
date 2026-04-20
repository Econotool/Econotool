"""Time-series diagnostics: Granger causality and exogeneity checks."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from econtools.inference.hypothesis import TestResult


def select_var_lag(
    df: pd.DataFrame,
    *,
    maxlags: int,
    ic: str = "aic",
) -> int:
    """Select VAR lag order by information criterion."""
    model = VAR(df)
    res = model.select_order(maxlags)
    ic = ic.lower()
    selected = getattr(res, ic)
    if selected is None:
        raise ValueError(f"Could not select lag using ic='{ic}'.")
    return int(selected)


def granger_causality(
    y: Iterable[float] | pd.Series,
    x: Iterable[float] | pd.Series,
    *,
    maxlags: int,
    ic: str | None = "aic",
    test: str = "ssr_ftest",
) -> TestResult:
    """Granger causality test: does x help predict y?"""
    y_ser = pd.Series(list(y), dtype=float)
    x_ser = pd.Series(list(x), dtype=float)
    df = pd.DataFrame({"y": y_ser, "x": x_ser}).dropna()

    if ic is not None:
        lag = select_var_lag(df[["y", "x"]], maxlags=maxlags, ic=ic)
    else:
        lag = maxlags

    res = grangercausalitytests(df[["y", "x"]], maxlag=lag, verbose=False)
    stat, pval, df_denom, df_num = res[lag][0][test]
    return TestResult(
        test_name=f"Granger ({test})",
        statistic=float(stat),
        pvalue=float(pval),
        df=(float(df_num), float(df_denom)),
        distribution="F",
        null_hypothesis="x does not Granger-cause y",
        reject=float(pval) < 0.05,
    )


def lead_exogeneity_test(
    y: Iterable[float] | pd.Series,
    x: Iterable[float] | pd.Series,
    *,
    lead: int = 1,
    lags: int = 1,
    add_trend: bool = True,
) -> TestResult:
    """Test strict exogeneity by checking lead(s) of x in y equation.

    H0: coefficients on lead(x) are zero.
    """
    y_ser = pd.Series(list(y), dtype=float)
    x_ser = pd.Series(list(x), dtype=float)
    df = pd.DataFrame({"y": y_ser, "x": x_ser})

    # Build lags and leads
    for k in range(1, lags + 1):
        df[f"x_lag{k}"] = df["x"].shift(k)
    for k in range(1, lead + 1):
        df[f"x_lead{k}"] = df["x"].shift(-k)
    if add_trend:
        df["trend"] = np.arange(len(df), dtype=float)

    df = df.dropna()
    exog_cols = ["x"] + [f"x_lag{k}" for k in range(1, lags + 1)] + [
        f"x_lead{k}" for k in range(1, lead + 1)
    ]
    if add_trend:
        exog_cols = ["trend"] + exog_cols

    exog = sm.add_constant(df[exog_cols], has_constant="add")
    fit = sm.OLS(df["y"], exog).fit()

    # Test lead coefficients jointly zero
    lead_cols = [f"x_lead{k}" for k in range(1, lead + 1)]
    R = np.zeros((len(lead_cols), exog.shape[1]))
    col_index = {name: i for i, name in enumerate(exog.columns)}
    for i, name in enumerate(lead_cols):
        R[i, col_index[name]] = 1.0
    test = fit.f_test(R)

    return TestResult(
        test_name="Lead exogeneity (F)",
        statistic=float(np.squeeze(test.fvalue)),
        pvalue=float(np.squeeze(test.pvalue)),
        df=(float(test.df_num), float(test.df_denom)),
        distribution="F",
        null_hypothesis="Leads of x have zero coefficients",
        reject=float(np.squeeze(test.pvalue)) < 0.05,
    )
