"""Stationarity tests: ADF, KPSS, Phillips-Perron."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
try:
    from arch.unitroot import PhillipsPerron as _PhillipsPerron
except Exception:  # pragma: no cover - optional dependency
    _PhillipsPerron = None

from econtools.inference.hypothesis import TestResult


def adf_test(
    series: Iterable[float] | pd.Series,
    *,
    maxlag: int | None = None,
    autolag: str | None = "AIC",
    regression: str = "c",
) -> TestResult:
    """Augmented Dickey-Fuller test.

    H0: unit root (non-stationary).
    """
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 5:
        raise ValueError("Series too short for ADF test.")
    stat, pval, usedlag, nobs, _crit, _icbest = adfuller(
        y, maxlag=maxlag, autolag=autolag, regression=regression
    )
    return TestResult(
        test_name="ADF",
        statistic=float(stat),
        pvalue=float(pval),
        df=float(usedlag),
        distribution="ADF",
        null_hypothesis="Unit root (non-stationary)",
        reject=float(pval) < 0.05,
    )


def kpss_test(
    series: Iterable[float] | pd.Series,
    *,
    regression: str = "c",
    nlags: str | int = "auto",
) -> TestResult:
    """KPSS test.

    H0: level (or trend) stationarity.
    """
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 5:
        raise ValueError("Series too short for KPSS test.")
    stat, pval, usedlags, _crit = kpss(y, regression=regression, nlags=nlags)
    return TestResult(
        test_name="KPSS",
        statistic=float(stat),
        pvalue=float(pval),
        df=float(usedlags),
        distribution="KPSS",
        null_hypothesis="Stationary (level or trend)",
        reject=float(pval) < 0.05,
    )


def pp_test(
    series: Iterable[float] | pd.Series,
    *,
    lags: int | None = None,
    trend: str = "c",
) -> TestResult:
    """Phillips-Perron test.

    H0: unit root (non-stationary).
    """
    if _PhillipsPerron is None:
        raise RuntimeError("Phillips-Perron test requires the 'arch' package.")
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 5:
        raise ValueError("Series too short for PP test.")
    pp = _PhillipsPerron(y, lags=lags, trend=trend)
    stat = pp.stat
    pval = pp.pvalue
    usedlag = pp.lags
    return TestResult(
        test_name="Phillips-Perron",
        statistic=float(stat),
        pvalue=float(pval),
        df=float(usedlag),
        distribution="PP",
        null_hypothesis="Unit root (non-stationary)",
        reject=float(pval) < 0.05,
    )
