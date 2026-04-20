"""Normality tests for regression residuals.

Public API
----------
jarque_bera(result) -> TestResult
shapiro_wilk(result) -> TestResult
omnibus_test(result) -> TestResult
"""

from __future__ import annotations

import numpy as np
import scipy.stats
from statsmodels.stats.stattools import jarque_bera as _sm_jarque_bera
from statsmodels.stats.stattools import omni_normtest as _sm_omni

from econtools.inference.hypothesis import TestResult
from econtools.models._results import RegressionResult


def jarque_bera(result: RegressionResult) -> TestResult:
    """Jarque-Bera test for normality of residuals.

    H₀: residuals are normally distributed (skewness=0, excess kurtosis=0).

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with JB statistic and p-value (Chi²(2) distribution).
    """
    jb_stat, jb_pval, _skew, _kurtosis = _sm_jarque_bera(result.resid)
    return TestResult(
        test_name="Jarque-Bera",
        statistic=float(jb_stat),
        pvalue=float(jb_pval),
        df=2,
        distribution="Chi2",
        null_hypothesis="Residuals are normally distributed",
        reject=float(jb_pval) < 0.05,
    )


def shapiro_wilk(result: RegressionResult) -> TestResult:
    """Shapiro-Wilk test for normality of residuals.

    H₀: residuals are drawn from a normal distribution.  Best suited for
    small samples (n < 5000).

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with W-statistic and p-value.
    """
    resid = np.asarray(result.resid, dtype=float)
    # scipy shapiro has a 5000-obs limit; truncate with warning-free approach
    if len(resid) > 5000:
        resid = resid[:5000]
    w_stat, p_val = scipy.stats.shapiro(resid)
    return TestResult(
        test_name="Shapiro-Wilk",
        statistic=float(w_stat),
        pvalue=float(p_val),
        df=None,
        distribution="Shapiro-Wilk",
        null_hypothesis="Residuals are normally distributed",
        reject=float(p_val) < 0.05,
    )


def omnibus_test(result: RegressionResult) -> TestResult:
    """D'Agostino-Pearson omnibus test for normality of residuals.

    Combines tests for skewness and kurtosis into a single Chi²(2) test.

    H₀: residuals are normally distributed.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with omnibus statistic and p-value (Chi²(2)).
    """
    stat, pval = _sm_omni(result.resid)
    return TestResult(
        test_name="Omnibus",
        statistic=float(stat),
        pvalue=float(pval),
        df=2,
        distribution="Chi2",
        null_hypothesis="Residuals are normally distributed",
        reject=float(pval) < 0.05,
    )
