"""Specification tests for functional form and structural stability.

Public API
----------
reset_test(result, power=3, use_f=True) -> TestResult
harvey_collier(result) -> TestResult
chow_test(result_pooled, result_1, result_2) -> TestResult
"""

from __future__ import annotations

import numpy as np
import scipy.stats
import statsmodels.stats.diagnostic as smsd

from econtools.inference.hypothesis import TestResult
from econtools.models._results import RegressionResult


def reset_test(
    result: RegressionResult,
    power: int = 3,
    use_f: bool = True,
) -> TestResult:
    """Ramsey RESET test for functional form misspecification.

    Adds powers of the fitted values (up to ``power``) as additional
    regressors and tests whether their coefficients are jointly zero.

    H₀: functional form is correctly specified.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    power:
        Highest power of fitted values to include (default 3).
    use_f:
        Report an F-statistic if True (default), Chi² otherwise.

    Returns
    -------
    TestResult
    """
    sm_test = smsd.linear_reset(result.raw, power=power, use_f=use_f)
    stat = float(np.squeeze(sm_test.statistic))
    pval = float(np.squeeze(sm_test.pvalue))

    return TestResult(
        test_name="RESET",
        statistic=stat,
        pvalue=pval,
        df=None,
        distribution="F" if use_f else "Chi2",
        null_hypothesis="Functional form is correctly specified",
        reject=pval < 0.05,
    )


def harvey_collier(result: RegressionResult) -> TestResult:
    """Harvey-Collier t-test for linearity.

    Tests whether the true relationship is linear by regressing
    recursive residuals on a constant.

    H₀: the functional form is linear.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with t-statistic and p-value.
    """
    t_stat, p_val = smsd.linear_harvey_collier(result.raw)
    return TestResult(
        test_name="Harvey-Collier",
        statistic=float(t_stat),
        pvalue=float(p_val),
        df=None,
        distribution="t",
        null_hypothesis="Linear functional form",
        reject=float(p_val) < 0.05,
    )


def chow_test(
    result_pooled: RegressionResult,
    result_1: RegressionResult,
    result_2: RegressionResult,
) -> TestResult:
    """Chow test for structural break (parameter stability across subsamples).

    Tests whether regression coefficients are the same in two subsamples
    by comparing the restricted (pooled) and unrestricted (separate) SSRs.

    H₀: same coefficients in both subsamples (no structural break).

    F = [(SSR_pooled - SSR_1 - SSR_2) / k] / [(SSR_1 + SSR_2) / (n - 2k)]

    Parameters
    ----------
    result_pooled:
        OLS result from the full (pooled) sample.
    result_1:
        OLS result from subsample 1.
    result_2:
        OLS result from subsample 2.

    Returns
    -------
    TestResult with F-statistic and p-value.
    """
    ssr_p = float(result_pooled.raw.ssr)
    ssr_1 = float(result_1.raw.ssr)
    ssr_2 = float(result_2.raw.ssr)

    k = int(result_pooled.raw.df_model) + 1  # includes intercept
    n = int(result_pooled.raw.nobs)

    df_num = k
    df_den = n - 2 * k

    if df_den <= 0:
        raise ValueError(
            f"Insufficient observations for Chow test: n={n}, k={k}, "
            f"need n > 2k={2*k}."
        )

    f_stat = ((ssr_p - ssr_1 - ssr_2) / df_num) / ((ssr_1 + ssr_2) / df_den)
    p_val = float(scipy.stats.f.sf(f_stat, df_num, df_den))

    return TestResult(
        test_name="Chow",
        statistic=float(f_stat),
        pvalue=p_val,
        df=(float(df_num), float(df_den)),
        distribution="F",
        null_hypothesis="No structural break (same coefficients in both subsamples)",
        reject=p_val < 0.05,
        details={
            "ssr_pooled": ssr_p,
            "ssr_1": ssr_1,
            "ssr_2": ssr_2,
            "n": n,
            "k": k,
        },
    )
