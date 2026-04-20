"""Heteroskedasticity tests: Breusch-Pagan and White.

Public API
----------
breusch_pagan(result) -> TestResult
white_test(result)    -> TestResult
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import statsmodels.stats.diagnostic as smsd

from econtools.inference.hypothesis import TestResult
from econtools.models._results import RegressionResult


def breusch_pagan(result: RegressionResult) -> TestResult:
    """Breusch-Pagan test for heteroskedasticity (LM version).

    H₀: homoskedasticity (errors have constant variance).

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with LM statistic and p-value.
    """
    lm_stat, lm_pval, _f_stat, _f_pval = smsd.het_breuschpagan(
        result.resid, result.raw.model.exog
    )
    return TestResult(
        test_name="Breusch-Pagan",
        statistic=float(lm_stat),
        pvalue=float(lm_pval),
        df=None,
        distribution="Chi2",
        null_hypothesis="Homoskedasticity",
        reject=float(lm_pval) < 0.05,
    )


def white_test(result: RegressionResult) -> TestResult:
    """White test for heteroskedasticity (LM version).

    H₀: homoskedasticity.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult with LM statistic and p-value.  On a wide design where the
    auxiliary regression (squares + pairwise interactions) becomes
    rank-deficient or over-parameterised, this function returns a sentinel
    :class:`TestResult` with ``statistic=nan``, ``pvalue=nan``,
    ``reject=False`` and an ``error`` entry in ``details`` — rather than
    raising — so downstream paper/diagnostic tables degrade gracefully.
    """
    exog = result.raw.model.exog
    try:
        nobs, nvars0 = exog.shape
    except Exception:
        nobs, nvars0 = (0, 0)
    # Number of regressors in the auxiliary regression = k + k*(k-1)/2 + k
    # (squares + interactions + originals).  statsmodels builds this via
    # np.triu_indices so the count is nvars0*(nvars0+1)/2.
    try:
        p_aux = int(nvars0 * (nvars0 + 1) // 2)
    except Exception:
        p_aux = 0

    try:
        lm_stat, lm_pval, _f_stat, _f_pval = smsd.het_white(
            result.resid, exog
        )
    except (AssertionError, np.linalg.LinAlgError, ValueError) as err:
        warnings.warn(
            f"White test skipped: auxiliary regression degenerate "
            f"(n={nobs}, p_aux={p_aux}): {err}",
            RuntimeWarning,
            stacklevel=2,
        )
        return TestResult(
            test_name="White",
            statistic=float("nan"),
            pvalue=float("nan"),
            df=None,
            distribution="Chi2",
            null_hypothesis="Homoskedasticity",
            reject=False,
            details={
                "error": (
                    f"auxiliary regression degenerate: n={nobs}, "
                    f"p_aux={p_aux}"
                ),
                "exception": f"{type(err).__name__}: {err}",
            },
        )

    lm_stat_f = float(lm_stat)
    lm_pval_f = float(lm_pval)
    if math.isnan(lm_stat_f) or math.isnan(lm_pval_f):
        warnings.warn(
            f"White test returned NaN (n={nobs}, p_aux={p_aux}); "
            "auxiliary regression likely degenerate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return TestResult(
            test_name="White",
            statistic=lm_stat_f,
            pvalue=lm_pval_f,
            df=None,
            distribution="Chi2",
            null_hypothesis="Homoskedasticity",
            reject=False,
            details={
                "error": (
                    f"auxiliary regression degenerate: n={nobs}, "
                    f"p_aux={p_aux}"
                ),
            },
        )
    return TestResult(
        test_name="White",
        statistic=lm_stat_f,
        pvalue=lm_pval_f,
        df=None,
        distribution="Chi2",
        null_hypothesis="Homoskedasticity",
        reject=lm_pval_f < 0.05,
    )


def goldfeld_quandt(
    result: RegressionResult,
    *,
    split: float = 0.5,
    alternative: str = "increasing",
) -> TestResult:
    """Goldfeld-Quandt test for heteroskedasticity.

    Splits the sample (ordered by a regressor) and tests whether the
    residual variance is equal in both subsamples.

    H₀: equal error variance in the two subsamples.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    split:
        Fraction at which to split the data (default 0.5).
    alternative:
        ``'increasing'``, ``'decreasing'``, or ``'two-sided'`` (default
        ``'increasing'``).

    Returns
    -------
    TestResult with F-statistic and p-value.
    """
    f_stat, f_pval, ordering = smsd.het_goldfeldquandt(
        result.raw.model.endog,
        result.raw.model.exog,
        split=split,
        alternative=alternative,
    )
    return TestResult(
        test_name="Goldfeld-Quandt",
        statistic=float(f_stat),
        pvalue=float(f_pval),
        df=None,
        distribution="F",
        null_hypothesis="Homoskedasticity (equal variance in subsamples)",
        reject=float(f_pval) < 0.05,
        details={"ordering": ordering, "split": split},
    )
