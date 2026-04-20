"""Hypothesis tests for regression coefficients.

Public API
----------
TestResult                  — frozen dataclass for test output (from _core.types)
wald_test(result, R, q, use_f) -> TestResult
f_test(result, R, q)        -> TestResult
t_test_coeff(result, var_name, value) -> TestResult
lr_test(result_restricted, result_unrestricted) -> TestResult
score_test(result_restricted, exog_extra) -> TestResult
conf_int(result, alpha)     -> pd.DataFrame
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import scipy.stats

from econtools._core.types import RegressionResult, TestResult  # noqa: F401


def wald_test(
    result: RegressionResult,
    R: np.ndarray,
    q: np.ndarray | None = None,
    use_f: bool = True,
) -> TestResult:
    """Wald test for linear restrictions Rβ = q.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    R:
        Restriction matrix of shape (n_restrictions, n_params).
    q:
        RHS vector of shape (n_restrictions,). Defaults to zeros.
    use_f:
        If True, report an F-statistic; otherwise Chi² statistic.

    Returns
    -------
    TestResult
    """
    r_matrix: Union[np.ndarray, tuple] = R if q is None else (R, q)
    # scalar=False suppresses the FutureWarning in statsmodels 0.14
    sm_test = result.raw.wald_test(r_matrix, use_f=use_f, scalar=False)

    stat = float(np.squeeze(sm_test.statistic))
    pval = float(np.squeeze(sm_test.pvalue))

    if use_f:
        df_num = float(sm_test.df_num)
        df_denom = float(sm_test.df_denom)
        df: Union[tuple[float, float], float, None] = (df_num, df_denom)
        dist = "F"
        name = "Wald/F-test"
    else:
        # chi2 ContrastResults exposes df_denom not df_num; derive from R
        R_arr = np.atleast_2d(r_matrix[0] if isinstance(r_matrix, tuple) else r_matrix)
        df = float(R_arr.shape[0])
        dist = "Chi2"
        name = "Wald test (Chi²)"

    return TestResult(
        test_name=name,
        statistic=stat,
        pvalue=pval,
        df=df,
        distribution=dist,
        null_hypothesis="Rβ = q",
        reject=pval < 0.05,
    )


def f_test(
    result: RegressionResult,
    R: np.ndarray,
    q: np.ndarray | None = None,
) -> TestResult:
    """Convenience F-test wrapper around :func:`wald_test`.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    R:
        Restriction matrix.
    q:
        RHS vector (defaults to zeros).

    Returns
    -------
    TestResult
    """
    return wald_test(result, R, q=q, use_f=True)


def t_test_coeff(
    result: RegressionResult,
    var_name: str,
    value: float = 0.0,
) -> TestResult:
    """Two-sided t-test for a single coefficient.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    var_name:
        Regressor name (must be in ``result.params.index``).
    value:
        Null hypothesis value (default 0).

    Returns
    -------
    TestResult

    Raises
    ------
    ValueError
        If ``var_name`` is not in the model.
    """
    if var_name not in result.params.index:
        raise ValueError(
            f"'{var_name}' not found in model params. "
            f"Available: {list(result.params.index)}"
        )

    coeff = float(result.params[var_name])
    se = float(result.bse[var_name])
    stat = (coeff - value) / se
    df_resid = result.fit.df_resid
    pval = float(2.0 * (1.0 - scipy.stats.t.cdf(abs(stat), df=df_resid)))

    return TestResult(
        test_name="t-test",
        statistic=float(stat),
        pvalue=pval,
        df=float(df_resid),
        distribution="t",
        null_hypothesis=f"β[{var_name}] = {value}",
        reject=pval < 0.05,
    )


def lr_test(
    result_restricted: RegressionResult,
    result_unrestricted: RegressionResult,
) -> TestResult:
    """Likelihood-ratio test for nested models."""
    llf_r = float(result_restricted.fit.log_likelihood)
    llf_ur = float(result_unrestricted.fit.log_likelihood)
    stat = 2.0 * (llf_ur - llf_r)

    df = int(len(result_unrestricted.params) - len(result_restricted.params))
    if df <= 0:
        raise ValueError("Unrestricted model must have more parameters than restricted model.")

    pval = float(scipy.stats.chi2.sf(stat, df=df))
    return TestResult(
        test_name="LR test",
        statistic=stat,
        pvalue=pval,
        df=float(df),
        distribution="Chi2",
        null_hypothesis="Restricted vs unrestricted",
        reject=pval < 0.05,
    )


def score_test(
    result_restricted: RegressionResult,
    exog_extra: np.ndarray | pd.DataFrame,
) -> TestResult:
    """Score (LM) test for omitted variables using a restricted model."""
    if not hasattr(result_restricted.raw, "score_test"):
        raise ValueError("Underlying result does not support score_test().")

    stat, pval, df = result_restricted.raw.score_test(exog_extra=exog_extra)
    stat_f = float(np.squeeze(stat))
    pval_f = float(np.squeeze(pval))
    df_f = float(np.squeeze(df))

    return TestResult(
        test_name="Score (LM) test",
        statistic=stat_f,
        pvalue=pval_f,
        df=df_f,
        distribution="Chi2",
        null_hypothesis="Extra regressors jointly zero",
        reject=pval_f < 0.05,
    )


def conf_int(
    result: RegressionResult,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return confidence intervals as a DataFrame.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    alpha:
        Significance level (default 0.05 → 95% CI).

    Returns
    -------
    DataFrame with columns ``'lower'`` and ``'upper'``, indexed by
    regressor name.

    Note
    ----
    The stored CI in the result uses the ``alpha`` passed at fit time.
    This function recomputes the CI at the requested ``alpha`` using the
    raw statsmodels result.
    """
    try:
        ci = result.raw.conf_int(alpha=alpha)
    except TypeError:
        # linearmodels backends use ``level = 1 - alpha`` instead of ``alpha``.
        ci = result.raw.conf_int(level=1.0 - alpha)
    if hasattr(ci, "iloc"):
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
    else:  # numpy array fallback
        lower = ci[:, 0]
        upper = ci[:, 1]
    return pd.DataFrame(
        {"lower": lower, "upper": upper},
        index=result.params.index,
    )
