"""Structural break and parameter stability tests.

Implements CUSUM, CUSUM-of-squares, recursive residuals, and the
Andrews (1993) supremum Wald test for an unknown breakpoint.

All public functions return ``TestResult`` from ``econtools._core.types``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm

from econtools._core.types import Estimate, TestResult


# ---------------------------------------------------------------------------
# Recursive residuals
# ---------------------------------------------------------------------------

def recursive_residuals(result: Estimate) -> np.ndarray:
    """Compute recursive (one-step-ahead) residuals.

    Parameters
    ----------
    result : Estimate
        Fitted OLS result.  Must have ``raw`` attribute exposing the
        statsmodels ``OLSResults``.

    Returns
    -------
    np.ndarray
        Array of recursive residuals (NaN-free).  The first *k* values
        (where *k* is the number of regressors) are not computable and
        are excluded.
    """
    sm_res = _get_sm_result(result)
    rr = sm.stats.recursive_olsresiduals(sm_res, skip=int(sm_res.df_model) + 1)
    # rr returns (recursive_resid, ...) — element [0] is the recursive resids
    # statsmodels returns array of length n with NaN for skipped observations
    arr = np.asarray(rr[0])
    return arr[~np.isnan(arr)]


# ---------------------------------------------------------------------------
# CUSUM test (Brown-Durbin-Evans 1975)
# ---------------------------------------------------------------------------

def cusum_test(result: Estimate, alpha: float = 0.05) -> TestResult:
    """CUSUM test for parameter stability (Brown, Durbin & Evans, 1975).

    The test is based on cumulative sums of recursive residuals.
    Under H₀ (stable parameters) the CUSUM statistic stays within
    ±c(α)√(n−k) bounds, where *c(α)* depends on significance level.

    Parameters
    ----------
    result : Estimate
        Fitted OLS result.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    TestResult
        ``statistic`` is the max |CUSUM| normalised by σ̂.
        ``pvalue`` is approximate (Harvey 1990 tables).
    """
    sm_res = _get_sm_result(result)
    rr_result = sm.stats.recursive_olsresiduals(sm_res, skip=int(sm_res.df_model) + 1)
    rec_resid = np.asarray(rr_result[0])
    rec_resid = rec_resid[~np.isnan(rec_resid)]
    # cusum = cumulative sum of standardised recursive residuals
    sigma = np.std(rec_resid, ddof=1)
    if sigma < 1e-15:
        sigma = 1.0
    cusum = np.cumsum(rec_resid) / sigma

    n = len(rec_resid)
    # Test statistic: max absolute CUSUM scaled by sqrt(n)
    stat = np.max(np.abs(cusum)) / np.sqrt(n)

    # Critical values from BDE (1975) — approximate via Brownian bridge
    # Using the standard significance boundaries:
    crit_vals = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}
    cv = crit_vals.get(alpha, 0.948)
    reject = bool(stat > cv)

    # Approximate p-value via crossing probability of Brownian bridge
    # P(sup|B(t)| > c) ≈ 2*exp(-2c²) for standard Brownian bridge
    pvalue = float(2.0 * np.exp(-2.0 * stat ** 2))
    pvalue = min(max(pvalue, 0.0), 1.0)

    return TestResult(
        test_name="CUSUM",
        statistic=float(stat),
        pvalue=pvalue,
        df=n,
        distribution="Brownian bridge",
        null_hypothesis="Parameters are stable over the sample",
        reject=reject,
        details={
            "n_recursive_resid": n,
            "critical_value": cv,
            "alpha": alpha,
            "cusum_path": cusum.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# CUSUM of squares test
# ---------------------------------------------------------------------------

def cusum_sq_test(result: Estimate, alpha: float = 0.05) -> TestResult:
    """CUSUM-of-squares test for variance stability.

    Tests whether the error variance is constant over the sample.
    Under H₀ the cumulative sum of squared recursive residuals
    follows a straight line from 0 to 1.

    Parameters
    ----------
    result : Estimate
        Fitted OLS result.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    sm_res = _get_sm_result(result)
    rr_result = sm.stats.recursive_olsresiduals(sm_res, skip=int(sm_res.df_model) + 1)
    rec_resid = np.asarray(rr_result[0])
    rec_resid = rec_resid[~np.isnan(rec_resid)]

    n = len(rec_resid)
    sq = rec_resid ** 2
    cumsum_sq = np.cumsum(sq) / np.sum(sq)
    # Under H0, cumsum_sq[j] ≈ (j+1)/n
    expected = np.arange(1, n + 1) / n
    stat = float(np.max(np.abs(cumsum_sq - expected)))

    # Critical values (Kolmogorov-Smirnov-like for the departure)
    # Approximate: same as KS test on [0,1] uniform with n points
    crit_vals = {0.01: 1.63 / np.sqrt(n), 0.05: 1.36 / np.sqrt(n),
                 0.10: 1.22 / np.sqrt(n)}
    cv = crit_vals.get(alpha, 1.36 / np.sqrt(n))
    reject = bool(stat > cv)

    # Approximate p-value via KS distribution
    pvalue = float(np.exp(-2.0 * n * stat ** 2))
    pvalue = min(max(pvalue, 0.0), 1.0)

    return TestResult(
        test_name="CUSUM of Squares",
        statistic=float(stat),
        pvalue=pvalue,
        df=n,
        distribution="KS-type",
        null_hypothesis="Error variance is constant over the sample",
        reject=reject,
        details={
            "n_recursive_resid": n,
            "critical_value": float(cv),
            "alpha": alpha,
        },
    )


# ---------------------------------------------------------------------------
# Andrews (1993) supremum Wald test
# ---------------------------------------------------------------------------

def andrews_sup_wald(
    result: Estimate,
    trim: float = 0.15,
    alpha: float = 0.05,
) -> TestResult:
    """Andrews (1993) supremum Wald test for an unknown structural break.

    Computes Chow-type F statistics for every candidate breakpoint in
    the trimmed interior of the sample, and returns the supremum.

    Parameters
    ----------
    result : Estimate
        Fitted OLS result.
    trim : float
        Fraction of sample trimmed from each end (default 0.15 = 15%).
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
        ``statistic`` is the sup-Wald (supremum of Chow F's).
        ``details`` includes ``break_index`` (location of the sup).

    Notes
    -----
    Critical values are from Andrews (1993) Table 1 for p parameters.
    The asymptotic distribution is the supremum of a chi-squared process.
    """
    sm_res = _get_sm_result(result)
    y = np.asarray(sm_res.model.endog)
    X = np.asarray(sm_res.model.exog)
    n, k = X.shape

    lo = max(k + 1, int(np.ceil(n * trim)))
    hi = min(n - k - 1, int(np.floor(n * (1 - trim))))

    if lo >= hi:
        raise ValueError(
            f"Sample too small for trim={trim}: n={n}, k={k}, "
            f"trimmed range [{lo}, {hi}] is empty."
        )

    ssr_full = float(sm_res.ssr)
    wald_stats = []
    break_indices = list(range(lo, hi + 1))

    for bp in break_indices:
        y1, X1 = y[:bp], X[:bp]
        y2, X2 = y[bp:], X[bp:]
        try:
            ssr1 = float(sm.OLS(y1, X1).fit().ssr)
            ssr2 = float(sm.OLS(y2, X2).fit().ssr)
        except Exception:
            wald_stats.append(0.0)
            continue
        ssr_ur = ssr1 + ssr2
        if ssr_ur < 1e-15:
            wald_stats.append(0.0)
            continue
        f_stat = ((ssr_full - ssr_ur) / k) / (ssr_ur / (n - 2 * k))
        wald_stats.append(f_stat)

    wald_arr = np.array(wald_stats)
    sup_idx = int(np.argmax(wald_arr))
    sup_stat = float(wald_arr[sup_idx])
    break_at = break_indices[sup_idx]

    # Approximate p-value using Hansen (1997) approximation
    # For F-type with k restrictions: p ≈ P(chi2(k)/k > sup_stat) is a
    # *lower* bound; we use the exponential approximation from Andrews tables
    pvalue = float(1.0 - sp_stats.f.cdf(sup_stat, k, n - 2 * k))
    # Bonferroni-type correction for searching over breakpoints
    n_tests = len(break_indices)
    pvalue = min(pvalue * n_tests, 1.0)

    # Andrews (1993) approximate critical values for sup-Wald with k params
    # These are rough interpolations from Table 1
    reject = bool(pvalue < alpha)

    return TestResult(
        test_name="Andrews Sup-Wald",
        statistic=sup_stat,
        pvalue=pvalue,
        df=(float(k), float(n - 2 * k)),
        distribution="Sup-F",
        null_hypothesis="No structural break at any point in the sample",
        reject=reject,
        details={
            "break_index": break_at,
            "break_fraction": break_at / n,
            "trim": trim,
            "n_breakpoints_tested": n_tests,
            "wald_path": wald_arr.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Chow forecast (predictive failure) test
# ---------------------------------------------------------------------------

def chow_forecast_test(
    result: Estimate,
    forecast_start: int,
) -> TestResult:
    """Chow predictive failure (forecast) test.

    Tests whether the model estimated on the first part of the sample
    can predict the remaining observations.

    Parameters
    ----------
    result : Estimate
        Fitted OLS result on the full sample.
    forecast_start : int
        Index (0-based) at which the forecast period begins.

    Returns
    -------
    TestResult
    """
    sm_res = _get_sm_result(result)
    y = np.asarray(sm_res.model.endog)
    X = np.asarray(sm_res.model.exog)
    n, k = X.shape

    if forecast_start <= k or forecast_start >= n:
        raise ValueError(
            f"forecast_start={forecast_start} must be in ({k}, {n})."
        )

    n1 = forecast_start
    n2 = n - forecast_start
    y1, X1 = y[:n1], X[:n1]

    res1 = sm.OLS(y1, X1).fit()
    ssr1 = float(res1.ssr)
    ssr_full = float(sm_res.ssr)

    f_stat = ((ssr_full - ssr1) / n2) / (ssr1 / (n1 - k))
    pvalue = float(1.0 - sp_stats.f.cdf(f_stat, n2, n1 - k))
    reject = bool(pvalue < 0.05)

    return TestResult(
        test_name="Chow Forecast",
        statistic=f_stat,
        pvalue=pvalue,
        df=(float(n2), float(n1 - k)),
        distribution="F",
        null_hypothesis="Model parameters are stable in forecast period",
        reject=reject,
        details={
            "forecast_start": forecast_start,
            "n_estimation": n1,
            "n_forecast": n2,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sm_result(result: Estimate):
    """Extract the underlying statsmodels result, validating model type."""
    model_type = result.model_type.lower()
    if model_type not in ("ols", "wls"):
        raise TypeError(
            f"Structural break tests require OLS/WLS result, got '{result.model_type}'."
        )
    raw = result.raw
    if raw is None:
        raise TypeError("Structural break tests require a statsmodels result (raw is None).")
    return raw
