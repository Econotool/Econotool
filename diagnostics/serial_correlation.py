"""Serial correlation tests for residuals.

Public API
----------
box_pierce_q(result, lags) -> TestResult
ljung_box_q(result, lags)  -> TestResult
box_pierce_from_autocorr(acf, n_obs, lags) -> TestResult
ljung_box_from_autocorr(acf, n_obs, lags)  -> TestResult
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as _acf

from econtools.inference.hypothesis import TestResult
from econtools.models._results import RegressionResult


def box_pierce_q(result: RegressionResult, *, lags: int) -> TestResult:
    """Box-Pierce Q-statistic test for residual autocorrelation.

    H0: no serial correlation up to lag m.
    """
    _validate_lags(lags)
    out = acorr_ljungbox(result.resid, lags=[lags], boxpierce=True)
    # acorr_ljungbox returns a DataFrame in recent statsmodels
    if isinstance(out, pd.DataFrame):
        bp_stat = float(out["bp_stat"].iloc[-1])
        bp_pval = float(out["bp_pvalue"].iloc[-1])
    else:
        _lbq, _lb_p, bpq, bp_p = out
        bp_stat = float(bpq[-1])
        bp_pval = float(bp_p[-1])
    return TestResult(
        test_name="Box-Pierce Q",
        statistic=bp_stat,
        pvalue=bp_pval,
        df=float(lags),
        distribution="Chi2",
        null_hypothesis=f"No serial correlation up to lag {lags}",
        reject=bp_pval < 0.05,
    )


def ljung_box_q(result: RegressionResult, *, lags: int) -> TestResult:
    """Ljung-Box Q-statistic test for residual autocorrelation.

    H0: no serial correlation up to lag m.
    """
    _validate_lags(lags)
    out = acorr_ljungbox(result.resid, lags=[lags])
    if isinstance(out, pd.DataFrame):
        lb_stat = float(out["lb_stat"].iloc[-1])
        lb_pval = float(out["lb_pvalue"].iloc[-1])
    else:
        lbq, lb_p = out
        lb_stat = float(lbq[-1])
        lb_pval = float(lb_p[-1])
    return TestResult(
        test_name="Ljung-Box Q",
        statistic=lb_stat,
        pvalue=lb_pval,
        df=float(lags),
        distribution="Chi2",
        null_hypothesis=f"No serial correlation up to lag {lags}",
        reject=lb_pval < 0.05,
    )


def box_pierce_from_autocorr(
    autocorr: Sequence[float] | np.ndarray,
    n_obs: int,
    *,
    lags: int | None = None,
) -> TestResult:
    """Box-Pierce Q-statistic from provided autocorrelations."""
    acf = _prepare_autocorr(autocorr, lags)
    _validate_n_obs(n_obs, len(acf))
    q = float(n_obs * np.sum(acf**2))
    df = float(len(acf))
    pval = float(scipy.stats.chi2.sf(q, df=df))
    return TestResult(
        test_name="Box-Pierce Q",
        statistic=q,
        pvalue=pval,
        df=df,
        distribution="Chi2",
        null_hypothesis=f"No serial correlation up to lag {int(df)}",
        reject=pval < 0.05,
    )


def ljung_box_from_autocorr(
    autocorr: Sequence[float] | np.ndarray,
    n_obs: int,
    *,
    lags: int | None = None,
) -> TestResult:
    """Ljung-Box Q-statistic from provided autocorrelations."""
    acf = _prepare_autocorr(autocorr, lags)
    _validate_n_obs(n_obs, len(acf))
    lag_idx = np.arange(1, len(acf) + 1, dtype=float)
    q = float(
        n_obs * (n_obs + 2.0) * np.sum(acf**2 / (n_obs - lag_idx))
    )
    df = float(len(acf))
    pval = float(scipy.stats.chi2.sf(q, df=df))
    return TestResult(
        test_name="Ljung-Box Q",
        statistic=q,
        pvalue=pval,
        df=df,
        distribution="Chi2",
        null_hypothesis=f"No serial correlation up to lag {int(df)}",
        reject=pval < 0.05,
    )


def autocorr_from_series(series: Iterable[float], *, lags: int) -> np.ndarray:
    """Compute sample autocorrelations for lag 1..m."""
    _validate_lags(lags)
    vals = np.asarray(list(series), dtype=float)
    acf_vals = _acf(vals, nlags=lags, fft=False)
    return acf_vals[1 : lags + 1]


def _prepare_autocorr(
    autocorr: Sequence[float] | np.ndarray,
    lags: int | None,
) -> np.ndarray:
    acf = np.asarray(autocorr, dtype=float).ravel()
    if acf.size == 0:
        raise ValueError("autocorr must have at least one value.")
    if lags is not None:
        _validate_lags(lags)
        if lags > acf.size:
            raise ValueError("lags exceeds available autocorrelations.")
        acf = acf[:lags]
    return acf


def _validate_lags(lags: int) -> None:
    if lags <= 0:
        raise ValueError("lags must be a positive integer.")


def durbin_watson(result: RegressionResult) -> TestResult:
    """Durbin-Watson test statistic for first-order autocorrelation.

    H₀: no first-order autocorrelation (DW ≈ 2).

    The DW statistic has no closed-form p-value — values near 2 indicate
    no autocorrelation, near 0 indicate positive, near 4 indicate negative.
    The ``reject`` field is set heuristically (DW < 1.5 or DW > 2.5).

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    TestResult
        ``statistic`` is the DW value in [0, 4].  ``pvalue`` is ``NaN``
        (DW has no exact p-value; compare to Durbin-Watson tables).
    """
    from statsmodels.stats.stattools import durbin_watson as _dw

    dw = float(_dw(result.resid))
    return TestResult(
        test_name="Durbin-Watson",
        statistic=dw,
        pvalue=float("nan"),
        df=None,
        distribution="DW",
        null_hypothesis="No first-order autocorrelation",
        reject=(dw < 1.5 or dw > 2.5),
        details={"interpretation": (
            "positive autocorrelation" if dw < 1.5
            else "negative autocorrelation" if dw > 2.5
            else "no evidence of autocorrelation"
        )},
    )


def breusch_godfrey(result: RegressionResult, *, nlags: int = 1) -> TestResult:
    """Breusch-Godfrey LM test for serial correlation up to order *nlags*.

    H₀: no serial correlation up to lag *nlags*.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    nlags:
        Number of lags to include (default 1).

    Returns
    -------
    TestResult with LM statistic and p-value (Chi²(nlags)).
    """
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey

    if nlags <= 0:
        raise ValueError("nlags must be a positive integer.")
    lm_stat, lm_pval, f_stat, f_pval = acorr_breusch_godfrey(
        result.raw, nlags=nlags
    )
    return TestResult(
        test_name="Breusch-Godfrey",
        statistic=float(lm_stat),
        pvalue=float(lm_pval),
        df=float(nlags),
        distribution="Chi2",
        null_hypothesis=f"No serial correlation up to lag {nlags}",
        reject=float(lm_pval) < 0.05,
        details={"f_stat": float(f_stat), "f_pval": float(f_pval)},
    )


def _validate_n_obs(n_obs: int, lags: int) -> None:
    if n_obs <= lags:
        raise ValueError("n_obs must exceed the number of lags.")
