"""Detrending utilities for time-series data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.nonparametric.smoothers_lowess import lowess


@dataclass(frozen=True)
class TrendResult:
    method: str
    trend: pd.Series
    resid: pd.Series
    params: dict[str, float] | None
    aic: float | None
    bic: float | None
    rmse: float
    n_obs: int


def fit_trend(
    y: Iterable[float] | pd.Series,
    x: Iterable[float] | pd.Series | None = None,
    *,
    method: str = "linear",
    hp_lambda: float = 1600.0,
    ma_window: int = 5,
    lowess_frac: float = 0.3,
) -> TrendResult:
    """Fit a deterministic trend and return fitted values + residuals."""
    y_ser = _to_series(y, name="y")
    x_ser = _to_series(x, name="x") if x is not None else pd.Series(
        np.arange(len(y_ser)), index=y_ser.index, name="x"
    )

    method_l = method.lower()
    if method_l in {"linear", "quadratic", "cubic", "log-linear", "log-quadratic"}:
        return _fit_parametric(y_ser, x_ser, method_l)
    if method_l == "hp":
        return _fit_hp(y_ser, hp_lambda)
    if method_l == "moving_average":
        return _fit_moving_average(y_ser, ma_window)
    if method_l == "lowess":
        return _fit_lowess(y_ser, x_ser, lowess_frac)
    raise ValueError(
        "Unknown method. Choose from "
        "'linear', 'quadratic', 'cubic', 'log-linear', 'log-quadratic', "
        "'hp', 'moving_average', 'lowess'."
    )


def assess_trend_options(
    y: Iterable[float] | pd.Series,
    x: Iterable[float] | pd.Series | None = None,
    *,
    methods: list[str],
    hp_lambda: float = 1600.0,
    ma_window: int = 5,
    lowess_frac: float = 0.3,
) -> list[TrendResult]:
    """Fit multiple trend options and return a list of TrendResult."""
    results: list[TrendResult] = []
    for method in methods:
        res = fit_trend(
            y,
            x,
            method=method,
            hp_lambda=hp_lambda,
            ma_window=ma_window,
            lowess_frac=lowess_frac,
        )
        results.append(res)
    return results


def fit_polynomial_trend(
    y: Iterable[float] | pd.Series,
    x: Iterable[float] | pd.Series | None = None,
    *,
    degree: int = 1,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.Series, pd.Series]:
    """Fit a polynomial time trend of given degree.

    Returns (fit_result, trend, resid).
    """
    if degree < 1:
        raise ValueError("degree must be >= 1.")
    y_ser = _to_series(y, name="y")
    x_ser = _to_series(x, name="x") if x is not None else pd.Series(
        np.arange(len(y_ser)), index=y_ser.index, name="x"
    )

    data = {"x": x_ser}
    for d in range(2, degree + 1):
        data[f"x{d}"] = x_ser**d
    exog = sm.add_constant(pd.DataFrame(data), has_constant="add")
    fit = sm.OLS(y_ser, exog).fit()
    trend = fit.fittedvalues
    resid = y_ser - trend
    return fit, pd.Series(trend, index=y_ser.index, name="trend"), pd.Series(
        resid, index=y_ser.index, name="resid"
    )


def predict_trend(
    fit: sm.regression.linear_model.RegressionResultsWrapper,
    x_new: Iterable[float],
    *,
    degree: int,
    alpha: float = 0.05,
    prediction: str = "mean",
) -> pd.DataFrame:
    """Predict trend at new x values with confidence intervals.

    prediction: "mean" (default) or "obs".
    """
    if degree < 1:
        raise ValueError("degree must be >= 1.")
    if prediction not in {"mean", "obs"}:
        raise ValueError("prediction must be 'mean' or 'obs'.")
    x_vals = np.asarray(list(x_new), dtype=float)
    data = {"x": x_vals}
    for d in range(2, degree + 1):
        data[f"x{d}"] = x_vals**d
    exog = sm.add_constant(pd.DataFrame(data), has_constant="add")
    pred = fit.get_prediction(exog)
    frame = pred.summary_frame(alpha=alpha)
    if prediction == "mean":
        return frame[["mean", "mean_ci_lower", "mean_ci_upper"]].copy()
    return frame[["mean", "obs_ci_lower", "obs_ci_upper"]].copy()


def _fit_parametric(y: pd.Series, x: pd.Series, method: str) -> TrendResult:
    if method == "linear":
        exog = sm.add_constant(x)
    elif method == "quadratic":
        exog = sm.add_constant(pd.DataFrame({"x": x, "x2": x**2}))
    elif method == "cubic":
        exog = sm.add_constant(pd.DataFrame({"x": x, "x2": x**2, "x3": x**3}))
    elif method == "log-linear":
        if (y <= 0).any():
            raise ValueError("log-linear trend requires positive y.")
        exog = sm.add_constant(x)
        y_model = np.log(y)
    elif method == "log-quadratic":
        if (y <= 0).any():
            raise ValueError("log-quadratic trend requires positive y.")
        exog = sm.add_constant(pd.DataFrame({"x": x, "x2": x**2}))
        y_model = np.log(y)
    else:
        raise ValueError("Unknown parametric method.")

    if method in {"log-linear", "log-quadratic"}:
        fit = sm.OLS(y_model, exog).fit()
        log_trend = fit.fittedvalues
        trend = np.exp(log_trend)
        resid = y - trend
    else:
        fit = sm.OLS(y, exog).fit()
        trend = fit.fittedvalues
        resid = y - trend

    rmse = float(np.sqrt(np.mean(resid**2)))
    params = {str(k): float(v) for k, v in fit.params.items()}
    return TrendResult(
        method=method,
        trend=pd.Series(trend, index=y.index, name="trend"),
        resid=pd.Series(resid, index=y.index, name="resid"),
        params=params,
        aic=float(fit.aic),
        bic=float(fit.bic),
        rmse=rmse,
        n_obs=int(fit.nobs),
    )


def _fit_hp(y: pd.Series, hp_lambda: float) -> TrendResult:
    cycle, trend = hpfilter(y, lamb=hp_lambda)
    resid = cycle
    rmse = float(np.sqrt(np.mean(resid**2)))
    return TrendResult(
        method="hp",
        trend=pd.Series(trend, index=y.index, name="trend"),
        resid=pd.Series(resid, index=y.index, name="resid"),
        params={"lambda": float(hp_lambda)},
        aic=None,
        bic=None,
        rmse=rmse,
        n_obs=int(len(y)),
    )


def _fit_moving_average(y: pd.Series, ma_window: int) -> TrendResult:
    if ma_window <= 1:
        raise ValueError("ma_window must be >= 2.")
    trend = y.rolling(ma_window, center=True).mean()
    resid = y - trend
    rmse = float(np.sqrt(np.nanmean(resid**2)))
    return TrendResult(
        method="moving_average",
        trend=pd.Series(trend, index=y.index, name="trend"),
        resid=pd.Series(resid, index=y.index, name="resid"),
        params={"window": float(ma_window)},
        aic=None,
        bic=None,
        rmse=rmse,
        n_obs=int(len(y)),
    )


def _fit_lowess(y: pd.Series, x: pd.Series, frac: float) -> TrendResult:
    if not (0.0 < frac <= 1.0):
        raise ValueError("lowess_frac must be in (0, 1].")
    fitted = lowess(y.values, x.values, frac=frac, return_sorted=False)
    trend = pd.Series(fitted, index=y.index, name="trend")
    resid = y - trend
    rmse = float(np.sqrt(np.mean(resid**2)))
    return TrendResult(
        method="lowess",
        trend=trend,
        resid=pd.Series(resid, index=y.index, name="resid"),
        params={"frac": float(frac)},
        aic=None,
        bic=None,
        rmse=rmse,
        n_obs=int(len(y)),
    )


def _to_series(values: Iterable[float] | pd.Series | None, *, name: str) -> pd.Series:
    if values is None:
        raise ValueError(f"{name} cannot be None.")
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(list(values), dtype=float, name=name)
