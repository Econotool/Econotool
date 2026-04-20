"""Finance-style event study with market-model abnormal returns.

This module implements the classic MacKinlay (1997) / Campbell-Lo-MacKinlay
event study methodology: estimate a market model over a pre-event window,
compute abnormal returns (AR), accumulate to cumulative abnormal returns
(CAR), and apply a battery of parametric and non-parametric tests to the
null hypothesis that CAR = 0 over the event window.

Distinct from :mod:`econtools.evaluation.did`, which exports a DID-style
``event_study`` (treatment × relative-time coefficients).  This module is
for the security-price / abnormal-returns use case (e.g., CVS-Aetna merger,
Project Example 1).

Public API
----------
MarketModel                     — frozen dataclass of market-model params
EventStudyCARResult             — frozen dataclass aggregating AR, CAR, tests
estimate_market_model           — OLS of r_i on constant + r_m
compute_abnormal_returns        — r_i − (α̂ + β̂·r_m)
cumulative_abnormal_returns     — sum AR over window
patell_car_test                 — Patell (1976) standardised Z-test
bmp_car_test                    — Boehmer-Musumeci-Poulsen (1991) cross-sectional Z
corrado_rank_test               — Corrado (1989) non-parametric rank test
cowan_sign_test                 — Cowan (1992) generalised sign test
event_study_car                 — top-level wrapper

References
----------
Patell, J.M. (1976) "Corporate forecasts of earnings per share and stock
    price behavior", Journal of Accounting Research 14(2), 246-276.
Brown, S.J. & Warner, J.B. (1985) "Using daily stock returns", JFE 14(1).
Corrado, C.J. (1989) "A nonparametric test for abnormal security-price
    performance in event studies", JFE 23(2), 385-395.
Boehmer, E., Musumeci, J. & Poulsen, A.B. (1991) "Event-study methodology
    under conditions of event-induced variance", JFE 30(2), 253-272.
Cowan, A.R. (1992) "Nonparametric event study tests", Review of
    Quantitative Finance and Accounting 2, 343-358.
MacKinlay, A.C. (1997) "Event studies in economics and finance", JEL 35(1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm

from econtools._core.types import TestResult


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketModel:
    """Fitted market-model parameters for a single firm.

    Attributes
    ----------
    alpha : float
        Intercept (α̂).
    beta : float
        Market-factor loading (β̂).
    sigma : float
        Residual standard deviation over the estimation window.
    n_obs : int
        Number of usable observations in the estimation window.
    r_squared : float
        Goodness of fit over the estimation window.
    """

    alpha: float
    beta: float
    sigma: float
    n_obs: int
    r_squared: float


@dataclass(frozen=True)
class EventStudyCARResult:
    """Aggregated output of a finance-style event study.

    Attributes
    ----------
    abnormal_returns : pd.DataFrame
        Abnormal returns over the event window.  Rows are integer event
        time (relative to the event date); columns are firm identifiers.
    car : pd.Series
        Per-firm cumulative abnormal return over the event window.
    caar : float
        Cross-sectional mean of ``car`` (cumulative average AR).
    tests : dict[str, TestResult]
        Test-battery output keyed by test name.
    market_model : MarketModel | dict[str, MarketModel]
        Estimated market model(s).  Single firm ⇒ ``MarketModel``;
        multiple firms ⇒ mapping from firm identifier to ``MarketModel``.
    event_window : tuple[int, int]
        Inclusive (start, end) event-time offsets.
    estimation_window : tuple[int, int]
        Inclusive (start, end) estimation-time offsets (negative).
    details : dict
        Ancillary information (dropped indices, NaN counts, etc.).
    """

    abnormal_returns: pd.DataFrame
    car: pd.Series
    caar: float
    tests: dict
    market_model: Union["MarketModel", dict]
    event_window: tuple[int, int]
    estimation_window: tuple[int, int]
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def _locate_event_index(index: pd.Index, event_date: pd.Timestamp) -> int:
    """Return the integer position of ``event_date`` within ``index``.

    If the exact date is absent (e.g. weekend / non-trading day), fall back
    to the first subsequent trading day in the index.
    """
    ts = pd.Timestamp(event_date)
    if ts in index:
        return int(index.get_loc(ts))
    # searchsorted gives insertion point; that's the first >= date
    pos = int(index.searchsorted(ts))
    if pos >= len(index):
        raise ValueError(
            f"event_date {ts} is after the last observation in the data"
        )
    return pos


def _slice_window(
    series: pd.Series,
    event_loc: int,
    window: tuple[int, int],
) -> pd.Series:
    """Slice ``series`` over a window defined by trading-day offsets.

    Parameters
    ----------
    series : pd.Series
        Returns series indexed by trading date.
    event_loc : int
        Integer position of event_date in the series index.
    window : (int, int)
        Inclusive (start, end) offsets relative to the event.

    Returns
    -------
    pd.Series with an integer event-time index (0 = event_date).
    """
    start, end = window
    if start > end:
        raise ValueError(f"window start {start} > end {end}")
    lo = event_loc + start
    hi = event_loc + end
    if lo < 0 or hi >= len(series):
        raise ValueError(
            f"window ({start}, {end}) extends beyond available data "
            f"(event_loc={event_loc}, len={len(series)})"
        )
    sub = series.iloc[lo : hi + 1].copy()
    sub.index = pd.RangeIndex(start=start, stop=end + 1, name="event_time")
    return sub


# ---------------------------------------------------------------------------
# Market model estimation
# ---------------------------------------------------------------------------

def estimate_market_model(
    asset_returns: pd.Series,
    market_returns: pd.Series,
) -> MarketModel:
    """Estimate the market model r_i,t = α + β·r_m,t + ε_i,t by OLS.

    Rows with any NaN in ``asset_returns`` or ``market_returns`` are
    dropped before estimation.

    Parameters
    ----------
    asset_returns : pd.Series
        Returns of the focal asset over the estimation window.
    market_returns : pd.Series
        Returns of the market proxy over the same window.

    Returns
    -------
    MarketModel
    """
    aligned = pd.concat(
        [asset_returns.rename("r_i"), market_returns.rename("r_m")],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 3:
        raise ValueError(
            f"Not enough non-missing observations to estimate the market "
            f"model (got {len(aligned)}, need ≥ 3)."
        )

    y = aligned["r_i"].astype(float)
    X = sm.add_constant(aligned["r_m"].astype(float))
    res = sm.OLS(y, X).fit()

    alpha = float(res.params["const"])
    beta = float(res.params["r_m"])
    # Residual std with df correction (matches statsmodels' scale**0.5)
    sigma = float(np.sqrt(res.scale))
    n_obs = int(res.nobs)
    r_squared = float(res.rsquared)

    return MarketModel(
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        n_obs=n_obs,
        r_squared=r_squared,
    )


# ---------------------------------------------------------------------------
# Abnormal returns
# ---------------------------------------------------------------------------

def compute_abnormal_returns(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    model: MarketModel,
) -> pd.Series:
    """Compute abnormal returns AR_t = r_i,t − (α̂ + β̂·r_m,t).

    Parameters
    ----------
    asset_returns : pd.Series
        Returns of the focal asset (any window).
    market_returns : pd.Series
        Returns of the market proxy (same index).
    model : MarketModel
        Previously estimated market-model parameters.

    Returns
    -------
    pd.Series of abnormal returns, indexed like ``asset_returns``.
    """
    expected = model.alpha + model.beta * market_returns.astype(float)
    ar = asset_returns.astype(float) - expected
    return ar


def cumulative_abnormal_returns(
    ar: pd.Series,
    window: tuple[int, int] | None = None,
) -> float:
    """Sum abnormal returns over ``window`` (default: entire series).

    NaN entries are treated as zero so that controlled observations (e.g.
    confounding-event zeroing) don't propagate.  If every value in the
    window is NaN, the result is NaN.

    Parameters
    ----------
    ar : pd.Series
        Abnormal-return series indexed by integer event time (or any
        index if ``window`` is None).
    window : (int, int), optional
        Inclusive event-time range to cumulate over.  If ``None``, the
        full series is summed.

    Returns
    -------
    float
    """
    if window is not None:
        start, end = window
        sub = ar.loc[(ar.index >= start) & (ar.index <= end)]
    else:
        sub = ar
    if len(sub) == 0 or sub.isna().all():
        return float("nan")
    return float(sub.fillna(0.0).sum())


# ---------------------------------------------------------------------------
# Tests — parametric
# ---------------------------------------------------------------------------

def patell_car_test(
    ar: pd.Series,
    model: MarketModel,
    event_window: tuple[int, int],
) -> TestResult:
    """Patell (1976) standardised cumulative abnormal return test.

    Standardises each AR by the estimation-period residual standard
    deviation, sums over the event window, and tests under N(0, 1).

    For long estimation windows this effectively ignores the extra
    sampling-error term in Var(AR_t) and reduces to Z = CAR / (σ·√L),
    where L is the event-window length.

    Parameters
    ----------
    ar : pd.Series
        Abnormal returns over the event window (integer event-time index).
    model : MarketModel
        Fitted market model.
    event_window : (int, int)
        Inclusive event-time range.

    Returns
    -------
    TestResult
    """
    start, end = event_window
    sub = ar.loc[(ar.index >= start) & (ar.index <= end)].fillna(0.0)
    L = int(len(sub))
    if L == 0:
        raise ValueError("Event window is empty.")

    sigma = float(model.sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Market-model sigma must be positive and finite.")

    car = float(sub.sum())
    se_car = sigma * np.sqrt(L)
    z = car / se_car
    pval = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))

    return TestResult(
        test_name="Patell Z-test (CAR)",
        statistic=float(z),
        pvalue=float(pval),
        df=None,
        distribution="N(0,1)",
        null_hypothesis="CAR = 0 over the event window",
        reject=bool(pval < 0.05),
        details={
            "car": car,
            "se_car": float(se_car),
            "window_length": L,
            "estimation_sigma": sigma,
        },
    )


def bmp_car_test(
    ars_by_firm: pd.DataFrame,
    models_by_firm: dict,
    event_window: tuple[int, int],
) -> TestResult:
    """Boehmer-Musumeci-Poulsen (1991) standardised cross-sectional test.

    Robust to event-induced variance.  For each firm i, compute the
    standardised CAR:

        SCAR_i = CAR_i / (σ̂_i · √L)

    then compute the cross-sectional t-statistic:

        t = (1/N) Σ SCAR_i / (s_SCAR / √N)

    where s_SCAR is the sample standard deviation of SCAR across firms.
    Reference distribution is t with N−1 degrees of freedom.

    Parameters
    ----------
    ars_by_firm : pd.DataFrame
        Event-window abnormal returns (rows=event time, cols=firms).
    models_by_firm : dict[str, MarketModel]
        Market models keyed by column name in ``ars_by_firm``.
    event_window : (int, int)
        Inclusive event-time range.

    Returns
    -------
    TestResult
    """
    start, end = event_window
    sub = ars_by_firm.loc[
        (ars_by_firm.index >= start) & (ars_by_firm.index <= end)
    ].fillna(0.0)
    L = int(len(sub))
    if L == 0:
        raise ValueError("Event window is empty.")

    firms = [c for c in ars_by_firm.columns if c in models_by_firm]
    if len(firms) < 2:
        raise ValueError(
            "BMP test requires at least 2 firms; got "
            f"{len(firms)}."
        )

    scars = []
    for f in firms:
        sigma_i = float(models_by_firm[f].sigma)
        if not np.isfinite(sigma_i) or sigma_i <= 0:
            continue
        car_i = float(sub[f].sum())
        scar_i = car_i / (sigma_i * np.sqrt(L))
        scars.append(scar_i)

    scars_arr = np.asarray(scars, dtype=float)
    N = int(len(scars_arr))
    if N < 2:
        raise ValueError("Need ≥ 2 firms with valid sigma for BMP test.")

    mean_scar = float(scars_arr.mean())
    sd_scar = float(scars_arr.std(ddof=1))
    if sd_scar <= 0:
        t_stat = float("inf") if mean_scar != 0 else 0.0
        pval = 0.0 if mean_scar != 0 else 1.0
    else:
        t_stat = mean_scar * np.sqrt(N) / sd_scar
        pval = 2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=N - 1))

    return TestResult(
        test_name="Boehmer-Musumeci-Poulsen standardised cross-sectional",
        statistic=float(t_stat),
        pvalue=float(pval),
        df=float(N - 1),
        distribution="t",
        null_hypothesis="Mean standardised CAR across firms = 0",
        reject=bool(pval < 0.05),
        details={
            "n_firms": N,
            "mean_scar": mean_scar,
            "sd_scar": sd_scar,
            "window_length": L,
        },
    )


# ---------------------------------------------------------------------------
# Tests — non-parametric
# ---------------------------------------------------------------------------

def corrado_rank_test(
    ars_by_firm: pd.DataFrame,
    event_window: tuple[int, int],
) -> TestResult:
    """Corrado (1989) non-parametric rank test.

    For each firm, rank its abnormal returns across the combined
    estimation + event window (ranks 1..T_i with average ties).  Compare
    the mean event-window rank to its null expectation ((T_i + 1)/2).

    Expects ``ars_by_firm`` to contain *all* AR observations for each firm
    (estimation + event), indexed so that ``event_window`` selects the
    event rows by label.

    Parameters
    ----------
    ars_by_firm : pd.DataFrame
        Abnormal returns for every observation in estimation + event
        windows.  Rows indexed by integer event time; cols by firm.
    event_window : (int, int)
        Inclusive event-time range.

    Returns
    -------
    TestResult
    """
    start, end = event_window
    # Ranks per column, ignoring NaN (NaN -> NaN rank)
    ranks = ars_by_firm.rank(axis=0, method="average", na_option="keep")
    event_mask = (ars_by_firm.index >= start) & (ars_by_firm.index <= end)
    L = int(event_mask.sum())
    if L == 0:
        raise ValueError("Event window is empty.")

    firms = ars_by_firm.columns.tolist()
    N = len(firms)
    if N < 1:
        raise ValueError("Need ≥ 1 firm for rank test.")

    # Per-firm: mean event-window rank minus (T_i+1)/2, divided by SD of ranks
    firm_stats = []
    for f in firms:
        col_ranks = ranks[f].dropna()
        T_i = len(col_ranks)
        if T_i < 2:
            continue
        mean_rank_event = float(col_ranks.loc[
            (col_ranks.index >= start) & (col_ranks.index <= end)
        ].mean())
        expected = (T_i + 1.0) / 2.0
        # Under the null, variance of a single rank is (T^2 - 1)/12
        var_rank = (T_i * T_i - 1.0) / 12.0
        sd_mean_rank = np.sqrt(var_rank / L)
        if sd_mean_rank <= 0:
            continue
        z_i = (mean_rank_event - expected) / sd_mean_rank
        firm_stats.append(z_i)

    z_arr = np.asarray(firm_stats, dtype=float)
    if len(z_arr) == 0:
        raise ValueError("No usable firms for rank test.")

    # Cross-sectional mean Z, referred to N(0,1) via central limit
    z_stat = float(z_arr.mean() * np.sqrt(len(z_arr)))
    pval = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z_stat)))

    return TestResult(
        test_name="Corrado rank test",
        statistic=z_stat,
        pvalue=float(pval),
        df=None,
        distribution="N(0,1)",
        null_hypothesis="Median event-window rank equals its null expectation",
        reject=bool(pval < 0.05),
        details={
            "n_firms": int(len(z_arr)),
            "window_length": L,
        },
    )


def cowan_sign_test(cars_by_firm: pd.Series) -> TestResult:
    """Cowan (1992) generalised sign test.

    Compares the fraction of firms with positive CAR to its null
    expectation of 0.5 (equivalent to a binomial sign test with p=0.5).
    Uses a normal approximation with continuity correction.

    Parameters
    ----------
    cars_by_firm : pd.Series
        Per-firm cumulative abnormal returns.

    Returns
    -------
    TestResult
    """
    vals = cars_by_firm.dropna().to_numpy(dtype=float)
    N = int(len(vals))
    if N < 1:
        raise ValueError("Need ≥ 1 firm for sign test.")

    n_pos = int((vals > 0).sum())
    p_hat = n_pos / N
    expected = 0.5

    # Normal approximation to the binomial(N, 0.5)
    se = np.sqrt(0.5 * 0.5 / N)
    z_stat = (p_hat - expected) / se
    pval = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z_stat)))

    return TestResult(
        test_name="Cowan generalised sign test",
        statistic=float(z_stat),
        pvalue=float(pval),
        df=None,
        distribution="N(0,1)",
        null_hypothesis="P(CAR > 0) = 0.5 across firms",
        reject=bool(pval < 0.05),
        details={
            "n_firms": N,
            "n_positive": n_pos,
            "fraction_positive": float(p_hat),
        },
    )


# ---------------------------------------------------------------------------
# Top-level convenience wrapper
# ---------------------------------------------------------------------------

def _coerce_returns_frame(returns: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Return ``returns`` as a DataFrame, preserving its index."""
    if isinstance(returns, pd.Series):
        name = returns.name if returns.name is not None else "firm"
        return returns.to_frame(name=name)
    if isinstance(returns, pd.DataFrame):
        return returns
    raise TypeError(
        f"`returns` must be a pd.Series or pd.DataFrame, got {type(returns)}."
    )


def event_study_car(
    returns: Union[pd.Series, pd.DataFrame],
    market: pd.Series,
    event_date: pd.Timestamp,
    estimation_window: tuple[int, int] = (-250, -11),
    event_window: tuple[int, int] = (-1, 1),
) -> EventStudyCARResult:
    """Finance-style event study: market model + CAR + test battery.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Firm return series (or wide DataFrame of several firms).  Indexed
        by trading date.
    market : pd.Series
        Market-proxy returns on the same trading-date index.
    event_date : pd.Timestamp
        Day 0 of the event.  If not a trading day in the index, the first
        subsequent trading day is used.
    estimation_window : (int, int)
        Inclusive trading-day offsets defining the estimation window.
        Defaults to (-250, -11) — about one trading year, ending 11 days
        before the event.
    event_window : (int, int)
        Inclusive trading-day offsets defining the event window.
        Defaults to (-1, 1).

    Returns
    -------
    EventStudyCARResult
    """
    frame = _coerce_returns_frame(returns)
    firms = list(frame.columns)
    if len(firms) == 0:
        raise ValueError("`returns` must contain at least one firm column.")

    # Align indices
    if not frame.index.equals(market.index):
        joined = frame.join(market.rename("__market__"), how="inner")
        frame = joined[firms]
        market_aligned = joined["__market__"]
    else:
        market_aligned = market

    event_loc = _locate_event_index(frame.index, event_date)

    # --- Estimation ---
    models: dict[str, MarketModel] = {}
    nan_report: dict[str, int] = {}
    for f in firms:
        r_i_est = _slice_window(frame[f], event_loc, estimation_window)
        r_m_est = _slice_window(market_aligned, event_loc, estimation_window)
        nan_report[f] = int(r_i_est.isna().sum() + r_m_est.isna().sum())
        models[f] = estimate_market_model(r_i_est, r_m_est)

    # --- AR over a combined (estimation + event) window for rank test ---
    combined_start = min(estimation_window[0], event_window[0])
    combined_end = max(estimation_window[1], event_window[1])

    ar_combined = {}
    ar_event = {}
    for f in firms:
        r_i_all = _slice_window(
            frame[f], event_loc, (combined_start, combined_end)
        )
        r_m_all = _slice_window(
            market_aligned, event_loc, (combined_start, combined_end)
        )
        ar_all = compute_abnormal_returns(r_i_all, r_m_all, models[f])
        ar_combined[f] = ar_all
        mask = (ar_all.index >= event_window[0]) & (ar_all.index <= event_window[1])
        ar_event[f] = ar_all.loc[mask]

    ar_event_df = pd.DataFrame(ar_event)
    ar_combined_df = pd.DataFrame(ar_combined)

    # --- CAR / CAAR ---
    car = ar_event_df.fillna(0.0).sum(axis=0)
    caar = float(car.mean()) if len(car) > 0 else float("nan")

    # --- Tests ---
    tests: dict[str, TestResult] = {}
    # Per-firm Patell (use first firm for single-firm case, all for multi)
    for f in firms:
        key = f"patell_{f}" if len(firms) > 1 else "patell"
        tests[key] = patell_car_test(
            ar_event_df[f], models[f], event_window
        )
    # Cross-sectional tests require ≥ 2 firms
    if len(firms) >= 2:
        try:
            tests["bmp"] = bmp_car_test(ar_event_df, models, event_window)
        except ValueError:
            pass
        try:
            tests["rank"] = corrado_rank_test(ar_combined_df, event_window)
        except ValueError:
            pass
        tests["sign"] = cowan_sign_test(car)

    market_model_out: Union[MarketModel, dict]
    if len(firms) == 1:
        market_model_out = models[firms[0]]
    else:
        market_model_out = models

    return EventStudyCARResult(
        abnormal_returns=ar_event_df,
        car=car,
        caar=caar,
        tests=tests,
        market_model=market_model_out,
        event_window=tuple(event_window),
        estimation_window=tuple(estimation_window),
        details={
            "firms": firms,
            "event_date": pd.Timestamp(event_date),
            "event_index": event_loc,
            "nan_in_estimation": nan_report,
        },
    )


__all__ = [
    "MarketModel",
    "EventStudyCARResult",
    "estimate_market_model",
    "compute_abnormal_returns",
    "cumulative_abnormal_returns",
    "patell_car_test",
    "bmp_car_test",
    "corrado_rank_test",
    "cowan_sign_test",
    "event_study_car",
]
