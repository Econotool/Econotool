"""Difference-in-Differences estimation and event study analysis.

Provides canonical 2×2 DID, event-study with leads and lags,
parallel trends pre-tests, and staggered treatment support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm

from econtools._core.types import TestResult


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DIDResult:
    """Result of a difference-in-differences estimation.

    Attributes
    ----------
    att : float
        Average treatment effect on the treated.
    se : float
        Standard error of the ATT.
    tstat : float
        t-statistic.
    pvalue : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    nobs : int
        Number of observations.
    params : pd.Series
        Full coefficient vector.
    details : dict
        Ancillary info (group means, etc.).
    """

    att: float
    se: float
    tstat: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    nobs: int
    params: pd.Series
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EventStudyResult:
    """Result of an event study regression.

    Attributes
    ----------
    coefficients : pd.Series
        Lead/lag coefficients (index = relative time).
    std_errors : pd.Series
        Standard errors.
    ci_lower : pd.Series
        Lower 95% CI.
    ci_upper : pd.Series
        Upper 95% CI.
    pvalues : pd.Series
        Two-sided p-values.
    reference_period : int
        Omitted (normalised-to-zero) period.
    pre_trend_f : float
        Joint F-statistic for pre-treatment coefficients = 0.
    pre_trend_pvalue : float
        p-value for the pre-trend F-test.
    nobs : int
        Number of observations.
    params : pd.Series
        Full coefficient vector (including controls, FE dummies).
    """

    coefficients: pd.Series
    std_errors: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series
    pvalues: pd.Series
    reference_period: int
    pre_trend_f: float
    pre_trend_pvalue: float
    nobs: int
    params: pd.Series


# ---------------------------------------------------------------------------
# Canonical 2×2 DID
# ---------------------------------------------------------------------------

def did_estimate(
    df: pd.DataFrame,
    y: str,
    treat: str,
    post: str,
    controls: Optional[list[str]] = None,
    cluster: Optional[str] = None,
) -> DIDResult:
    """Canonical 2×2 difference-in-differences estimator.

    Estimates: Y = α + β₁·treat + β₂·post + δ·(treat×post) + X'γ + ε

    The ATT is δ̂, the coefficient on the interaction term.

    Parameters
    ----------
    df : DataFrame
        Data with columns for outcome, treatment, post indicator, and
        optionally controls and cluster variable.
    y : str
        Outcome variable name.
    treat : str
        Binary treatment group indicator (1 = treated, 0 = control).
    post : str
        Binary post-period indicator (1 = after treatment, 0 = before).
    controls : list of str, optional
        Additional control variables.
    cluster : str, optional
        Variable name for cluster-robust standard errors.

    Returns
    -------
    DIDResult
    """
    data = df.copy()
    interaction_col = f"{treat}_x_{post}"
    data[interaction_col] = data[treat] * data[post]

    rhs_vars = [treat, post, interaction_col]
    if controls:
        rhs_vars.extend(controls)

    X = sm.add_constant(data[rhs_vars].astype(float))
    Y = data[y].astype(float)

    # Fit with appropriate SEs
    if cluster is not None:
        groups = data[cluster]
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        res = sm.OLS(Y, X).fit(cov_type="HC1")

    att = float(res.params[interaction_col])
    se = float(res.bse[interaction_col])
    tstat = float(res.tvalues[interaction_col])
    pval = float(res.pvalues[interaction_col])
    ci = res.conf_int().loc[interaction_col]

    # Compute group means for the 2×2 table
    means = {}
    for t_val in [0, 1]:
        for p_val in [0, 1]:
            mask = (data[treat] == t_val) & (data[post] == p_val)
            label = f"{'treat' if t_val else 'ctrl'}_{'post' if p_val else 'pre'}"
            means[label] = float(data.loc[mask, y].mean()) if mask.any() else float("nan")

    # Simple DID from means (for comparison)
    simple_did = (
        (means.get("treat_post", 0) - means.get("treat_pre", 0))
        - (means.get("ctrl_post", 0) - means.get("ctrl_pre", 0))
    )

    return DIDResult(
        att=att,
        se=se,
        tstat=tstat,
        pvalue=pval,
        ci_lower=float(ci.iloc[0]),
        ci_upper=float(ci.iloc[1]),
        nobs=int(res.nobs),
        params=res.params,
        details={
            "group_means": means,
            "simple_did": simple_did,
            "r_squared": float(res.rsquared),
            "cov_type": res.cov_type,
        },
    )


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------

def event_study(
    df: pd.DataFrame,
    y: str,
    entity: str,
    time: str,
    treat_time: str,
    controls: Optional[list[str]] = None,
    ref_period: int = -1,
    n_leads: Optional[int] = None,
    n_lags: Optional[int] = None,
    cluster: Optional[str] = None,
    entity_fe: bool = True,
    time_fe: bool = True,
) -> EventStudyResult:
    """Event study regression with leads and lags.

    Estimates relative-time dummies interacted with treatment, absorbing
    entity and time fixed effects.

    Parameters
    ----------
    df : DataFrame
    y : str
        Outcome variable.
    entity : str
        Entity (unit) identifier column.
    time : str
        Time period column (integer).
    treat_time : str
        Column with the treatment adoption time for each unit.
        Set to ``np.inf`` or ``NaN`` for never-treated units.
    controls : list of str, optional
        Additional control variables.
    ref_period : int
        Reference (omitted) relative time period (default -1).
    n_leads : int, optional
        Number of pre-treatment leads to include.  If None, uses all
        available pre-periods.
    n_lags : int, optional
        Number of post-treatment lags.  If None, uses all available.
    cluster : str, optional
        Cluster variable for SEs (defaults to ``entity`` if not given).
    entity_fe : bool
        Include entity fixed effects (default True).
    time_fe : bool
        Include time fixed effects (default True).

    Returns
    -------
    EventStudyResult
    """
    data = df.copy()

    # Compute relative time
    data["_rel_time"] = data[time] - data[treat_time]
    # Never-treated units: set relative time to a large sentinel
    never_treated = data[treat_time].isna() | np.isinf(data[treat_time])
    data.loc[never_treated, "_rel_time"] = np.nan

    # Determine lead/lag range
    valid_rt = data["_rel_time"].dropna().astype(int)
    if valid_rt.empty:
        raise ValueError("No treated units found (all treat_time is NaN/inf).")

    rt_min = int(valid_rt.min()) if n_leads is None else -abs(n_leads)
    rt_max = int(valid_rt.max()) if n_lags is None else abs(n_lags)

    # Create relative-time dummies (excluding reference period)
    rel_times = sorted(set(range(rt_min, rt_max + 1)) - {ref_period})
    for rt in rel_times:
        col = f"_rt_{rt}" if rt < 0 else f"_rt_p{rt}"
        data[col] = ((data["_rel_time"] == rt)).astype(float)
        # Never-treated: all dummies = 0 (they serve as controls)
        data.loc[never_treated, col] = 0.0

    rt_cols = [f"_rt_{rt}" if rt < 0 else f"_rt_p{rt}" for rt in rel_times]

    # Build design matrix
    rhs = list(rt_cols)
    if controls:
        rhs.extend(controls)

    # Fixed effects via dummies (for moderate panel sizes)
    if entity_fe:
        ent_dummies = pd.get_dummies(data[entity], prefix="_ent", drop_first=True, dtype=float)
        data = pd.concat([data, ent_dummies], axis=1)
        rhs.extend(ent_dummies.columns.tolist())

    if time_fe:
        time_dummies = pd.get_dummies(data[time], prefix="_time", drop_first=True, dtype=float)
        data = pd.concat([data, time_dummies], axis=1)
        rhs.extend(time_dummies.columns.tolist())

    X = sm.add_constant(data[rhs].astype(float))
    Y = data[y].astype(float)

    # Cluster SEs
    cl = cluster if cluster is not None else entity
    groups = data[cl]
    try:
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:
        res = sm.OLS(Y, X).fit(cov_type="HC1")

    # Extract event-study coefficients
    coefs = {}
    ses = {}
    pvs = {}
    for rt, col in zip(rel_times, rt_cols):
        coefs[rt] = float(res.params.get(col, np.nan))
        ses[rt] = float(res.bse.get(col, np.nan))
        pvs[rt] = float(res.pvalues.get(col, np.nan))

    # Add reference period = 0
    coefs[ref_period] = 0.0
    ses[ref_period] = 0.0
    pvs[ref_period] = 1.0

    coef_s = pd.Series(coefs).sort_index()
    se_s = pd.Series(ses).sort_index()
    pv_s = pd.Series(pvs).sort_index()
    ci_lo = coef_s - 1.96 * se_s
    ci_hi = coef_s + 1.96 * se_s

    # Pre-trend F-test: joint test that all pre-treatment coefficients = 0
    pre_cols = [col for rt, col in zip(rel_times, rt_cols) if rt < 0]
    if pre_cols:
        try:
            r_matrix = np.zeros((len(pre_cols), len(res.params)))
            for i, pc in enumerate(pre_cols):
                idx = list(res.params.index).index(pc)
                r_matrix[i, idx] = 1.0
            f_result = res.f_test(r_matrix)
            pre_f = float(f_result.fvalue)
            pre_pval = float(f_result.pvalue)
        except Exception:
            pre_f = float("nan")
            pre_pval = float("nan")
    else:
        pre_f = float("nan")
        pre_pval = float("nan")

    return EventStudyResult(
        coefficients=coef_s,
        std_errors=se_s,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        pvalues=pv_s,
        reference_period=ref_period,
        pre_trend_f=pre_f,
        pre_trend_pvalue=pre_pval,
        nobs=int(res.nobs),
        params=res.params,
    )


# ---------------------------------------------------------------------------
# Parallel trends test (standalone)
# ---------------------------------------------------------------------------

def parallel_trends_test(
    df: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    entity: Optional[str] = None,
) -> TestResult:
    """Test for parallel pre-trends between treatment and control groups.

    Estimates group-specific time trends in the pre-period and tests
    whether the treatment group trend differs from the control group.

    Parameters
    ----------
    df : DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment group indicator.
    time : str
        Time period variable (numeric).
    entity : str, optional
        Entity identifier for clustering.

    Returns
    -------
    TestResult
    """
    data = df.copy()
    data["_time_x_treat"] = data[time].astype(float) * data[treat].astype(float)

    X_vars = [treat, time, "_time_x_treat"]
    X = sm.add_constant(data[X_vars].astype(float))
    Y = data[y].astype(float)

    if entity is not None:
        groups = data[entity]
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        res = sm.OLS(Y, X).fit(cov_type="HC1")

    stat = float(res.tvalues["_time_x_treat"])
    pval = float(res.pvalues["_time_x_treat"])

    return TestResult(
        test_name="Parallel Trends (differential trend)",
        statistic=stat,
        pvalue=pval,
        df=float(res.df_resid),
        distribution="t",
        null_hypothesis="Treatment and control groups have equal time trends",
        reject=bool(pval < 0.05),
        details={
            "differential_trend_coef": float(res.params["_time_x_treat"]),
            "differential_trend_se": float(res.bse["_time_x_treat"]),
        },
    )
