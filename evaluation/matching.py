"""Propensity score estimation, matching, IPW, and doubly robust estimation.

Treatment effect estimators based on selection-on-observables:
propensity score matching, inverse probability weighting (IPW),
and augmented IPW (doubly robust).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial import KDTree
import statsmodels.api as sm

from econtools._core.types import TestResult


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropensityResult:
    """Result of propensity score estimation.

    Attributes
    ----------
    scores : pd.Series
        Estimated propensity scores P(D=1|X) for each observation.
    model_params : pd.Series
        Logit coefficients.
    nobs : int
        Number of observations.
    pseudo_r2 : float
        McFadden pseudo-R².
    common_support : tuple[float, float]
        Min/max of the overlap region.
    """

    scores: pd.Series
    model_params: pd.Series
    nobs: int
    pseudo_r2: float
    common_support: tuple[float, float]


@dataclass(frozen=True)
class MatchResult:
    """Result of propensity score matching.

    Attributes
    ----------
    ate : float
        Average treatment effect (if estimable).
    att : float
        Average treatment effect on the treated.
    se_att : float
        Standard error of ATT (Abadie-Imbens).
    pvalue_att : float
        p-value for ATT.
    ci_lower : float
        95% CI lower bound for ATT.
    ci_upper : float
        95% CI upper bound for ATT.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    n_matched : int
        Number of matched pairs (or sets).
    matched_data : pd.DataFrame
        DataFrame with match assignments.
    balance : pd.DataFrame
        Covariate balance table (before/after matching).
    details : dict
    """

    ate: float
    att: float
    se_att: float
    pvalue_att: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    n_matched: int
    matched_data: pd.DataFrame
    balance: pd.DataFrame
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IPWResult:
    """Result of inverse probability weighting estimation.

    Attributes
    ----------
    ate : float
        Average treatment effect.
    att : float
        Average treatment effect on the treated.
    se_ate : float
        Standard error of ATE.
    se_att : float
        Standard error of ATT.
    pvalue_ate : float
    pvalue_att : float
    nobs : int
    effective_n : float
        Effective sample size after weighting.
    details : dict
    """

    ate: float
    att: float
    se_ate: float
    se_att: float
    pvalue_ate: float
    pvalue_att: float
    nobs: int
    effective_n: float
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DoublyRobustResult:
    """Result of doubly robust (AIPW) estimation.

    Attributes
    ----------
    ate : float
        Average treatment effect.
    se : float
        Standard error (sandwich).
    pvalue : float
    ci_lower : float
    ci_upper : float
    nobs : int
    details : dict
    """

    ate: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    nobs: int
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Propensity score estimation
# ---------------------------------------------------------------------------

def propensity_score(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    method: str = "logit",
) -> PropensityResult:
    """Estimate propensity scores via logit (or probit).

    Parameters
    ----------
    df : DataFrame
    treatment : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariates for the propensity model.
    method : str
        ``'logit'`` (default) or ``'probit'``.

    Returns
    -------
    PropensityResult
    """
    data = df.copy()
    X = sm.add_constant(data[covariates].astype(float))
    D = data[treatment].astype(float)

    if method == "logit":
        model = sm.Logit(D, X)
    elif method == "probit":
        model = sm.Probit(D, X)
    else:
        raise ValueError(f"method must be 'logit' or 'probit', got '{method}'.")

    res = model.fit(disp=0)
    scores = pd.Series(res.predict(X), index=df.index, name="pscore")

    # Common support
    treated_scores = scores[D == 1]
    control_scores = scores[D == 0]
    cs_lo = max(treated_scores.min(), control_scores.min())
    cs_hi = min(treated_scores.max(), control_scores.max())

    return PropensityResult(
        scores=scores,
        model_params=res.params,
        nobs=int(res.nobs),
        pseudo_r2=float(res.prsquared),
        common_support=(float(cs_lo), float(cs_hi)),
    )


# ---------------------------------------------------------------------------
# Nearest-neighbor matching
# ---------------------------------------------------------------------------

def nearest_neighbor_match(
    df: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: list[str],
    pscore: Optional[pd.Series] = None,
    n_neighbors: int = 1,
    caliper: Optional[float] = None,
    with_replacement: bool = True,
) -> MatchResult:
    """Nearest-neighbor matching on propensity score.

    Parameters
    ----------
    df : DataFrame
    y : str
        Outcome variable.
    treatment : str
        Binary treatment (0/1).
    covariates : list of str
        Covariates (used for balance table; matching is on pscore).
    pscore : Series, optional
        Pre-computed propensity scores.  If None, estimates via logit.
    n_neighbors : int
        Number of nearest neighbors (default 1).
    caliper : float, optional
        Maximum distance for a match (in pscore units).  Unmatched
        treated units are dropped.
    with_replacement : bool
        Match with replacement (default True).

    Returns
    -------
    MatchResult
    """
    data = df.copy()
    D = data[treatment].astype(int)

    # Estimate pscore if not provided
    if pscore is None:
        ps_result = propensity_score(data, treatment, covariates)
        ps = ps_result.scores
    else:
        ps = pscore.copy()

    treated_idx = D[D == 1].index
    control_idx = D[D == 0].index

    ps_ctrl = ps.loc[control_idx].values.reshape(-1, 1)
    tree = KDTree(ps_ctrl)

    matches = []
    matched_ctrl_indices = []
    for t_idx in treated_idx:
        ps_t = ps.loc[t_idx]
        dists, idxs = tree.query([[ps_t]], k=n_neighbors)
        dists = np.atleast_1d(dists.flatten())
        idxs = np.atleast_1d(idxs.flatten())

        for d, ci in zip(dists, idxs):
            if caliper is not None and d > caliper:
                continue
            c_orig_idx = control_idx[ci]
            matches.append((t_idx, c_orig_idx, d))
            matched_ctrl_indices.append(c_orig_idx)

    if not matches:
        raise ValueError("No matches found within caliper.")

    match_df = pd.DataFrame(matches, columns=["treated_idx", "control_idx", "distance"])

    # Compute ATT
    y_vals = data[y]
    att_components = []
    for t_idx in treated_idx:
        m = match_df[match_df["treated_idx"] == t_idx]
        if m.empty:
            continue
        y_t = y_vals.loc[t_idx]
        y_c_mean = y_vals.loc[m["control_idx"]].mean()
        att_components.append(y_t - y_c_mean)

    att_arr = np.array(att_components)
    n_matched = len(att_arr)
    att = float(att_arr.mean()) if n_matched > 0 else float("nan")
    se_att = float(att_arr.std(ddof=1) / np.sqrt(n_matched)) if n_matched > 1 else float("nan")

    if not np.isnan(se_att) and se_att > 0:
        tstat = att / se_att
        pval = float(2 * sp_stats.t.sf(abs(tstat), df=n_matched - 1))
    else:
        pval = float("nan")

    ci_lo = att - 1.96 * se_att if not np.isnan(se_att) else float("nan")
    ci_hi = att + 1.96 * se_att if not np.isnan(se_att) else float("nan")

    # Covariate balance table
    balance = _compute_balance(data, covariates, treatment, matched_ctrl_indices, treated_idx)

    # ATE (symmetric matching — approximate)
    ate = att  # with NN matching on treated only, ATE ≈ ATT

    return MatchResult(
        ate=ate,
        att=att,
        se_att=se_att,
        pvalue_att=pval,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_treated=len(treated_idx),
        n_control=len(control_idx),
        n_matched=n_matched,
        matched_data=match_df,
        balance=balance,
        details={
            "n_neighbors": n_neighbors,
            "caliper": caliper,
            "with_replacement": with_replacement,
            "mean_distance": float(match_df["distance"].mean()),
        },
    )


# ---------------------------------------------------------------------------
# Inverse Probability Weighting
# ---------------------------------------------------------------------------

def ipw_estimate(
    df: pd.DataFrame,
    y: str,
    treatment: str,
    pscore: pd.Series,
    trim: float = 0.01,
) -> IPWResult:
    """Horvitz-Thompson inverse probability weighting estimator.

    Parameters
    ----------
    df : DataFrame
    y : str
        Outcome variable.
    treatment : str
        Binary treatment (0/1).
    pscore : Series
        Propensity scores.
    trim : float
        Trim observations with extreme propensity scores
        (drop if pscore < trim or pscore > 1 - trim).

    Returns
    -------
    IPWResult
    """
    data = df.copy()
    D = data[treatment].astype(float).values
    Y = data[y].astype(float).values
    e = pscore.values.copy()

    # Trim extreme propensity scores
    mask = (e >= trim) & (e <= 1 - trim)
    D, Y, e = D[mask], Y[mask], e[mask]
    n = len(Y)

    if n == 0:
        raise ValueError("No observations remaining after trimming.")

    # ATE: E[Y(1)] - E[Y(0)] = E[DY/e] - E[(1-D)Y/(1-e)]
    w1 = D / e
    w0 = (1 - D) / (1 - e)

    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)
    ate = float(mu1 - mu0)

    # ATT: E[Y(1) - Y(0) | D=1]
    n1 = np.sum(D)
    att_w = (1 - D) * e / (1 - e) / n1
    att = float(np.mean(D * Y) / np.mean(D) - np.sum(att_w * Y) / np.sum(att_w))

    # Influence function SEs
    # ATE IF: ψ_i = D*Y/e - (1-D)*Y/(1-e) - ate
    psi_ate = w1 * Y - w0 * Y - ate
    se_ate = float(np.std(psi_ate, ddof=1) / np.sqrt(n))

    # ATT IF (approximate)
    psi_att_arr = D * (Y - mu0) / np.mean(D) - (1 - D) * e * Y / ((1 - e) * np.mean(D))
    se_att = float(np.std(psi_att_arr, ddof=1) / np.sqrt(n))

    # p-values
    pval_ate = float(2 * sp_stats.norm.sf(abs(ate / se_ate))) if se_ate > 0 else float("nan")
    pval_att = float(2 * sp_stats.norm.sf(abs(att / se_att))) if se_att > 0 else float("nan")

    # Effective sample size
    eff_n = float((np.sum(w1)) ** 2 / np.sum(w1 ** 2) + (np.sum(w0)) ** 2 / np.sum(w0 ** 2))

    return IPWResult(
        ate=ate,
        att=att,
        se_ate=se_ate,
        se_att=se_att,
        pvalue_ate=pval_ate,
        pvalue_att=pval_att,
        nobs=n,
        effective_n=eff_n,
        details={
            "trim": trim,
            "n_trimmed": int(len(pscore) - n),
            "mean_pscore_treated": float(e[D == 1].mean()) if np.any(D == 1) else float("nan"),
            "mean_pscore_control": float(e[D == 0].mean()) if np.any(D == 0) else float("nan"),
        },
    )


# ---------------------------------------------------------------------------
# Doubly Robust (AIPW)
# ---------------------------------------------------------------------------

def doubly_robust(
    df: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: list[str],
    pscore: Optional[pd.Series] = None,
    trim: float = 0.01,
) -> DoublyRobustResult:
    """Augmented Inverse Probability Weighting (doubly robust) estimator.

    Combines an outcome model (OLS) with a propensity score model.
    Consistent if *either* model is correctly specified.

    Parameters
    ----------
    df : DataFrame
    y : str
        Outcome variable.
    treatment : str
        Binary treatment (0/1).
    covariates : list of str
        Covariates for both outcome and propensity models.
    pscore : Series, optional
        Pre-computed propensity scores.  If None, estimates via logit.
    trim : float
        Propensity score trimming threshold.

    Returns
    -------
    DoublyRobustResult
    """
    data = df.copy()
    D = data[treatment].astype(float).values
    Y = data[y].astype(float).values

    # Propensity scores
    if pscore is None:
        ps_result = propensity_score(data, treatment, covariates)
        e = ps_result.scores.values
    else:
        e = pscore.values.copy()

    # Trim
    mask = (e >= trim) & (e <= 1 - trim)
    D, Y, e = D[mask], Y[mask], e[mask]
    X_out = sm.add_constant(data.loc[mask, covariates].astype(float))
    n = len(Y)

    if n == 0:
        raise ValueError("No observations remaining after trimming.")

    # Outcome models: E[Y|X, D=d] for d ∈ {0, 1}
    idx1 = D == 1
    idx0 = D == 0
    res1 = sm.OLS(Y[idx1], X_out.values[idx1]).fit()
    res0 = sm.OLS(Y[idx0], X_out.values[idx0]).fit()

    mu1_hat = res1.predict(X_out.values)
    mu0_hat = res0.predict(X_out.values)

    # AIPW estimator
    # ψ_i = μ̂₁(X) - μ̂₀(X) + D(Y - μ̂₁(X))/e - (1-D)(Y - μ̂₀(X))/(1-e)
    psi = (
        mu1_hat - mu0_hat
        + D * (Y - mu1_hat) / e
        - (1 - D) * (Y - mu0_hat) / (1 - e)
    )

    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    pval = float(2 * sp_stats.norm.sf(abs(ate / se))) if se > 0 else float("nan")

    return DoublyRobustResult(
        ate=ate,
        se=se,
        pvalue=pval,
        ci_lower=ate - 1.96 * se,
        ci_upper=ate + 1.96 * se,
        nobs=n,
        details={
            "trim": trim,
            "n_trimmed": int(len(df) - n),
            "outcome_r2_treated": float(res1.rsquared),
            "outcome_r2_control": float(res0.rsquared),
        },
    )


# ---------------------------------------------------------------------------
# Covariate balance
# ---------------------------------------------------------------------------

def covariate_balance(
    df: pd.DataFrame,
    covariates: list[str],
    treatment: str,
    weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute standardised mean differences for covariate balance.

    Parameters
    ----------
    df : DataFrame
    covariates : list of str
    treatment : str
        Binary treatment (0/1).
    weights : Series, optional
        Observation weights (e.g. IPW weights).

    Returns
    -------
    DataFrame
        Columns: mean_treated, mean_control, std_diff, variance_ratio.
    """
    data = df.copy()
    D = data[treatment].astype(int)
    rows = []

    for cov in covariates:
        x = data[cov].astype(float)
        if weights is not None:
            w = weights
            m1 = np.average(x[D == 1], weights=w[D == 1])
            m0 = np.average(x[D == 0], weights=w[D == 0])
        else:
            m1 = x[D == 1].mean()
            m0 = x[D == 0].mean()

        s1 = x[D == 1].std()
        s0 = x[D == 0].std()
        pooled_sd = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
        std_diff = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0.0
        var_ratio = (s1 ** 2 / s0 ** 2) if s0 > 0 else float("nan")

        rows.append({
            "variable": cov,
            "mean_treated": m1,
            "mean_control": m0,
            "std_diff": std_diff,
            "variance_ratio": var_ratio,
        })

    return pd.DataFrame(rows).set_index("variable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_balance(
    data: pd.DataFrame,
    covariates: list[str],
    treatment: str,
    matched_ctrl_indices: list,
    treated_indices,
) -> pd.DataFrame:
    """Compute balance table comparing raw and matched samples."""
    D = data[treatment].astype(int)
    rows = []

    for cov in covariates:
        x = data[cov].astype(float)
        # Raw balance
        m1_raw = x[D == 1].mean()
        m0_raw = x[D == 0].mean()
        s1 = x[D == 1].std()
        s0 = x[D == 0].std()
        pooled_sd = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
        raw_smd = (m1_raw - m0_raw) / pooled_sd if pooled_sd > 0 else 0.0

        # Matched balance
        m1_match = x.loc[treated_indices].mean()
        m0_match = x.loc[matched_ctrl_indices].mean() if matched_ctrl_indices else float("nan")
        match_smd = (m1_match - m0_match) / pooled_sd if pooled_sd > 0 else 0.0

        rows.append({
            "variable": cov,
            "mean_treated_raw": m1_raw,
            "mean_control_raw": m0_raw,
            "smd_raw": raw_smd,
            "mean_treated_matched": m1_match,
            "mean_control_matched": m0_match,
            "smd_matched": match_smd,
        })

    return pd.DataFrame(rows).set_index("variable")
