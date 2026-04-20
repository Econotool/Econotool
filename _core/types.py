"""Canonical shared types for econtools.

Public API
----------
FitMetrics      — frozen dataclass of goodness-of-fit statistics
Estimate        — frozen dataclass for any fitted model result
RegressionResult — alias for Estimate (backward compatibility)
TestResult      — frozen dataclass for hypothesis test output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

import pandas as pd


@dataclass(frozen=True)
class FitMetrics:
    """Goodness-of-fit statistics for a fitted regression model.

    The three required fields (``nobs``, ``df_model``, ``df_resid``) are
    always available.  All others default to ``float('nan')`` so that
    panel, IV, and binary models can omit metrics that don't apply.
    """

    # --- always present ---
    nobs: int
    df_model: float
    df_resid: float

    # --- OLS / WLS ---
    r_squared: float = float("nan")
    r_squared_adj: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    log_likelihood: float = float("nan")
    f_stat: float = float("nan")
    f_pvalue: float = float("nan")
    rmse: float = float("nan")
    ssr: float = float("nan")
    ess: float = float("nan")
    tss: float = float("nan")

    # --- panel ---
    r_squared_within: float = float("nan")
    r_squared_between: float = float("nan")

    # --- binary (Probit / Logit) ---
    pseudo_r_squared: float = float("nan")

    # --- GMM / IV ---
    j_stat: float = float("nan")     # Hansen J overidentification statistic
    j_pvalue: float = float("nan")   # p-value for J-test
    kappa: float = float("nan")      # LIML kappa


@dataclass(frozen=True)
class Estimate:
    """Normalised result object for any fitted econometric model.

    The raw statsmodels or linearmodels result is accessible via ``.raw``
    for escape-hatch access to library-specific attributes.
    """

    model_type: str
    dep_var: str
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int_lower: pd.Series
    conf_int_upper: pd.Series
    resid: pd.Series
    fitted: pd.Series
    cov_params: pd.DataFrame
    cov_type: str
    fit: FitMetrics
    raw: Any
    formula: str | None = None


# Backward-compatible alias — keeps ``from econtools.models._results import
# RegressionResult`` working throughout the Phase B/C transition.
RegressionResult = Estimate


@dataclass(frozen=True)
class TestResult:
    """Output of a single hypothesis test.

    Parameters
    ----------
    test_name:
        Human-readable name of the test.
    statistic:
        Test statistic value.
    pvalue:
        Asymptotic p-value.
    df:
        Degrees of freedom — ``(df_num, df_denom)`` for F-tests, scalar
        for χ² / t-tests, or ``None`` if not applicable.
    distribution:
        Reference distribution string (e.g. ``'F'``, ``'Chi2'``, ``'t'``).
    null_hypothesis:
        Plain-text statement of H₀.
    reject:
        ``True`` if H₀ is rejected at the 5% level.
    details:
        Optional structured ancillary information (critical values, lag
        lengths, etc.).
    """

    test_name: str
    statistic: float
    pvalue: float
    df: Union[tuple[float, float], float, None]
    distribution: str
    null_hypothesis: str
    reject: bool
    details: dict = field(default_factory=dict)
