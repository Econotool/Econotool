"""Standalone numpy estimators for bootstrap replications.

These are pure-numpy implementations of OLS and 2SLS used exclusively
inside the bootstrap engine for speed. Do not use these for production
estimation — use ``econtools.fit.fit_model`` instead.

Internal API.
"""

from __future__ import annotations

import warnings

import numpy as np

# (coefs, residuals, fitted_values)
_OLSResult = tuple[np.ndarray, np.ndarray, np.ndarray]


def ols_fit(X: np.ndarray, y: np.ndarray) -> _OLSResult:
    """Fit OLS via closed-form (X'X)^{-1}X'y.

    Falls back to ``numpy.linalg.lstsq`` when X'X is near-singular
    (condition number > 1e12) or when the solve raises ``LinAlgError``.

    Parameters
    ----------
    X:
        Design matrix of shape (n, k).
    y:
        Response vector of length n.

    Returns
    -------
    coefs: shape (k,)
    residuals: shape (n,)
    fitted: shape (n,)
    """
    XtX = X.T @ X
    Xty = X.T @ y

    use_lstsq = False
    try:
        cond = np.linalg.cond(XtX)
        if cond > 1e12:
            use_lstsq = True
    except np.linalg.LinAlgError:
        use_lstsq = True

    if not use_lstsq:
        try:
            coefs = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            use_lstsq = True

    if use_lstsq:
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    fitted = X @ coefs
    resid = y - fitted
    return coefs, resid, fitted


def twosls_fit(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """2SLS estimator.

    First stage: regress each endogenous regressor on [X_exog, Z].
    Second stage: regress y on [X_exog, X_endog_hat].

    Structural residuals are computed using actual X_endog (not hat),
    consistent with the definition needed for the wild bootstrap.

    Currently enforces a single endogenous variable. Raise ``NotImplementedError``
    for multiple endogenous regressors.

    Parameters
    ----------
    y:
        Dependent variable, length n.
    X_exog:
        Exogenous regressors (with constant prepended if requested), shape (n, k_exog).
    X_endog:
        Endogenous regressors, shape (n, k_endog). Currently k_endog must be 1.
    Z:
        Excluded instruments, shape (n, k_z).

    Returns
    -------
    coefs : shape (k_exog + k_endog,) — second-stage coefficients [exog | endog]
    resid : structural residuals using actual X_endog
    fitted : structural fitted values using actual X_endog
    first_stage_F : partial F-statistic (scalar). NaN if not computable.
    """
    if X_endog.ndim == 1:
        X_endog = X_endog[:, np.newaxis]
    if Z.ndim == 1:
        Z = Z[:, np.newaxis]

    k_endog = X_endog.shape[1]
    if k_endog > 1:
        raise NotImplementedError(
            "run_bootstrap currently supports only one endogenous regressor for 2SLS. "
            "Pass a single column in `endog`."
        )

    # --- First stage ---
    X_first = np.column_stack([X_exog, Z])
    X_endog_hat = np.empty_like(X_endog)
    for j in range(k_endog):
        c1, _, _ = ols_fit(X_first, X_endog[:, j])
        X_endog_hat[:, j] = X_first @ c1

    # --- Second stage ---
    X_second = np.column_stack([X_exog, X_endog_hat])
    coefs, _, _ = ols_fit(X_second, y)

    # --- Structural residuals (actual endog, not hat) ---
    X_structural = np.column_stack([X_exog, X_endog])
    fitted = X_structural @ coefs
    resid = y - fitted

    # --- First-stage partial F (only for single endog) ---
    first_stage_F = _partial_f_stat(X_first, X_exog, X_endog[:, 0])

    return coefs, resid, fitted, first_stage_F


def _partial_f_stat(
    X_full: np.ndarray,
    X_restricted: np.ndarray,
    y: np.ndarray,
) -> float:
    """Partial F-statistic for excluded instruments in first stage.

    Computes F = [(RSS_r - RSS_u) / q] / [RSS_u / (n - k_u)]
    where q = number of excluded instruments.

    Returns NaN if not computable.
    """
    n = len(y)
    k_full = X_full.shape[1]
    k_restricted = X_restricted.shape[1]
    q = k_full - k_restricted

    if q <= 0:
        return float("nan")

    _, resid_full, _ = ols_fit(X_full, y)
    _, resid_restr, _ = ols_fit(X_restricted, y)

    rss_full = float(np.dot(resid_full, resid_full))
    rss_restr = float(np.dot(resid_restr, resid_restr))

    df_resid = n - k_full
    if df_resid <= 0 or rss_full < 1e-15:
        return float("nan")

    return ((rss_restr - rss_full) / q) / (rss_full / df_resid)
