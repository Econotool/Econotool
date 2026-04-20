"""Influence diagnostics — Cook's distance, DFFITS, DFBETAs.

Uses ``statsmodels.stats.outliers_influence.OLSInfluence`` under the hood.
Requires an OLS/WLS result (accessed via ``Estimate.raw``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import OLSInfluence

from econtools._core.types import Estimate


def _get_influence(result: Estimate) -> OLSInfluence:
    """Extract OLSInfluence from an Estimate, validating the model type."""
    raw = result.raw
    # OLSInfluence needs a statsmodels RegressionResults with .model.exog as 2-D array
    if not hasattr(raw, "model") or not hasattr(raw.model, "exog"):
        raise TypeError(
            "Influence diagnostics require an OLS or WLS result. "
            f"Got model_type='{result.model_type}'."
        )
    exog = getattr(raw.model, "exog", None)
    if exog is None or exog.ndim != 2:
        raise TypeError(
            "Influence diagnostics require an OLS or WLS result. "
            f"Got model_type='{result.model_type}'."
        )
    return OLSInfluence(raw)


def cooks_distance(result: Estimate) -> pd.Series:
    """Cook's distance for each observation.

    Parameters
    ----------
    result:
        An :class:`Estimate` from an OLS or WLS fit.

    Returns
    -------
    pd.Series
        Cook's distance indexed like the residuals.
    """
    infl = _get_influence(result)
    d = infl.cooks_distance[0]
    return pd.Series(d, index=result.resid.index, name="cooks_d")


def dffits(result: Estimate) -> pd.Series:
    """DFFITS influence measure for each observation.

    DFFITS measures how much the fitted value changes when observation *i*
    is deleted, scaled by its standard error.

    Parameters
    ----------
    result:
        An :class:`Estimate` from an OLS or WLS fit.

    Returns
    -------
    pd.Series
        DFFITS values indexed like the residuals.
    """
    infl = _get_influence(result)
    d = infl.dffits[0]
    return pd.Series(d, index=result.resid.index, name="dffits")


def dfbetas(result: Estimate) -> pd.DataFrame:
    """DFBETAs influence measure for each observation and coefficient.

    DFBETAs measure how much each coefficient changes when observation *i*
    is deleted, scaled by the coefficient's standard error.

    Parameters
    ----------
    result:
        An :class:`Estimate` from an OLS or WLS fit.

    Returns
    -------
    pd.DataFrame
        Rows = observations, columns = coefficient names.
    """
    infl = _get_influence(result)
    d = infl.dfbetas
    return pd.DataFrame(
        d,
        index=result.resid.index,
        columns=result.params.index,
    )


def hat_matrix_diag(result: Estimate) -> pd.Series:
    """Diagonal of the hat (leverage) matrix.

    Parameters
    ----------
    result:
        An :class:`Estimate` from an OLS or WLS fit.

    Returns
    -------
    pd.Series
        Leverage values (h_ii) indexed like the residuals.
    """
    infl = _get_influence(result)
    h = infl.hat_matrix_diag
    return pd.Series(h, index=result.resid.index, name="leverage")


def studentized_residuals(result: Estimate, external: bool = True) -> pd.Series:
    """Studentized residuals (internally or externally studentized).

    Parameters
    ----------
    result:
        An :class:`Estimate` from an OLS or WLS fit.
    external:
        If ``True`` (default), use leave-one-out variance (externally
        studentized).  If ``False``, use the full-sample variance.

    Returns
    -------
    pd.Series
        Studentized residuals indexed like the original residuals.
    """
    infl = _get_influence(result)
    if external:
        r = infl.resid_studentized_external
    else:
        r = infl.resid_studentized_internal
    return pd.Series(r, index=result.resid.index, name="studentized_resid")


def influence_summary(result: Estimate, threshold_cooks: float = 1.0) -> pd.DataFrame:
    """Summary table of influence diagnostics for each observation.

    Returns a DataFrame with columns: ``leverage``, ``studentized_resid``,
    ``cooks_d``, ``dffits``, and a boolean ``flagged`` column that is
    ``True`` when Cook's distance exceeds *threshold_cooks* or leverage
    exceeds ``2 * k / n``.

    Parameters
    ----------
    result:
        An :class:`Estimate` from an OLS or WLS fit.
    threshold_cooks:
        Cook's distance threshold for flagging (default 1.0).
    """
    cd = cooks_distance(result)
    df_ = dffits(result)
    lev = hat_matrix_diag(result)
    sr = studentized_residuals(result, external=True)

    n = result.fit.nobs
    k = len(result.params)
    leverage_cutoff = 2.0 * k / n

    summary = pd.DataFrame({
        "leverage": lev,
        "studentized_resid": sr,
        "cooks_d": cd,
        "dffits": df_,
    })
    summary["flagged"] = (cd > threshold_cooks) | (lev > leverage_cutoff)
    return summary
