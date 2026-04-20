"""Transformation utilities for the data pipeline.

All functions are pure: ``(DataFrame, ...) → DataFrame``.
New columns follow the project naming convention:
  log_<col>, lag_<col>_k<n>, lead_<col>_k<n>, d_<col>, g_<col>,
  <col>_sq, <col3> for cubic, <col1>_x_<col2>.

Public API
----------
log_col(df, col, name=None)
log1p_col(df, col, name=None)
lag(df, col, entity, k=1, name=None)
lead(df, col, entity, k=1, name=None)
diff_col(df, col, entity, k=1, name=None)
growth_rate(df, col, entity, k=1, name=None)
dummies(df, col, drop_first=True, prefix=None)
interact(df, col1, col2, name=None)
poly(df, col, degree=2, name=None)
standardise(df, col, name=None)
demean_within(df, col, entity, name=None)
time_trend(df, entity, name='t')
rolling_mean(df, col, entity, k, name=None)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Log transforms
# ---------------------------------------------------------------------------


def log_col(
    df: pd.DataFrame,
    col: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``log_<col>`` = ln(col) to a copy of *df*.

    Parameters
    ----------
    df:
        Input DataFrame.
    col:
        Column to transform.  Must be strictly positive (non-null).
    name:
        Override the new column name.

    Returns
    -------
    DataFrame with new column appended.
    """
    out = df.copy()
    new_col = name or f"log_{col}"
    out[new_col] = np.log(out[col])
    return out


def log1p_col(
    df: pd.DataFrame,
    col: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``log1p_<col>`` = ln(1 + col) to a copy of *df*.

    Safe for zero-inclusive variables.
    """
    out = df.copy()
    new_col = name or f"log1p_{col}"
    out[new_col] = np.log1p(out[col])
    return out


# ---------------------------------------------------------------------------
# Panel lags / leads / diffs
# ---------------------------------------------------------------------------


def _groupby_entity(df: pd.DataFrame, entity: str) -> pd.core.groupby.SeriesGroupBy:
    """Helper: group by entity column, handling MultiIndex DataFrames."""
    if isinstance(df.index, pd.MultiIndex):
        # entity is an index level name
        return df.groupby(level=entity)
    return df.groupby(entity)


def lag(
    df: pd.DataFrame,
    col: str,
    entity: str,
    k: int = 1,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``lag_<col>_k<k>`` = col shifted *k* periods back within entity.

    The DataFrame must be sorted by (entity, time) for correct results.
    """
    out = df.copy()
    new_col = name or f"lag_{col}_k{k}"
    out[new_col] = _groupby_entity(out, entity)[col].shift(k)
    return out


def lead(
    df: pd.DataFrame,
    col: str,
    entity: str,
    k: int = 1,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``lead_<col>_k<k>`` = col shifted *k* periods forward within entity."""
    out = df.copy()
    new_col = name or f"lead_{col}_k{k}"
    out[new_col] = _groupby_entity(out, entity)[col].shift(-k)
    return out


def diff_col(
    df: pd.DataFrame,
    col: str,
    entity: str,
    k: int = 1,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``d_<col>`` = first difference (Δₖ xᵢₜ) within entity."""
    out = df.copy()
    new_col = name or f"d_{col}"
    out[new_col] = _groupby_entity(out, entity)[col].diff(k)
    return out


def growth_rate(
    df: pd.DataFrame,
    col: str,
    entity: str,
    k: int = 1,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``g_<col>`` = percentage change (xₜ - xₜ₋ₖ)/xₜ₋ₖ within entity."""
    out = df.copy()
    new_col = name or f"g_{col}"
    out[new_col] = _groupby_entity(out, entity)[col].pct_change(k)
    return out


# ---------------------------------------------------------------------------
# Dummies, interactions, polynomials
# ---------------------------------------------------------------------------


def dummies(
    df: pd.DataFrame,
    col: str,
    drop_first: bool = True,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add one-hot dummy columns for *col* and return a copy of *df*.

    Parameters
    ----------
    df:
        Input DataFrame.
    col:
        Column to expand.
    drop_first:
        Drop the first category to avoid perfect multicollinearity.
    prefix:
        Prefix for new column names (default: *col*).

    Returns
    -------
    DataFrame with *col* replaced by dummy columns.
    """
    prefix = prefix or col
    dummies_df = pd.get_dummies(df[col], prefix=prefix, drop_first=drop_first)
    out = df.drop(columns=[col]).copy()
    return pd.concat([out, dummies_df], axis=1)


def interact(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``<col1>_x_<col2>`` = col1 * col2 to a copy of *df*."""
    out = df.copy()
    new_col = name or f"{col1}_x_{col2}"
    out[new_col] = out[col1] * out[col2]
    return out


def poly(
    df: pd.DataFrame,
    col: str,
    degree: int = 2,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``<col>_sq`` (degree=2) or ``<col>_d<n>`` for higher degrees.

    Only adds the *degree*-th power (not all lower powers — call multiple
    times if you need a full polynomial basis).
    """
    out = df.copy()
    suffix = "sq" if degree == 2 else f"d{degree}"
    new_col = name or f"{col}_{suffix}"
    out[new_col] = out[col] ** degree
    return out


# ---------------------------------------------------------------------------
# Standardisation and within-demeaning
# ---------------------------------------------------------------------------


def standardise(
    df: pd.DataFrame,
    col: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``z_<col>`` = (col - mean) / std to a copy of *df*.

    Uses the full-sample mean and std (not group-level).
    """
    out = df.copy()
    new_col = name or f"z_{col}"
    mu = out[col].mean()
    sigma = out[col].std()
    out[new_col] = (out[col] - mu) / sigma
    return out


def demean_within(
    df: pd.DataFrame,
    col: str,
    entity: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``dm_<col>`` = col - entity_mean(col) (within/FE demeaning)."""
    out = df.copy()
    new_col = name or f"dm_{col}"
    group_mean = _groupby_entity(out, entity)[col].transform("mean")
    out[new_col] = out[col] - group_mean
    return out


# ---------------------------------------------------------------------------
# Time trend and rolling
# ---------------------------------------------------------------------------


def time_trend(
    df: pd.DataFrame,
    entity: str,
    name: str = "t",
) -> pd.DataFrame:
    """Add integer time trend (0, 1, 2, …) within each entity.

    Parameters
    ----------
    df:
        Input DataFrame (should be sorted by entity, time).
    entity:
        Entity column or index level name.
    name:
        Name for the new trend column (default ``'t'``).
    """
    out = df.copy()
    out[name] = _groupby_entity(out, entity).cumcount()
    return out


def rolling_mean(
    df: pd.DataFrame,
    col: str,
    entity: str,
    k: int,
    name: str | None = None,
) -> pd.DataFrame:
    """Add ``roll<k>_<col>`` = rolling mean of width *k* within entity."""
    out = df.copy()
    new_col = name or f"roll{k}_{col}"
    out[new_col] = _groupby_entity(out, entity)[col].transform(
        lambda x: x.rolling(k).mean()
    )
    return out
