"""Inspection and profiling utilities for the data pipeline.

All functions take a DataFrame and return a DataFrame or dict —
outputs are loggable and composable.

Public API
----------
missing_report(df, threshold=None)
audit_dtypes(df)
cardinality(df)
balance_report(df, entity, time)
panel_summary(df)
dist_summary(df, col)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# missing_report
# ---------------------------------------------------------------------------


def missing_report(
    df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Return a per-column summary of missing values.

    Parameters
    ----------
    df:
        Input DataFrame.
    threshold:
        If given, only return columns with a missing fraction > threshold
        (e.g. ``0.05`` returns columns with >5 % missing).

    Returns
    -------
    DataFrame with columns ``['count', 'pct', 'dtype']`` indexed by column name.
    """
    n = len(df)
    counts = df.isnull().sum()
    pcts = counts / n * 100
    result = pd.DataFrame(
        {"count": counts, "pct": pcts.round(2), "dtype": df.dtypes}
    )
    result.index.name = "column"
    if threshold is not None:
        result = result[result["pct"] > threshold * 100]
    return result.sort_values("pct", ascending=False)


# ---------------------------------------------------------------------------
# audit_dtypes
# ---------------------------------------------------------------------------


def audit_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Flag object-dtype columns that look like they should be numeric.

    A column is flagged if it has ``object`` dtype and all non-null values
    can be parsed as floats.

    Returns
    -------
    DataFrame with columns ``['dtype', 'n_nonnull', 'parseable_as_float',
    'suggestion']``.
    """
    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        series = df[col].dropna()
        parseable = False
        if dtype == object and len(series) > 0:
            try:
                pd.to_numeric(series, errors="raise")
                parseable = True
            except (ValueError, TypeError):
                pass
        rows.append(
            {
                "column": col,
                "dtype": str(dtype),
                "n_nonnull": len(series),
                "parseable_as_float": parseable,
                "suggestion": "cast to numeric" if parseable else "",
            }
        )
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------------------
# cardinality
# ---------------------------------------------------------------------------


def cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique-value counts per column.

    Returns
    -------
    DataFrame with columns ``['n_unique', 'pct_unique', 'dtype']``.
    """
    n = len(df)
    n_unique = df.nunique(dropna=False)
    result = pd.DataFrame(
        {
            "n_unique": n_unique,
            "pct_unique": (n_unique / n * 100).round(2),
            "dtype": df.dtypes,
        }
    )
    result.index.name = "column"
    return result.sort_values("n_unique")


# ---------------------------------------------------------------------------
# balance_report
# ---------------------------------------------------------------------------


def balance_report(
    df: pd.DataFrame,
    entity: str,
    time: str,
) -> pd.DataFrame:
    """Pivot entity × time showing observation counts; flag gaps.

    Parameters
    ----------
    df:
        Long-format panel DataFrame (not necessarily MultiIndex).
    entity:
        Column name (or index level) for the entity identifier.
    time:
        Column name (or index level) for the time period.

    Returns
    -------
    DataFrame pivoted entity × time; cells are observation counts (0 = gap).
    """
    _df = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
    pivot = (
        _df.groupby([entity, time])
        .size()
        .unstack(fill_value=0)
    )
    return pivot


# ---------------------------------------------------------------------------
# panel_summary
# ---------------------------------------------------------------------------


def panel_summary(
    df: pd.DataFrame,
    entity: str | None = None,
    time: str | None = None,
) -> dict[str, Any]:
    """Summarise panel structure: N, T, balance, obs per entity.

    Accepts either a ``MultiIndex`` DataFrame (entity, time) or a
    plain DataFrame with *entity* and *time* column arguments.

    Returns
    -------
    dict with keys: ``n_entities``, ``n_periods``, ``n_obs``,
    ``balanced``, ``min_obs_per_entity``, ``max_obs_per_entity``,
    ``avg_obs_per_entity``, ``entity_col``, ``time_col``.
    """
    if isinstance(df.index, pd.MultiIndex) and entity is None and time is None:
        entity_vals = df.index.get_level_values(0)
        time_vals = df.index.get_level_values(1)
        entity = df.index.names[0]
        time = df.index.names[1]
    else:
        if entity is None or time is None:
            raise ValueError(
                "Provide 'entity' and 'time' column names, or pass a "
                "MultiIndex DataFrame."
            )
        _df = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
        entity_vals = _df[entity]
        time_vals = _df[time]

    obs_per_entity = pd.Series(entity_vals).value_counts()
    n_entities = entity_vals.nunique()
    n_periods = time_vals.nunique()
    n_obs = len(df)
    balanced = bool(n_obs == n_entities * n_periods)

    return {
        "n_entities": int(n_entities),
        "n_periods": int(n_periods),
        "n_obs": int(n_obs),
        "balanced": balanced,
        "min_obs_per_entity": int(obs_per_entity.min()),
        "max_obs_per_entity": int(obs_per_entity.max()),
        "avg_obs_per_entity": round(float(obs_per_entity.mean()), 2),
        "entity_col": entity,
        "time_col": time,
    }


# ---------------------------------------------------------------------------
# dist_summary
# ---------------------------------------------------------------------------


def dist_summary(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Return distribution statistics for a single column.

    Wraps ``scipy.stats.describe`` plus quantiles.

    Returns
    -------
    dict with keys: ``n``, ``mean``, ``variance``, ``std``,
    ``min``, ``max``, ``skewness``, ``kurtosis``,
    ``p1``, ``p5``, ``p25``, ``p50``, ``p75``, ``p95``, ``p99``.
    """
    series = df[col].dropna().astype(float)
    arr = series.to_numpy()
    desc = stats.describe(arr)
    quantiles = np.quantile(arr, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(desc.nobs),
        "mean": float(desc.mean),
        "variance": float(desc.variance),
        "std": float(np.sqrt(desc.variance)),
        "min": float(desc.minmax[0]),
        "max": float(desc.minmax[1]),
        "skewness": float(desc.skewness),
        "kurtosis": float(desc.kurtosis),  # excess kurtosis (Fisher)
        "p1": float(quantiles[0]),
        "p5": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
    }
