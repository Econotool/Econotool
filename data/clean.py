"""Cleaning utilities for the data pipeline.

All functions are pure: ``(DataFrame, ...) → DataFrame``.

Public API
----------
snake_case(name)          — convert a single string to snake_case
rename_snake(df)          — apply snake_case to all column names
winsorise(df, col, ...)   — clip column at quantiles
assert_nonnegative(df, cols)
assert_range(df, col, lo, hi)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# snake_case
# ---------------------------------------------------------------------------

_RE_UPPER_SEQUENCE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_RE_LOWER_UPPER = re.compile(r"([a-z0-9])([A-Z])")
_RE_NONALNUM = re.compile(r"[^a-z0-9]+")


def snake_case(name: str) -> str:
    """Convert *name* to ``lower_snake_case``.

    Handles CamelCase, ABBRCase, spaces, hyphens, dots, and consecutive separators.

    Examples
    --------
    >>> snake_case("YearsOfSchooling")
    'years_of_schooling'
    >>> snake_case("logWage2")
    'log_wage2'
    >>> snake_case("GDP per capita")
    'gdp_per_capita'
    >>> snake_case("ABCdef")
    'ab_cdef'
    """
    # Handle runs of uppercase: ABCdef → AB_Cdef
    s = _RE_UPPER_SEQUENCE.sub(r"\1_\2", name)
    # Insert underscore between lowercase/digit and following uppercase: logWage → log_Wage
    s = _RE_LOWER_UPPER.sub(r"\1_\2", s)
    s = s.lower()
    # Replace any sequence of non-alphanumeric characters with a single underscore
    s = _RE_NONALNUM.sub("_", s)
    # Strip leading/trailing underscores
    return s.strip("_")


def rename_snake(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with all column names converted to snake_case."""
    return df.rename(columns={c: snake_case(c) for c in df.columns})


# ---------------------------------------------------------------------------
# winsorise
# ---------------------------------------------------------------------------


def winsorise(
    df: pd.DataFrame,
    col: str,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Return a copy of *df* with *col* clipped at the specified quantiles.

    Parameters
    ----------
    df:
        Input DataFrame.
    col:
        Column to winsorise.
    lower:
        Lower quantile (default 0.01 → 1st percentile).
    upper:
        Upper quantile (default 0.99 → 99th percentile).

    Returns
    -------
    DataFrame with *col* replaced by the winsorised version.
    Adds a column ``{col}_winsorised`` boolean mask is NOT added —
    the clip is silent. Document in provenance if needed.
    """
    if lower >= upper:
        raise ValueError(f"lower ({lower}) must be < upper ({upper})")
    series = df[col].dropna()
    lo_val = float(np.quantile(series, lower))
    hi_val = float(np.quantile(series, upper))
    out = df.copy()
    out[col] = out[col].clip(lower=lo_val, upper=hi_val)
    return out


# ---------------------------------------------------------------------------
# Assertion utilities
# ---------------------------------------------------------------------------


def assert_nonnegative(df: pd.DataFrame, cols: list[str] | str) -> None:
    """Assert that all values in *cols* are >= 0.

    Parameters
    ----------
    df:
        DataFrame to check.
    cols:
        Column name or list of column names.

    Raises
    ------
    AssertionError
        If any value is negative, with the column name and count of violations.
    """
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        neg_count = int((df[col].dropna() < 0).sum())
        assert neg_count == 0, (
            f"Column '{col}' has {neg_count} negative value(s)."
        )


def assert_range(
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
) -> None:
    """Assert that all non-null values of *col* are in [*lo*, *hi*].

    Parameters
    ----------
    df:
        DataFrame to check.
    col:
        Column name.
    lo:
        Inclusive lower bound.
    hi:
        Inclusive upper bound.

    Raises
    ------
    AssertionError
        If any value falls outside [lo, hi], with count of violations.
    """
    series = df[col].dropna()
    out_of_range = int(((series < lo) | (series > hi)).sum())
    assert out_of_range == 0, (
        f"Column '{col}' has {out_of_range} value(s) outside [{lo}, {hi}]."
    )
