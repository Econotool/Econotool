"""Panel construction and merge utilities.

Public API
----------
set_panel_index(df, entity, time)
verify_panel_index(df)
is_balanced(df)
merge_audit(left, right, merged, keys)
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Panel indexing
# ---------------------------------------------------------------------------


def set_panel_index(
    df: pd.DataFrame,
    entity: str,
    time: str,
    sort: bool = True,
) -> pd.DataFrame:
    """Set a ``(entity, time)`` MultiIndex and optionally sort.

    linearmodels requires entity as the first level and time as the second.
    The original *entity* and *time* columns are dropped from the DataFrame
    body (they become the index).

    Parameters
    ----------
    df:
        Long-format panel DataFrame with *entity* and *time* as regular columns.
    entity:
        Column name for the entity identifier.
    time:
        Column name for the time period.
    sort:
        Sort the index after setting (required for correct lag/diff operations).

    Returns
    -------
    DataFrame with a ``(entity, time)`` MultiIndex.
    """
    out = df.set_index([entity, time])
    if sort:
        out = out.sort_index()
    return out


def verify_panel_index(df: pd.DataFrame) -> None:
    """Assert that *df* has a ``(entity, time)`` MultiIndex.

    Raises
    ------
    AssertionError
        If the index is not a 2-level MultiIndex.
    """
    assert isinstance(df.index, pd.MultiIndex), (
        "DataFrame does not have a MultiIndex. "
        "Call set_panel_index() first."
    )
    assert df.index.nlevels == 2, (
        f"Expected a 2-level MultiIndex (entity, time), "
        f"got {df.index.nlevels} levels: {df.index.names}."
    )


def is_balanced(df: pd.DataFrame) -> bool:
    """Return True if the panel is balanced (every entity has the same T periods).

    *df* must have a ``(entity, time)`` MultiIndex.
    """
    verify_panel_index(df)
    obs_per_entity = df.groupby(level=0).size()
    return bool(obs_per_entity.nunique() == 1)


# ---------------------------------------------------------------------------
# Merge audit
# ---------------------------------------------------------------------------


def merge_audit(
    left: pd.DataFrame,
    right: pd.DataFrame,
    merged: pd.DataFrame,
    keys: list[str] | str,
) -> dict[str, Any]:
    """Diagnose a merge: report matched, left-only, right-only, and duplicate rows.

    Parameters
    ----------
    left:
        Left DataFrame before merge.
    right:
        Right DataFrame before merge.
    merged:
        Result of ``pd.merge(left, right, ...)``.
    keys:
        Merge key column(s) — same as used in the merge call.

    Returns
    -------
    dict with keys:
        ``n_left``, ``n_right``, ``n_merged``,
        ``n_matched``, ``n_left_only``, ``n_right_only``,
        ``n_duplicate_keys_left``, ``n_duplicate_keys_right``,
        ``row_increase``.
    """
    if isinstance(keys, str):
        keys = [keys]

    # Indicator merge to count matched / unmatched
    indicator_merge = pd.merge(
        left[keys].drop_duplicates(),
        right[keys].drop_duplicates(),
        on=keys,
        how="outer",
        indicator=True,
    )
    n_matched = int((indicator_merge["_merge"] == "both").sum())
    n_left_only = int((indicator_merge["_merge"] == "left_only").sum())
    n_right_only = int((indicator_merge["_merge"] == "right_only").sum())

    n_dup_left = int(left[keys].duplicated().sum())
    n_dup_right = int(right[keys].duplicated().sum())

    return {
        "n_left": len(left),
        "n_right": len(right),
        "n_merged": len(merged),
        "n_matched": n_matched,
        "n_left_only": n_left_only,
        "n_right_only": n_right_only,
        "n_duplicate_keys_left": n_dup_left,
        "n_duplicate_keys_right": n_dup_right,
        "row_increase": len(merged) - len(left),
    }
