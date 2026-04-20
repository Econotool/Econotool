"""Data pipeline: ingest, inspect, clean, transform, construct, save."""

from econtools.data.io import (
    load_raw,
    load_dta,
    load_csv,
    save_curated,
    verify_hash,
    versioned_path,
)
from econtools.data.inspect import (
    missing_report,
    cardinality,
    balance_report,
    panel_summary,
    dist_summary,
    audit_dtypes,
)
from econtools.data.clean import (
    snake_case,
    rename_snake,
    winsorise,
    assert_nonnegative,
    assert_range,
)
from econtools.data.transform import (
    log_col,
    log1p_col,
    lag,
    lead,
    diff_col,
    growth_rate,
    dummies,
    interact,
    poly,
    standardise,
    demean_within,
    time_trend,
    rolling_mean,
)
from econtools.data.trend import (
    fit_trend,
    assess_trend_options,
    TrendResult,
    fit_polynomial_trend,
    predict_trend,
)
from econtools.data.construct import (
    set_panel_index,
    verify_panel_index,
    is_balanced,
    merge_audit,
)

__all__ = [
    # io
    "load_raw", "load_dta", "load_csv", "save_curated", "verify_hash", "versioned_path",
    # inspect
    "missing_report", "cardinality", "balance_report", "panel_summary", "dist_summary", "audit_dtypes",
    # clean
    "snake_case", "rename_snake", "winsorise", "assert_nonnegative", "assert_range",
    # transform
    "log_col", "log1p_col", "lag", "lead", "diff_col", "growth_rate",
    "dummies", "interact", "poly", "standardise", "demean_within", "time_trend", "rolling_mean",
    # trend
    "fit_trend", "assess_trend_options", "TrendResult",
    "fit_polynomial_trend", "predict_trend",
    # construct
    "set_panel_index", "verify_panel_index", "is_balanced", "merge_audit",
]
