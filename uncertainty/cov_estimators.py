"""Covariance estimator resolution — thin wrapper around _core.cov_mapping.

Public API
----------
VALID_COV_TYPES
resolve_cov_args(cov_type, *, backend, maxlags, groups) -> dict
"""

from econtools._core.cov_mapping import (  # noqa: F401
    VALID_COV_TYPES,
    resolve_cov_args,
)

__all__ = ["VALID_COV_TYPES", "resolve_cov_args"]
