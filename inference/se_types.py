"""SE-type resolution helpers.

This module is now a thin shim around ``econtools._core.cov_mapping``.
The unified resolver supports both statsmodels and linearmodels backends.

Public API
----------
VALID_COV_TYPES
resolve_cov_args(cov_type, maxlags=None, groups=None) -> dict
"""

from __future__ import annotations

from typing import Any

from econtools._core.cov_mapping import (  # noqa: F401
    VALID_COV_TYPES,
    resolve_cov_args as _resolve_cov_args,
)


def resolve_cov_args(
    cov_type: str,
    maxlags: int | None = None,
    groups: Any = None,
) -> dict[str, Any]:
    """Map a friendly cov_type label to statsmodels ``.fit()`` kwargs.

    Thin wrapper around :func:`econtools._core.cov_mapping.resolve_cov_args`
    with ``backend='sm'`` so existing call-sites continue to work unchanged.
    """
    return _resolve_cov_args(cov_type, backend="sm", maxlags=maxlags, groups=groups)


__all__ = ["VALID_COV_TYPES", "resolve_cov_args"]
