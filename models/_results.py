"""Frozen dataclasses for regression results.

This module is now a re-export shim.  The canonical definitions live in
``econtools._core.types``.  Import from here as normal — all public names
are preserved for backward compatibility.

Public API
----------
FitMetrics
RegressionResult  (alias for Estimate)
"""

from econtools._core.types import (  # noqa: F401
    Estimate,
    FitMetrics,
    RegressionResult,
)

__all__ = ["Estimate", "FitMetrics", "RegressionResult"]
