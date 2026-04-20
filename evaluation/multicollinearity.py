"""Multicollinearity diagnostics — re-export from diagnostics.multicollinearity."""

from econtools.diagnostics.multicollinearity import (  # noqa: F401
    compute_vif,
    condition_number,
)

__all__ = ["compute_vif", "condition_number"]
