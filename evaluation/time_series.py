"""Time-series diagnostics — re-export from diagnostics.time_series."""

from econtools.diagnostics.time_series import (  # noqa: F401
    granger_causality,
    lead_exogeneity_test,
    select_var_lag,
)

__all__ = ["granger_causality", "lead_exogeneity_test", "select_var_lag"]
