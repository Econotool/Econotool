"""Normality tests — re-export from diagnostics.normality."""

from econtools.diagnostics.normality import (  # noqa: F401
    jarque_bera,
    omnibus_test,
    shapiro_wilk,
)

__all__ = ["jarque_bera", "omnibus_test", "shapiro_wilk"]
