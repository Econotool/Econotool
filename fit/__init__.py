"""Estimation layer — Phase 2+.

The single public entry point is :func:`fit_model`, which dispatches to the
appropriate backend based on a :class:`~econtools.model.spec.ModelSpec`.

Old convenience functions (``fit_ols``, ``fit_wls``, etc.) are still
available from ``econtools.models`` during the transition period; they
will be removed in Phase D Step 15.
"""

from econtools.fit.estimators import fit_model  # noqa: F401

__all__ = ["fit_model"]
