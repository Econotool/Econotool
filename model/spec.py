"""Declarative model specification.

A :class:`ModelSpec` fully describes what the economist wants to estimate.
Pass it to :func:`econtools.fit.fit_model` to produce an :class:`Estimate`.

Multi-equation system estimators (SUR, 3SLS, system GMM) take a
:class:`SystemSpec` instead — a sibling container that carries a mapping
of equation labels to per-equation :class:`EquationSpec` objects.

Public API
----------
ModelSpec
EquationSpec
SystemSpec
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification of a single-equation econometric model.

    Parameters
    ----------
    dep_var:
        Name of the dependent variable column.
    exog_vars:
        Names of exogenous regressors (constant added unless
        ``add_constant=False``).
    endog_vars:
        Endogenous regressors (IV models only).
    instruments:
        Excluded instruments (IV models only).
    entity_col:
        Entity identifier column for panel models.
    time_col:
        Time identifier column for panel models.
    weights_col:
        Column of observation weights (WLS only).
    effects:
        Fixed-effects specification: ``'entity'``, ``'time'``, or
        ``'both'``.  ``None`` for pooled/cross-section models.
    estimator:
        One of ``'ols'``, ``'wls'``, ``'2sls'``, ``'liml'``,
        ``'fe'``, ``'re'``, ``'fd'``, ``'pooled'``, ``'probit'``,
        ``'logit'``, ``'gmm'``, ``'cue_gmm'``, ``'arellano_bond'``
        (aliases: ``'ab_gmm'``, ``'dynamic_panel'``).
    cov_type:
        Covariance estimator — one of :data:`econtools._core.cov_mapping.VALID_COV_TYPES`.
        Includes ``'cluster2'`` (two-way clustering, panel/``lm`` only) and
        ``'driscoll_kraay'`` (spatial HAC, ``lm`` only).
    cov_kwargs:
        Extra keyword arguments forwarded to the covariance estimator
        (e.g. ``maxlags``, ``groups``, ``groups2``).  GMM-specific keys:
        ``weight_type`` (``'robust'``, ``'unadjusted'``, ``'kernel'``,
        ``'clustered'``), ``iter_limit`` (2 = two-step, high = iterated),
        ``bandwidth``, ``kernel``.
    add_constant:
        Prepend a constant column to the design matrix.
    alpha:
        Significance level for confidence intervals (default 0.05 → 95% CI).
    lags:
        Number of lagged dependent-variable terms to include as regressors
        (Arellano-Bond only; default 1 reproduces the standard AR(1)
        dynamic panel).  Ignored by other estimators.
    max_lags:
        Deepest lag of the dependent variable to use as a level
        instrument for the first-differenced lagged endogenous regressors
        (Arellano-Bond only).  ``None`` uses all available lags.
    """

    dep_var: str
    exog_vars: list[str]
    endog_vars: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    entity_col: str | None = None
    time_col: str | None = None
    weights_col: str | None = None
    effects: str | None = None       # "entity" | "time" | "both"
    estimator: str = "ols"           # see docstring for full set
    cov_type: str = "classical"
    cov_kwargs: dict = field(default_factory=dict)
    add_constant: bool = True
    alpha: float = 0.05
    lags: int = 1
    max_lags: int | None = None


@dataclass(frozen=True)
class EquationSpec:
    """Single equation in a system model.

    Parameters
    ----------
    dep_var:
        Dependent variable column name for this equation.
    exog_vars:
        Exogenous regressor column names (constant added unless
        ``add_constant=False`` on the enclosing :class:`SystemSpec`).
    endog_vars:
        Endogenous regressors (3SLS / system GMM only).
    instruments:
        Excluded instruments (3SLS / system GMM only).
    """

    dep_var: str
    exog_vars: list[str]
    endog_vars: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SystemSpec:
    """Declarative specification of a multi-equation system.

    Used by SUR, 3SLS, and system-GMM estimators.  Pass to
    :func:`econtools.fit.fit_model` alongside a :class:`pandas.DataFrame`
    whose columns include every variable referenced across equations.

    Parameters
    ----------
    equations:
        Mapping from human-readable equation label to
        :class:`EquationSpec`.  Label ordering is preserved in the
        output parameter vector.
    estimator:
        Currently ``'sur'``.  ``'3sls'`` and ``'system_gmm'`` are
        planned.
    cov_type:
        Covariance estimator passed through to linearmodels.  One of
        ``'classical'``/``'unadjusted'``, ``'robust'``, ``'kernel'``,
        ``'cluster'``.
    cov_kwargs:
        Extra keyword arguments forwarded to the covariance estimator
        (e.g. ``maxlags``, ``bandwidth``, ``clusters``).
    add_constant:
        Prepend a constant column to each equation's exogenous design
        matrix.
    alpha:
        Significance level for confidence intervals (default 0.05).
    method:
        Estimation method for SUR — ``None`` (auto), ``'gls'``, or
        ``'ols'``.
    iterate:
        Iterate GLS until convergence (SUR only).
    iter_limit:
        Maximum iterations for iterative GLS (SUR only).
    full_cov:
        Use full cross-equation covariance matrix in GLS (SUR only).
    """

    equations: dict
    estimator: str = "sur"
    cov_type: str = "robust"
    cov_kwargs: dict = field(default_factory=dict)
    add_constant: bool = True
    alpha: float = 0.05
    method: str | None = None        # None | "gls" | "ols"
    iterate: bool = False
    iter_limit: int = 100
    full_cov: bool = True
