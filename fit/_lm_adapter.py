"""Linearmodels backend for the fit/ layer.

All functions take a :class:`~econtools.model.spec.ModelSpec` and a
:class:`pandas.DataFrame`, call the appropriate linearmodels estimator,
and return an :class:`~econtools._core.types.Estimate`.

Internal API.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from linearmodels.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)

from econtools._core.cov_mapping import resolve_cov_args
from econtools._core.types import Estimate
from econtools.fit._builders import (
    build_lm_gmm_result,
    build_lm_iv_result,
    build_lm_panel_result,
    build_lm_system_result,
)
from econtools.model.spec import ModelSpec, SystemSpec


def fit_iv_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit IV model (2SLS or LIML) from a :class:`ModelSpec`."""
    cols = (
        [spec.dep_var]
        + list(spec.exog_vars)
        + list(spec.endog_vars)
        + list(spec.instruments)
    )
    data = df[cols].dropna()

    y: pd.Series = data[spec.dep_var]
    exog: pd.DataFrame = data[list(spec.exog_vars)]
    endog: pd.DataFrame = data[list(spec.endog_vars)]
    instr: pd.DataFrame = data[list(spec.instruments)]

    if spec.add_constant:
        exog = sm.add_constant(exog)

    cov_args = resolve_cov_args(
        spec.cov_type,
        backend="lm",
        maxlags=spec.cov_kwargs.get("maxlags"),
        groups=spec.cov_kwargs.get("groups"),
    )

    estimator = spec.estimator.lower()
    if estimator == "liml":
        ModelClass = IVLIML
        model_type = "LIML"
    else:
        ModelClass = IV2SLS
        model_type = "2SLS"

    lm_result = ModelClass(y, exog, endog, instr).fit(**cov_args)

    result = build_lm_iv_result(lm_result, model_type, spec.dep_var, spec.cov_type, spec.alpha, y)

    # Extract LIML kappa if available
    if estimator == "liml":
        kappa = float(getattr(lm_result, "kappa", float("nan")))
        # Replace FitMetrics with kappa populated
        from dataclasses import replace
        fit_with_kappa = replace(result.fit, kappa=kappa)
        result = replace(result, fit=fit_with_kappa)

    return result


def fit_gmm_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit GMM-IV from a :class:`ModelSpec`.

    Supports two-step, iterated, and continuously updating (CUE) GMM via
    ``linearmodels.iv.IVGMM`` and ``linearmodels.iv.IVGMMCUE``.

    Weight type is controlled by ``spec.cov_kwargs['weight_type']`` (default
    ``'robust'``).  Iteration depth by ``spec.cov_kwargs['iter_limit']``
    (default 2 = two-step).  CUE ignores ``iter_limit``.
    """
    cols = (
        [spec.dep_var]
        + list(spec.exog_vars)
        + list(spec.endog_vars)
        + list(spec.instruments)
    )
    data = df[cols].dropna()

    y: pd.Series = data[spec.dep_var]
    exog: pd.DataFrame = data[list(spec.exog_vars)]
    endog: pd.DataFrame = data[list(spec.endog_vars)]
    instr: pd.DataFrame = data[list(spec.instruments)]

    if spec.add_constant:
        exog = sm.add_constant(exog)

    weight_type: str = spec.cov_kwargs.get("weight_type", "robust")
    iter_limit: int = spec.cov_kwargs.get("iter_limit", 2)

    # Constructor kwargs for the weight matrix
    wt_kwargs: dict = {}
    if weight_type == "kernel":
        wt_kwargs["kernel"] = spec.cov_kwargs.get("kernel", "bartlett")
        if "bandwidth" in spec.cov_kwargs:
            wt_kwargs["bandwidth"] = spec.cov_kwargs["bandwidth"]
    elif weight_type == "clustered":
        clusters = spec.cov_kwargs.get("groups")
        if clusters is None:
            raise ValueError(
                "weight_type='clustered' requires 'groups' in cov_kwargs."
            )
        wt_kwargs["clusters"] = clusters

    estimator_key = spec.estimator.lower()
    if estimator_key in ("cue_gmm", "ivgmmcue"):
        ModelClass = IVGMMCUE
        model_type = "GMM-CUE"
        constructor_kwargs: dict = {}
        if wt_kwargs:
            constructor_kwargs.update(wt_kwargs)
        lm_model = ModelClass(y, exog, endog, instr, **constructor_kwargs)
        cov_args = resolve_cov_args(
            spec.cov_type, backend="lm",
            maxlags=spec.cov_kwargs.get("maxlags"),
            groups=spec.cov_kwargs.get("groups"),
        )
        res = lm_model.fit(**cov_args)
    else:
        ModelClass = IVGMM
        model_type = "GMM"
        lm_model = ModelClass(
            y, exog, endog, instr,
            weight_type=weight_type,
            **wt_kwargs,
        )
        cov_args = resolve_cov_args(
            spec.cov_type, backend="lm",
            maxlags=spec.cov_kwargs.get("maxlags"),
            groups=spec.cov_kwargs.get("groups"),
        )
        res = lm_model.fit(iter_limit=iter_limit, **cov_args)

    return build_lm_gmm_result(res, model_type, spec.dep_var, spec.cov_type, spec.alpha, y)


def fit_panel_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit a panel model from a :class:`ModelSpec`.

    Supports ``'fd'`` (FirstDifference), ``'fe'`` (FixedEffects),
    ``'re'`` (RandomEffects), ``'between'`` (BetweenOLS),
    and ``'pooled'`` (PooledOLS).

    For FE models, ``spec.effects`` controls which effects are absorbed:
    ``'entity'`` (default), ``'time'``, or ``'both'``.
    """
    if spec.entity_col is None or spec.time_col is None:
        raise ValueError(
            "Panel models require ModelSpec.entity_col and ModelSpec.time_col."
        )

    cols = (
        [spec.dep_var]
        + list(spec.exog_vars)
        + [spec.entity_col, spec.time_col]
    )
    data = df[cols].dropna()
    panel = data.set_index([spec.entity_col, spec.time_col]).sort_index()

    y: pd.Series = panel[spec.dep_var]
    X: pd.DataFrame = panel[list(spec.exog_vars)]

    cov_resolved = resolve_cov_args(
        spec.cov_type,
        backend="lm",
        maxlags=spec.cov_kwargs.get("maxlags"),
        groups=spec.cov_kwargs.get("groups"),
    )
    cov_type_lm = cov_resolved.get("cov_type", "unadjusted")
    extra_fit_kwargs = {
        k: v for k, v in cov_resolved.items() if k != "cov_type"
    }

    estimator = spec.estimator.lower()

    if estimator in ("fd", "first_difference"):
        if spec.add_constant:
            X = X.assign(const=1.0)
        res = FirstDifferenceOLS(y, X).fit(cov_type=cov_type_lm, **extra_fit_kwargs)
        model_type = "FirstDifference"

    elif estimator == "fe":
        # PanelOLS demeans, so no constant should be added
        entity_effects = spec.effects in (None, "entity", "both")
        time_effects = spec.effects in ("time", "both")
        res = PanelOLS(
            y, X,
            entity_effects=entity_effects,
            time_effects=time_effects,
        ).fit(cov_type=cov_type_lm, **extra_fit_kwargs)
        model_type = "PanelFE"

        # Populate within/between R² if available
        from dataclasses import replace as _replace
        r2w = float(getattr(res, "rsquared_within", float("nan")))
        r2b = float(getattr(res, "rsquared_between", float("nan")))
        est = build_lm_panel_result(res, model_type, spec.dep_var, spec.cov_type, spec.alpha)
        fit_updated = _replace(est.fit, r_squared_within=r2w, r_squared_between=r2b)
        return _replace(est, fit=fit_updated)

    elif estimator == "re":
        if spec.add_constant:
            X = X.assign(const=1.0)
        res = RandomEffects(y, X).fit(cov_type=cov_type_lm, **extra_fit_kwargs)
        model_type = "RandomEffects"

    elif estimator == "between":
        if spec.add_constant:
            X = X.assign(const=1.0)
        res = BetweenOLS(y, X).fit(cov_type=cov_type_lm, **extra_fit_kwargs)
        model_type = "BetweenOLS"

    elif estimator == "pooled":
        if spec.add_constant:
            X = X.assign(const=1.0)
        res = PooledOLS(y, X).fit(cov_type=cov_type_lm, **extra_fit_kwargs)
        model_type = "PooledOLS"

    else:
        raise ValueError(
            f"Panel estimator '{spec.estimator}' is not supported. "
            "Choose from: 'fd', 'fe', 're', 'between', 'pooled'."
        )

    return build_lm_panel_result(res, model_type, spec.dep_var, spec.cov_type, spec.alpha)


# ---------------------------------------------------------------------------
# SUR (Seemingly Unrelated Regressions)
# ---------------------------------------------------------------------------

_SUR_COV_MAP: dict[str, str] = {
    "classical": "unadjusted",
    "unadjusted": "unadjusted",
    "homoskedastic": "unadjusted",
    "robust": "robust",
    "hc0": "robust", "hc1": "robust", "hc2": "robust", "hc3": "robust",
    "hac": "kernel",
    "newey_west": "kernel",
    "kernel": "kernel",
    "cluster": "clustered",
    "clustered": "clustered",
}


def fit_sur_from_spec(spec: SystemSpec, df: pd.DataFrame) -> Estimate:
    """Fit a SUR (Seemingly Unrelated Regressions) model from a :class:`SystemSpec`.

    Builds the per-equation ``{'dependent': y, 'exog': X}`` dict expected
    by :class:`linearmodels.system.SUR` and returns a single flat
    :class:`Estimate` with params/bse indexed by
    ``{equation_label}_{coef_name}``.
    """
    if not spec.equations:
        raise ValueError("SystemSpec.equations must be non-empty.")

    # Collect all referenced columns for a single dropna across equations
    all_cols: list[str] = []
    for eq in spec.equations.values():
        all_cols.append(eq.dep_var)
        all_cols.extend(eq.exog_vars)
        all_cols.extend(getattr(eq, "endog_vars", []) or [])
        all_cols.extend(getattr(eq, "instruments", []) or [])
    # Preserve order while deduplicating
    seen: set[str] = set()
    uniq_cols = [c for c in all_cols if not (c in seen or seen.add(c))]
    data = df[uniq_cols].dropna()

    equations_arg: dict[str, dict[str, pd.Series | pd.DataFrame]] = {}
    dep_vars: list[str] = []
    for label, eq in spec.equations.items():
        y = data[eq.dep_var]
        X = data[list(eq.exog_vars)].copy()
        if spec.add_constant:
            X = X.assign(const=1.0)
            # Put const first for readability
            X = X[["const"] + list(eq.exog_vars)]
        equations_arg[label] = {"dependent": y, "exog": X}
        dep_vars.append(eq.dep_var)

    # Resolve cov_type to linearmodels' system naming
    cov_key = spec.cov_type.lower()
    lm_cov_type = _SUR_COV_MAP.get(cov_key)
    if lm_cov_type is None:
        raise ValueError(
            f"cov_type='{spec.cov_type}' is not supported for SUR. "
            f"Choose from {sorted(set(_SUR_COV_MAP))}."
        )

    fit_kwargs: dict = {"cov_type": lm_cov_type}
    if spec.method is not None:
        fit_kwargs["method"] = spec.method
    if spec.iterate:
        fit_kwargs["iterate"] = True
        fit_kwargs["iter_limit"] = spec.iter_limit
    fit_kwargs["full_cov"] = spec.full_cov
    # Pass through cov-specific kwargs (bandwidth, kernel, clusters, ...)
    fit_kwargs.update(spec.cov_kwargs)

    estimator_key = spec.estimator.lower()
    if estimator_key != "sur":
        raise ValueError(
            f"fit_sur_from_spec only handles estimator='sur'; got '{spec.estimator}'."
        )

    from linearmodels.system import SUR
    res = SUR(equations_arg).fit(**fit_kwargs)

    return build_lm_system_result(
        res, model_type="SUR",
        dep_vars=dep_vars, cov_type_label=spec.cov_type, alpha=spec.alpha,
    )


# ---------------------------------------------------------------------------
# Arellano-Bond (difference GMM for dynamic panels)
# ---------------------------------------------------------------------------

def _build_ab_instruments(
    y_wide: pd.DataFrame,
    max_lags: int | None,
) -> pd.DataFrame:
    """Construct an AB-style lagged-level instrument matrix in long format.

    For each entity ``i`` and time ``t`` (post-differencing), the set of
    valid instruments for the differenced equation is
    ``{y_{i,1}, y_{i,2}, …, y_{i,t-2}}`` (deepest lag capped at
    ``max_lags`` when provided).

    Uses a "collapsed" layout: one column per instrument lag depth
    (``y_lag2``, ``y_lag3``, …).  Missing combinations (shallow ``t``
    with no deep-enough lag) are zero-filled, following the standard
    Holtz-Eakin/Newey/Rosen convention for unbalanced moment sets.
    """
    n_entities, n_periods = y_wide.shape
    if n_periods < 3:
        raise ValueError(
            "Arellano-Bond requires at least 3 time periods per entity."
        )

    # Maximum lag available for the deepest observation is (T-2)
    effective_max = n_periods - 2
    if max_lags is not None:
        effective_max = min(effective_max, int(max_lags))
    if effective_max < 2:
        raise ValueError(
            "max_lags must allow at least lag-2 instruments "
            f"(need ≥ 2, got {effective_max})."
        )

    # We produce observations for differenced equations t = 2, ..., T-1
    # (0-indexed), i.e. differences built from t vs t-1 with t >= 2 so
    # that y_{t-2} is available as instrument.
    records: list[dict] = []
    entities = list(y_wide.index)
    times = list(y_wide.columns)
    for i, ent in enumerate(entities):
        for t_idx in range(2, n_periods):
            row: dict = {
                "_entity": ent,
                "_time": times[t_idx],
            }
            # Instrument at lag ℓ = 2, 3, …, effective_max
            for lag in range(2, effective_max + 1):
                src_idx = t_idx - lag
                val = y_wide.iloc[i, src_idx] if src_idx >= 0 else 0.0
                if pd.isna(val):
                    val = 0.0
                row[f"y_lag{lag}"] = float(val)
            records.append(row)

    return pd.DataFrame.from_records(records)


def fit_arellano_bond_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit an Arellano-Bond difference-GMM dynamic panel model.

    Implementation outline
    ----------------------
    1. Build the balanced (entity, time) panel view of ``spec.dep_var``
       and ``spec.exog_vars``.
    2. First-difference everything.  The lagged dependent regressors
       (``y_{t-1}``, …, ``y_{t-L}``) become the endogenous variables of
       the differenced equation.  Any additional exogenous regressors in
       ``spec.exog_vars`` are first-differenced and treated as included
       exogenous.
    3. Build the AB instrument matrix: deeper lagged levels of
       ``y`` (lag 2, 3, …) serve as excluded instruments.
    4. Fit via :class:`linearmodels.iv.IVGMM` (two-step by default).
       The Hansen J statistic for over-identification populates
       ``fit.j_stat`` / ``fit.j_pvalue``.

    Parameters
    ----------
    spec:
        :class:`ModelSpec` with ``estimator='arellano_bond'``,
        ``entity_col`` and ``time_col`` set, and (optionally) ``lags``
        (default 1, giving standard AR(1) dynamic panel) and
        ``max_lags``.
    df:
        Long-format panel DataFrame.

    Returns
    -------
    Estimate
        Result with ``model_type='Arellano-Bond'`` and J-test populated.
    """
    from linearmodels.iv import IVGMM

    if spec.entity_col is None or spec.time_col is None:
        raise ValueError(
            "Arellano-Bond requires ModelSpec.entity_col and ModelSpec.time_col."
        )
    if spec.lags < 1:
        raise ValueError("ModelSpec.lags must be >= 1 for Arellano-Bond.")

    cols = [spec.dep_var, spec.entity_col, spec.time_col] + list(spec.exog_vars)
    panel = df[cols].dropna().sort_values([spec.entity_col, spec.time_col])

    # Wide pivot for the dependent variable: (entity, time) balanced panel
    y_wide = panel.pivot(
        index=spec.entity_col, columns=spec.time_col, values=spec.dep_var
    )
    time_points = list(y_wide.columns)
    n_periods = len(time_points)

    if n_periods < 3 + spec.lags - 1:
        raise ValueError(
            f"Arellano-Bond with lags={spec.lags} needs at least "
            f"{3 + spec.lags - 1} time periods; got {n_periods}."
        )

    # Exogenous regressors in wide form
    exog_wide: dict[str, pd.DataFrame] = {}
    for x_name in spec.exog_vars:
        exog_wide[x_name] = panel.pivot(
            index=spec.entity_col, columns=spec.time_col, values=x_name
        )

    # Differenced dependent variable: Δy_{i,t} for t = 1, ..., T-1 (0-indexed t>=1)
    dy = y_wide.diff(axis=1)

    # Build long-format arrays for differenced equation at t = lags+1, ..., T-1
    # (0-indexed), so that Δy_{t-L} = y_{t-L}-y_{t-L-1} is defined.
    # Instruments require t >= 2 (for lag-2 level), so we need t >= max(2, lags+1).
    t_start = max(2, spec.lags + 1)

    row_records: list[dict] = []
    entities = list(y_wide.index)
    for ent in entities:
        for t_idx in range(t_start, n_periods):
            rec: dict = {"_entity": ent, "_time": time_points[t_idx]}
            rec["dy"] = dy.loc[ent].iloc[t_idx]
            # Lagged differenced y: ℓ = 1, ..., spec.lags
            for lag in range(1, spec.lags + 1):
                rec[f"dy_lag{lag}"] = dy.loc[ent].iloc[t_idx - lag]
            # Differenced exogenous regressors
            for x_name, x_w in exog_wide.items():
                rec[f"d_{x_name}"] = x_w.loc[ent].iloc[t_idx] - x_w.loc[ent].iloc[t_idx - 1]
            row_records.append(rec)

    long_df = pd.DataFrame.from_records(row_records).dropna(
        subset=["dy"] + [f"dy_lag{lag}" for lag in range(1, spec.lags + 1)]
    ).reset_index(drop=True)

    # Instrument matrix from deeper levels of y
    instr_df = _build_ab_instruments(y_wide, spec.max_lags)
    # Align by (_entity, _time)
    merged = long_df.merge(
        instr_df, on=["_entity", "_time"], how="left"
    ).fillna(0.0)

    instr_cols = [c for c in instr_df.columns if c.startswith("y_lag")]
    endog_cols = [f"dy_lag{lag}" for lag in range(1, spec.lags + 1)]
    exog_cols = [f"d_{x}" for x in spec.exog_vars]

    y_vec = merged["dy"]
    endog = merged[endog_cols]
    exog = merged[exog_cols] if exog_cols else None
    if spec.add_constant and exog is None:
        exog = pd.DataFrame({"const": 1.0}, index=merged.index)
    elif spec.add_constant:
        exog = exog.assign(const=1.0)
    instr = merged[instr_cols]

    # GMM weight / cov settings
    weight_type: str = spec.cov_kwargs.get("weight_type", "robust")
    iter_limit: int = spec.cov_kwargs.get("iter_limit", 2)
    wt_kwargs: dict = {}
    if weight_type == "kernel":
        wt_kwargs["kernel"] = spec.cov_kwargs.get("kernel", "bartlett")
        if "bandwidth" in spec.cov_kwargs:
            wt_kwargs["bandwidth"] = spec.cov_kwargs["bandwidth"]

    lm_model = IVGMM(y_vec, exog, endog, instr, weight_type=weight_type, **wt_kwargs)

    # cov_type for AB defaults to robust (Windmeijer-unaware two-step);
    # users can request classical via spec.cov_type
    cov_args = resolve_cov_args(
        spec.cov_type, backend="lm",
        maxlags=spec.cov_kwargs.get("maxlags"),
        groups=spec.cov_kwargs.get("groups"),
    )
    res = lm_model.fit(iter_limit=iter_limit, **cov_args)

    est = build_lm_gmm_result(
        res, model_type="Arellano-Bond",
        dep_var=spec.dep_var, cov_type_label=spec.cov_type, alpha=spec.alpha,
        y=y_vec,
    )
    return est
