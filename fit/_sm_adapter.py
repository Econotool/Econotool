"""Statsmodels backend for the fit/ layer.

All functions take a :class:`~econtools.model.spec.ModelSpec` and a
:class:`pandas.DataFrame`, call the appropriate statsmodels estimator, and
return an :class:`~econtools._core.types.Estimate`.

Internal API.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from econtools._core.cov_mapping import resolve_cov_args
from econtools._core.types import Estimate
from econtools.fit._builders import build_binary_result, build_sm_result
from econtools.model.spec import ModelSpec


def fit_ols_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit OLS or WLS from a :class:`ModelSpec`."""
    cols = [spec.dep_var] + list(spec.exog_vars)
    if spec.weights_col:
        cols.append(spec.weights_col)
    data = df[cols].dropna()

    y: pd.Series = data[spec.dep_var]
    X: pd.DataFrame = data[list(spec.exog_vars)]
    if spec.add_constant:
        X = sm.add_constant(X)

    cov_args = resolve_cov_args(
        spec.cov_type,
        backend="sm",
        maxlags=spec.cov_kwargs.get("maxlags"),
        groups=spec.cov_kwargs.get("groups"),
    )

    if spec.weights_col and spec.estimator == "wls":
        w: pd.Series = data[spec.weights_col]
        sm_result = sm.WLS(y, X, weights=w).fit(**cov_args)
        model_type = "WLS"
    else:
        sm_result = sm.OLS(y, X).fit(**cov_args)
        model_type = "OLS"

    return build_sm_result(sm_result, model_type, spec.dep_var, spec.cov_type, spec.alpha)


def fit_probit_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit Probit from a :class:`ModelSpec`."""
    return _fit_binary_from_spec(sm.Probit, spec, df)


def fit_logit_from_spec(spec: ModelSpec, df: pd.DataFrame) -> Estimate:
    """Fit Logit from a :class:`ModelSpec`."""
    return _fit_binary_from_spec(sm.Logit, spec, df)


def _fit_binary_from_spec(
    model_class: type,
    spec: ModelSpec,
    df: pd.DataFrame,
) -> Estimate:
    """Shared implementation for Probit and Logit."""
    cols = [spec.dep_var] + list(spec.exog_vars)
    data = df[cols].dropna()

    y: pd.Series = data[spec.dep_var]
    X: pd.DataFrame = data[list(spec.exog_vars)]
    if spec.add_constant:
        X = sm.add_constant(X)

    if spec.cov_type != "classical":
        cov_args = resolve_cov_args(
            spec.cov_type,
            backend="sm",
            maxlags=spec.cov_kwargs.get("maxlags"),
            groups=spec.cov_kwargs.get("groups"),
        )
        cov_type_sm = cov_args.get("cov_type", "nonrobust")
        cov_kwds = cov_args.get("cov_kwds", {})
        sm_result = model_class(y, X).fit(disp=False)
        # Probit has get_robustcov_results; Logit does not (statsmodels quirk).
        # Fall back to re-fitting with cov_type/cov_kwds.
        if hasattr(sm_result, "get_robustcov_results"):
            sm_result = sm_result.get_robustcov_results(
                cov_type=cov_type_sm, **cov_kwds
            )
        else:
            sm_result = model_class(y, X).fit(
                disp=False, cov_type=cov_type_sm, cov_kwds=cov_kwds
            )
    else:
        sm_result = model_class(y, X).fit(disp=False)

    return build_binary_result(sm_result, spec.dep_var, spec.cov_type, spec.alpha)
