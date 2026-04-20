"""Consolidated result builders for all backends.

Four thin functions replace the four duplicate ``_build_result`` copies
scattered across ``models/ols.py``, ``models/iv.py``, ``models/panel.py``,
and ``models/probit.py``.

Internal API — not part of the public surface.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from econtools._core.types import Estimate, FitMetrics


def build_sm_result(
    sm_result: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
    *,
    formula: str | None = None,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted statsmodels OLS/WLS result."""
    ci = sm_result.conf_int(alpha=alpha)

    fvalue = sm_result.fvalue
    f_pvalue = sm_result.f_pvalue
    f_stat = float(np.squeeze(fvalue)) if fvalue is not None else float("nan")
    f_pval = float(np.squeeze(f_pvalue)) if f_pvalue is not None else float("nan")

    fit = FitMetrics(
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
        r_squared=float(sm_result.rsquared),
        r_squared_adj=float(sm_result.rsquared_adj),
        aic=float(sm_result.aic),
        bic=float(sm_result.bic),
        log_likelihood=float(sm_result.llf),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=float(np.sqrt(sm_result.mse_resid)),
        ssr=float(sm_result.ssr),
        ess=float(sm_result.ess),
        tss=float(sm_result.centered_tss),
    )

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.bse,
        tvalues=sm_result.tvalues,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=sm_result.resid,
        fitted=sm_result.fittedvalues,
        cov_params=sm_result.cov_params(),
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=formula,
    )


def build_lm_iv_result(
    sm_result: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
    y: pd.Series,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted linearmodels IV result."""
    try:
        ci = sm_result.conf_int(level=1 - alpha)
    except Exception:
        ci = sm_result.conf_int()

    f_stat = float("nan")
    f_pval = float("nan")
    f_obj = getattr(sm_result, "f_statistic", None)
    if f_obj is not None:
        f_stat = float(getattr(f_obj, "stat", f_stat))
        f_pval = float(getattr(f_obj, "pval", f_pval))

    resid = sm_result.resids
    ssr = float(np.sum(np.asarray(resid) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    ess = float(tss - ssr)
    rmse = math.sqrt(ssr / len(y))

    fit = FitMetrics(
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
        r_squared=float(sm_result.rsquared),
        r_squared_adj=float(sm_result.rsquared_adj),
        aic=float(getattr(sm_result, "aic", float("nan"))),
        bic=float(getattr(sm_result, "bic", float("nan"))),
        log_likelihood=float(getattr(sm_result, "loglik", float("nan"))),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=rmse,
        ssr=ssr,
        ess=ess,
        tss=tss,
    )

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.std_errors,
        tvalues=sm_result.tstats,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=sm_result.resids,
        fitted=sm_result.fitted_values,
        cov_params=sm_result.cov,
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=None,
    )


def build_lm_panel_result(
    res: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted linearmodels panel result."""
    ci = res.conf_int(level=1 - alpha)

    f_stat = float("nan")
    f_pval = float("nan")
    f_obj = getattr(res, "f_statistic", None)
    if f_obj is not None:
        f_stat = float(getattr(f_obj, "stat", f_stat))
        f_pval = float(getattr(f_obj, "pval", f_pval))

    resid = res.resids
    ssr = float(np.sum(np.asarray(resid) ** 2))
    y = np.asarray(res.model.dependent.values2d).squeeze()
    tss = float(np.sum((y - y.mean()) ** 2))
    ess = float(tss - ssr)
    rmse = math.sqrt(ssr / len(y))

    # Some panel models (BetweenOLS) return tuples for df_model/df_resid
    df_model_raw = res.df_model
    df_resid_raw = res.df_resid
    if isinstance(df_model_raw, tuple):
        df_model_raw = df_model_raw[0]
    if isinstance(df_resid_raw, tuple):
        df_resid_raw = df_resid_raw[0]

    fit = FitMetrics(
        nobs=int(res.nobs),
        df_model=float(df_model_raw),
        df_resid=float(df_resid_raw),
        r_squared=float(res.rsquared),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=rmse,
        ssr=ssr,
        ess=ess,
        tss=tss,
    )

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=res.params,
        bse=res.std_errors,
        tvalues=res.tstats,
        pvalues=res.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=res.resids,
        fitted=res.fitted_values,
        cov_params=res.cov,
        cov_type=cov_type_label,
        fit=fit,
        raw=res,
        formula=None,
    )


def build_lm_gmm_result(
    res: object,
    model_type: str,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
    y: pd.Series,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted linearmodels IVGMM/IVGMMCUE result."""
    try:
        ci = res.conf_int(level=1 - alpha)
    except Exception:
        ci = res.conf_int()

    f_stat = float("nan")
    f_pval = float("nan")
    f_obj = getattr(res, "f_statistic", None)
    if f_obj is not None:
        f_stat = float(getattr(f_obj, "stat", f_stat))
        f_pval = float(getattr(f_obj, "pval", f_pval))

    resid = res.resids
    ssr = float(np.sum(np.asarray(resid) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    ess = float(tss - ssr)
    rmse = math.sqrt(ssr / len(y))

    # GMM-specific: J-test statistic
    j_stat = float("nan")
    j_pvalue = float("nan")
    j_obj = getattr(res, "j_stat", None)
    if j_obj is not None:
        j_stat = float(getattr(j_obj, "stat", float("nan")))
        j_pvalue = float(getattr(j_obj, "pval", float("nan")))

    fit = FitMetrics(
        nobs=int(res.nobs),
        df_model=float(res.df_model),
        df_resid=float(res.df_resid),
        r_squared=float(res.rsquared),
        r_squared_adj=float(getattr(res, "rsquared_adj", float("nan"))),
        aic=float(getattr(res, "aic", float("nan"))),
        bic=float(getattr(res, "bic", float("nan"))),
        f_stat=f_stat,
        f_pvalue=f_pval,
        rmse=rmse,
        ssr=ssr,
        ess=ess,
        tss=tss,
        j_stat=j_stat,
        j_pvalue=j_pvalue,
    )

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=res.params,
        bse=res.std_errors,
        tvalues=res.tstats,
        pvalues=res.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=res.resids,
        fitted=res.fitted_values,
        cov_params=res.cov,
        cov_type=cov_type_label,
        fit=fit,
        raw=res,
        formula=None,
    )


def build_lm_system_result(
    res: object,
    model_type: str,
    dep_vars: list[str],
    cov_type_label: str,
    alpha: float,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted linearmodels SUR/3SLS result.

    The flat params vector is indexed by ``{equation_label}_{coef_name}``
    (linearmodels' native layout), so the result composes directly with
    :class:`econtools.output.tables.pub_latex.ResultsTable`.

    Residuals and fitted values are long-format concatenations across
    equations: each equation's ``nobs`` rows stacked in equation-label
    order.  Callers needing per-equation access can retrieve the raw
    :class:`linearmodels.system.results.SystemResults` via ``.raw`` and
    then ``raw.equations[label]``.
    """
    try:
        ci = res.conf_int(level=1 - alpha)
    except TypeError:
        ci = res.conf_int()

    # SUR reports nobs as (sum over equations) or per-equation depending
    # on version; total is the sum of per-equation nobs.
    eq_map = res.equations
    per_eq_nobs = [int(eq_map[k].nobs) for k in eq_map]
    total_nobs = int(sum(per_eq_nobs))

    # df_model is the total number of free parameters across equations
    try:
        df_model_total = float(sum(float(eq_map[k].df_model) for k in eq_map))
        df_resid_total = float(sum(float(eq_map[k].df_resid) for k in eq_map))
    except (AttributeError, TypeError):
        df_model_total = float(len(res.params))
        df_resid_total = float(total_nobs - df_model_total)

    # Residuals / fitted — linearmodels returns a DataFrame with one
    # column per equation (wide format).  Stack to a long Series so the
    # Estimate contract holds.
    resids_wide = res.resids
    fitted_wide = res.fitted_values
    if hasattr(resids_wide, "columns"):
        resid_long = pd.concat(
            [resids_wide[c].rename("resid") for c in resids_wide.columns],
            keys=list(resids_wide.columns),
            names=["equation", "obs"],
        )
        fitted_long = pd.concat(
            [fitted_wide[c].rename("fitted") for c in fitted_wide.columns],
            keys=list(fitted_wide.columns),
            names=["equation", "obs"],
        )
    else:
        resid_long = pd.Series(np.asarray(resids_wide).ravel(), name="resid")
        fitted_long = pd.Series(np.asarray(fitted_wide).ravel(), name="fitted")

    # Pooled R² across equations (weighted by nobs); not a canonical
    # scalar for SUR — consumers should consult per-equation rsquared on
    # ``raw.equations[label]``.
    r2_pooled = float("nan")
    try:
        per_eq_r2 = [float(eq_map[k].rsquared) for k in eq_map]
        weights = [n / total_nobs for n in per_eq_nobs]
        r2_pooled = float(sum(r2 * w for r2, w in zip(per_eq_r2, weights)))
    except (AttributeError, TypeError, ZeroDivisionError):
        pass

    ssr_total = float(np.sum(np.asarray(resid_long) ** 2))
    rmse_total = math.sqrt(ssr_total / total_nobs) if total_nobs else float("nan")

    fit = FitMetrics(
        nobs=total_nobs,
        df_model=df_model_total,
        df_resid=df_resid_total,
        r_squared=r2_pooled,
        rmse=rmse_total,
        ssr=ssr_total,
    )

    # Use the concatenated label "eq1+eq2+..." as the dep_var slot so
    # ResultsTable still has a scalar string to display.
    dep_var_label = " + ".join(dep_vars)

    return Estimate(
        model_type=model_type,
        dep_var=dep_var_label,
        params=res.params,
        bse=res.std_errors,
        tvalues=res.tstats,
        pvalues=res.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=resid_long,
        fitted=fitted_long,
        cov_params=res.cov,
        cov_type=cov_type_label,
        fit=fit,
        raw=res,
        formula=None,
    )


def build_binary_result(
    sm_result: object,
    dep_var: str,
    cov_type_label: str,
    alpha: float,
) -> Estimate:
    """Build an :class:`Estimate` from a fitted statsmodels Probit/Logit result."""
    ci = sm_result.conf_int(alpha=alpha)

    resid = getattr(sm_result, "resid_response", None)
    if resid is None:
        resid = getattr(sm_result, "resid", None)
    if resid is None:
        resid = pd.Series(dtype=float)

    fitted = sm_result.predict()

    fit = FitMetrics(
        nobs=int(sm_result.nobs),
        df_model=float(sm_result.df_model),
        df_resid=float(sm_result.df_resid),
        r_squared=float(getattr(sm_result, "prsquared", math.nan)),
        aic=float(sm_result.aic),
        bic=float(sm_result.bic),
        log_likelihood=float(sm_result.llf),
        pseudo_r_squared=float(getattr(sm_result, "prsquared", math.nan)),
    )

    # model_type is inferred from the class name
    class_name = type(sm_result.model).__name__.lower()
    model_type = "Logit (LDP)" if "logit" in class_name else "Probit (LDP)"

    return Estimate(
        model_type=model_type,
        dep_var=dep_var,
        params=sm_result.params,
        bse=sm_result.bse,
        tvalues=sm_result.tvalues,
        pvalues=sm_result.pvalues,
        conf_int_lower=ci.iloc[:, 0],
        conf_int_upper=ci.iloc[:, 1],
        resid=resid,
        fitted=fitted,
        cov_params=sm_result.cov_params(),
        cov_type=cov_type_label,
        fit=fit,
        raw=sm_result,
        formula=None,
    )
