"""Marginal effects for binary choice models (Probit / Logit).

Public API
----------
marginal_effects(result, at='overall') -> MarginalEffectsResult
marginal_effects_at_values(result, values) -> MarginalEffectsResult

MarginalEffectsResult — dataclass with effects, standard errors, p-values,
    confidence intervals, and summary DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from econtools._core.types import Estimate


@dataclass(frozen=True)
class MarginalEffectsResult:
    """Container for marginal effects output.

    Attributes
    ----------
    effects : pd.Series
        Marginal effect (dy/dx) for each regressor.
    std_errors : pd.Series
        Standard errors of marginal effects.
    z_values : pd.Series
        z-statistics (effect / SE).
    pvalues : pd.Series
        Two-sided p-values.
    conf_int_lower : pd.Series
        Lower bound of confidence interval.
    conf_int_upper : pd.Series
        Upper bound of confidence interval.
    at : str
        Where effects were evaluated ('overall', 'mean', or 'values').
    model_type : str
        'probit' or 'logit'.
    n_obs : int
        Number of observations.
    """

    effects: pd.Series
    std_errors: pd.Series
    z_values: pd.Series
    pvalues: pd.Series
    conf_int_lower: pd.Series
    conf_int_upper: pd.Series
    at: str
    model_type: str
    n_obs: int

    def summary(self) -> pd.DataFrame:
        """Summary table of marginal effects."""
        return pd.DataFrame(
            {
                "dy/dx": self.effects,
                "Std. Err.": self.std_errors,
                "z": self.z_values,
                "P>|z|": self.pvalues,
                "[0.025": self.conf_int_lower,
                "0.975]": self.conf_int_upper,
            }
        )


def marginal_effects(
    result: Estimate,
    *,
    at: str = "overall",
    alpha: float = 0.05,
) -> MarginalEffectsResult:
    """Compute marginal effects for a Probit or Logit model.

    Parameters
    ----------
    result:
        Fitted :class:`Estimate` from ``fit_model`` with estimator
        ``'probit'`` or ``'logit'``.
    at:
        Where to evaluate:

        - ``'overall'`` (default) — Average Marginal Effect (AME): compute
          marginal effect at every observation, then average.
        - ``'mean'`` — Marginal Effect at the Mean (MEM): evaluate at the
          sample mean of all regressors.
    alpha:
        Significance level for confidence intervals (default 0.05).

    Returns
    -------
    MarginalEffectsResult
    """
    _validate_binary_model(result)

    me = result.raw.get_margeff(at=at, dummy=True)
    return _build_result(me, result, at=at, alpha=alpha)


def marginal_effects_at_values(
    result: Estimate,
    values: Dict[str, float],
    *,
    alpha: float = 0.05,
) -> MarginalEffectsResult:
    """Compute marginal effects at specific covariate values.

    Parameters
    ----------
    result:
        Fitted :class:`Estimate` from ``fit_model`` with estimator
        ``'probit'`` or ``'logit'``.
    values:
        Dict mapping regressor names to the values at which to evaluate.
        All regressors in the model must be specified (excluding the constant).
    alpha:
        Significance level for confidence intervals (default 0.05).

    Returns
    -------
    MarginalEffectsResult
    """
    _validate_binary_model(result)

    exog_names = [str(n) for n in result.raw.model.exog_names]
    # Build the evaluation point as a dict keyed by column index
    # (statsmodels discrete_margins expects {col_index: value} for atexog)
    exog_non_const = [n for n in exog_names if n != "const"]
    for name in exog_non_const:
        if name not in values:
            raise ValueError(
                f"Missing value for regressor '{name}'. "
                f"All regressors must be specified: {exog_non_const}"
            )

    # Build a full exog row with the given values
    at_row = np.zeros(len(exog_names))
    for i, name in enumerate(exog_names):
        if name == "const":
            at_row[i] = 1.0
        else:
            at_row[i] = values[name]

    # statsmodels get_margeff(at='overall') then override exog_mean
    # The cleanest approach: use 'mean' and pass atexog dict
    atexog = {i: values[name] for i, name in enumerate(exog_names) if name != "const"}
    me = result.raw.get_margeff(at="mean", dummy=True, atexog=atexog)
    return _build_result(me, result, at="values", alpha=alpha)


def _validate_binary_model(result: Estimate) -> None:
    """Ensure the result comes from a Probit or Logit model."""
    model_type = getattr(result, "model_type", "")
    raw = result.raw
    class_name = type(raw).__name__.lower() if raw is not None else ""

    is_binary = (
        model_type in ("probit", "logit")
        or "probit" in class_name
        or "logit" in class_name
    )
    if not is_binary:
        raise TypeError(
            f"Marginal effects require a Probit or Logit model. "
            f"Got model_type='{model_type}'."
        )


def _build_result(
    me,
    result: Estimate,
    *,
    at: str,
    alpha: float,
) -> MarginalEffectsResult:
    """Convert statsmodels MargEffResults to our dataclass."""
    frame = me.summary_frame(alpha=alpha)

    effects = pd.Series(
        me.margeff, index=frame.index, name="dy/dx", dtype=float
    )
    std_errors = pd.Series(
        me.margeff_se, index=frame.index, name="std_err", dtype=float
    )
    z_values = effects / std_errors
    z_values.name = "z"
    pvalues = pd.Series(
        me.pvalues, index=frame.index, name="pvalue", dtype=float
    )
    ci = me.conf_int(alpha=alpha)
    ci_lower = pd.Series(ci[:, 0], index=frame.index, name="ci_lower")
    ci_upper = pd.Series(ci[:, 1], index=frame.index, name="ci_upper")

    model_type = getattr(result, "model_type", "unknown")
    if model_type == "unknown":
        class_name = type(result.raw).__name__.lower()
        if "probit" in class_name:
            model_type = "probit"
        elif "logit" in class_name:
            model_type = "logit"

    return MarginalEffectsResult(
        effects=effects,
        std_errors=std_errors,
        z_values=z_values,
        pvalues=pvalues,
        conf_int_lower=ci_lower,
        conf_int_upper=ci_upper,
        at=at,
        model_type=model_type,
        n_obs=int(result.raw.nobs),
    )
