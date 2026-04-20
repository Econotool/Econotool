"""Oaxaca-Blinder decomposition for wage/outcome gaps.

Implements the threefold (Blinder 1973, Oaxaca 1973) and twofold
decompositions, widely used in labor economics to decompose mean
outcome differences between groups into explained (endowments),
unexplained (coefficients), and interaction components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class OaxacaBlinderResult:
    """Result of an Oaxaca-Blinder decomposition.

    Attributes
    ----------
    gap : float
        Raw mean outcome difference (group A - group B).
    endowments : float
        Explained component (differences in characteristics).
    coefficients : float
        Unexplained component (differences in returns).
    interaction : float
        Interaction term (threefold only; zero for twofold).
    endowments_se : float
        Standard error of the endowments component.
    coefficients_se : float
        Standard error of the coefficients component.
    interaction_se : float
        Standard error of the interaction component.
    endowments_detail : pd.Series
        Variable-by-variable endowment contributions.
    coefficients_detail : pd.Series
        Variable-by-variable coefficient contributions.
    interaction_detail : pd.Series
        Variable-by-variable interaction contributions.
    nobs_a : int
        Observations in group A.
    nobs_b : int
        Observations in group B.
    params_a : pd.Series
        OLS coefficients for group A.
    params_b : pd.Series
        OLS coefficients for group B.
    details : dict
        Additional info (means, R², etc.).
    """

    gap: float
    endowments: float
    coefficients: float
    interaction: float
    endowments_se: float
    coefficients_se: float
    interaction_se: float
    endowments_detail: pd.Series
    coefficients_detail: pd.Series
    interaction_detail: pd.Series
    nobs_a: int
    nobs_b: int
    params_a: pd.Series
    params_b: pd.Series
    details: dict = field(default_factory=dict)


def oaxaca_blinder(
    df: pd.DataFrame,
    y: str,
    x_vars: list[str],
    group_var: str,
    reference: int = 0,
    method: str = "threefold",
) -> OaxacaBlinderResult:
    """Oaxaca-Blinder decomposition.

    Parameters
    ----------
    df : DataFrame
    y : str
        Outcome variable (e.g. log wage).
    x_vars : list of str
        Explanatory variables.
    group_var : str
        Binary group indicator (values 0 and 1).
    reference : int
        Reference group for twofold decomposition (0 or 1).
        Ignored for threefold.
    method : str
        ``'threefold'`` (default) or ``'twofold'``.

    Returns
    -------
    OaxacaBlinderResult

    Notes
    -----
    **Threefold** (Blinder 1973, Oaxaca 1973):

        Δȳ = (X̄_A - X̄_B)'β̂_B  +  X̄_B'(β̂_A - β̂_B)  +  (X̄_A - X̄_B)'(β̂_A - β̂_B)
              ─────────────────    ─────────────────────    ───────────────────────────
              endowments           coefficients              interaction

    **Twofold** (with reference group B):

        Δȳ = (X̄_A - X̄_B)'β̂_ref  +  [X̄_A'(β̂_A - β̂_ref) + X̄_B'(β̂_ref - β̂_B)]
    """
    if method not in ("threefold", "twofold"):
        raise ValueError(f"method must be 'threefold' or 'twofold', got '{method}'.")

    data = df.copy()
    groups = data[group_var].unique()
    if len(groups) != 2 or set(groups) != {0, 1}:
        raise ValueError(
            f"group_var must have exactly two values (0 and 1), got {sorted(groups)}."
        )

    # Split and estimate
    df_a = data[data[group_var] == 1]
    df_b = data[data[group_var] == 0]

    X_a = sm.add_constant(df_a[x_vars].astype(float))
    X_b = sm.add_constant(df_b[x_vars].astype(float))
    Y_a = df_a[y].astype(float)
    Y_b = df_b[y].astype(float)

    res_a = sm.OLS(Y_a, X_a).fit()
    res_b = sm.OLS(Y_b, X_b).fit()

    beta_a = res_a.params
    beta_b = res_b.params
    Xbar_a = X_a.mean()
    Xbar_b = X_b.mean()
    dX = Xbar_a - Xbar_b
    dbeta = beta_a - beta_b

    gap = float(Y_a.mean() - Y_b.mean())

    if method == "threefold":
        endow_detail = dX * beta_b
        coef_detail = Xbar_b * dbeta
        inter_detail = dX * dbeta
    else:
        # Twofold with reference group
        if reference == 0:
            beta_ref = beta_b
        else:
            beta_ref = beta_a
        endow_detail = dX * beta_ref
        coef_detail = Xbar_a * (beta_a - beta_ref) + Xbar_b * (beta_ref - beta_b)
        inter_detail = pd.Series(0.0, index=beta_a.index)

    endowments = float(endow_detail.sum())
    coefficients = float(coef_detail.sum())
    interaction = float(inter_detail.sum())

    # Standard errors via delta method (bootstrap is better but this is fast)
    # Variance of endowments: dX' Var(beta_b) dX
    V_b = res_b.cov_params()
    V_a = res_a.cov_params()
    n_a = len(df_a)
    n_b = len(df_b)

    endow_var = float(dX @ V_b @ dX) + float(np.sum(beta_b.values ** 2 * (
        X_a.var() / n_a + X_b.var() / n_b
    ).values))
    coef_var = float(Xbar_b @ (V_a + V_b) @ Xbar_b)
    inter_var = float(dX @ (V_a + V_b) @ dX)

    endow_se = float(np.sqrt(max(endow_var, 0)))
    coef_se = float(np.sqrt(max(coef_var, 0)))
    inter_se = float(np.sqrt(max(inter_var, 0)))

    return OaxacaBlinderResult(
        gap=gap,
        endowments=endowments,
        coefficients=coefficients,
        interaction=interaction,
        endowments_se=endow_se,
        coefficients_se=coef_se,
        interaction_se=inter_se,
        endowments_detail=endow_detail,
        coefficients_detail=coef_detail,
        interaction_detail=inter_detail,
        nobs_a=n_a,
        nobs_b=n_b,
        params_a=beta_a,
        params_b=beta_b,
        details={
            "mean_y_a": float(Y_a.mean()),
            "mean_y_b": float(Y_b.mean()),
            "r_squared_a": float(res_a.rsquared),
            "r_squared_b": float(res_b.rsquared),
            "method": method,
            "reference": reference,
        },
    )
