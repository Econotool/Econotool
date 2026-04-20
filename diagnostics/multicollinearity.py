"""Multicollinearity diagnostics: VIF and condition number.

Public API
----------
compute_vif(result)      -> pd.Series
condition_number(result) -> float
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from econtools.models._results import RegressionResult

# Column names that indicate a constant term — excluded from VIF
_CONSTANT_NAMES: frozenset[str] = frozenset({"const", "Intercept"})


def compute_vif(result: RegressionResult) -> pd.Series:
    """Compute Variance Inflation Factors for each regressor.

    The constant column (named ``'const'`` or ``'Intercept'``) and any
    column with zero variance are excluded.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    pd.Series indexed by regressor name, values = VIF.
    """
    exog: np.ndarray = result.raw.model.exog
    names: list[str] = list(result.raw.model.exog_names)

    vif_dict: dict[str, float] = {}
    for i, name in enumerate(names):
        # Skip constant columns
        if name in _CONSTANT_NAMES:
            continue
        # Skip zero-variance columns (belt-and-suspenders)
        if float(np.ptp(exog[:, i])) == 0.0:
            continue
        vif_dict[name] = float(variance_inflation_factor(exog, i))

    return pd.Series(vif_dict, name="VIF")


def condition_number(result: RegressionResult) -> float:
    """Return the condition number of the design matrix.

    A large condition number (> 30) suggests potential multicollinearity.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.

    Returns
    -------
    float
    """
    return float(result.raw.condition_number)
