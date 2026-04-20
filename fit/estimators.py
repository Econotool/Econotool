"""Central dispatch for econtools estimation.

The single public entry point is :func:`fit_model`, which takes a fully
specified :class:`~econtools.model.spec.ModelSpec` (single-equation) or a
:class:`~econtools.model.spec.SystemSpec` (multi-equation system) along
with a :class:`pandas.DataFrame` and returns an
:class:`~econtools._core.types.Estimate`.

Public API
----------
fit_model(spec, df) -> Estimate
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from econtools._core.types import Estimate
from econtools.model.spec import ModelSpec, SystemSpec


def fit_model(spec: Union[ModelSpec, SystemSpec], df: pd.DataFrame) -> Estimate:
    """Fit a model defined by *spec* on *df* and return an :class:`Estimate`.

    Parameters
    ----------
    spec:
        A :class:`~econtools.model.spec.ModelSpec` for single-equation
        models or a :class:`~econtools.model.spec.SystemSpec` for
        multi-equation system estimators (SUR).
    df:
        Input data.

    Returns
    -------
    Estimate
        Normalised result object; access ``.raw`` for library-specific
        attributes.

    Raises
    ------
    ValueError
        If ``spec.estimator`` is not supported.
    """
    # ----------------------------------------------------------------- #
    # Multi-equation system dispatch                                    #
    # ----------------------------------------------------------------- #
    if isinstance(spec, SystemSpec):
        estimator = spec.estimator.lower()
        if estimator == "sur":
            from econtools.fit._lm_adapter import fit_sur_from_spec
            return fit_sur_from_spec(spec, df)
        raise ValueError(
            f"Unknown system estimator '{spec.estimator}'. Supported: 'sur'."
        )

    # ----------------------------------------------------------------- #
    # Single-equation dispatch                                          #
    # ----------------------------------------------------------------- #
    estimator = spec.estimator.lower()

    if estimator in ("ols", "wls"):
        from econtools.fit._sm_adapter import fit_ols_from_spec
        return fit_ols_from_spec(spec, df)

    if estimator in ("2sls", "iv2sls", "liml"):
        from econtools.fit._lm_adapter import fit_iv_from_spec
        return fit_iv_from_spec(spec, df)

    if estimator in ("gmm", "ivgmm", "cue_gmm", "ivgmmcue"):
        from econtools.fit._lm_adapter import fit_gmm_from_spec
        return fit_gmm_from_spec(spec, df)

    if estimator in ("arellano_bond", "ab_gmm", "dynamic_panel"):
        from econtools.fit._lm_adapter import fit_arellano_bond_from_spec
        return fit_arellano_bond_from_spec(spec, df)

    if estimator in ("fe", "re", "fd", "first_difference", "pooled", "between"):
        from econtools.fit._lm_adapter import fit_panel_from_spec
        return fit_panel_from_spec(spec, df)

    if estimator == "probit":
        from econtools.fit._sm_adapter import fit_probit_from_spec
        return fit_probit_from_spec(spec, df)

    if estimator == "logit":
        from econtools.fit._sm_adapter import fit_logit_from_spec
        return fit_logit_from_spec(spec, df)

    raise ValueError(
        f"Unknown estimator '{spec.estimator}'. "
        "Choose from: 'ols', 'wls', '2sls', 'liml', 'fe', 're', 'fd', "
        "'between', 'pooled', 'probit', 'logit', 'gmm', 'cue_gmm', "
        "'arellano_bond'."
    )
