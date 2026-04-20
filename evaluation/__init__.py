"""Evaluation — diagnostics, hypothesis tests, binary metrics, marginal effects.

Phase 1+ content.  This package collects all statistical evaluation
logic: assumption tests, hypothesis tests, influence diagnostics, and
binary-model metrics.  Rendering lives in ``econtools.output``.
"""

from econtools.evaluation.binary_metrics import (  # noqa: F401
    _BinaryMetrics,
    _binary_metrics,
    _marginal_effects,
)
from econtools.evaluation.heteroskedasticity import (  # noqa: F401
    breusch_pagan,
    goldfeld_quandt,
    white_test,
)
from econtools.evaluation.hypothesis import (  # noqa: F401
    TestResult,
    wald_test,
    f_test,
    t_test_coeff,
    lr_test,
    score_test,
    conf_int,
)
from econtools.evaluation.marginal_effects import (  # noqa: F401
    MarginalEffectsResult,
    marginal_effects,
    marginal_effects_at_values,
)
from econtools.evaluation.multicollinearity import compute_vif, condition_number  # noqa: F401
from econtools.evaluation.normality import (  # noqa: F401
    jarque_bera,
    omnibus_test,
    shapiro_wilk,
)
from econtools.evaluation.serial_correlation import (  # noqa: F401
    autocorr_from_series,
    box_pierce_from_autocorr,
    box_pierce_q,
    breusch_godfrey,
    durbin_watson,
    ljung_box_from_autocorr,
    ljung_box_q,
)
from econtools.evaluation.specification import (  # noqa: F401
    chow_test,
    harvey_collier,
    reset_test,
)
from econtools.evaluation.stationarity import adf_test, kpss_test, pp_test  # noqa: F401
from econtools.evaluation.time_series import (  # noqa: F401
    granger_causality,
    lead_exogeneity_test,
    select_var_lag,
)
from econtools.evaluation.iv_checks import (  # noqa: F401
    basmann_f_test,
    basmann_test,
    run_iv_diagnostics,
    sargan_test,
    weak_instrument_tests,
    wu_hausman_test,
)
from econtools.evaluation.panel_checks import (  # noqa: F401
    breusch_pagan_lm_test,
    hausman_fe_re,
    lead_test_strict_exogeneity,
    run_panel_diagnostics,
)
from econtools.evaluation.structural_breaks import (  # noqa: F401
    cusum_test,
    cusum_sq_test,
    andrews_sup_wald,
    chow_forecast_test,
    recursive_residuals,
)
from econtools.evaluation.did import (  # noqa: F401
    DIDResult,
    EventStudyResult,
    did_estimate,
    event_study,
    parallel_trends_test,
)
from econtools.evaluation.event_study_car import (  # noqa: F401
    EventStudyCARResult,
    MarketModel,
    bmp_car_test,
    compute_abnormal_returns,
    corrado_rank_test,
    cowan_sign_test,
    cumulative_abnormal_returns,
    estimate_market_model,
    event_study_car,
    patell_car_test,
)
from econtools.evaluation.decomposition import (  # noqa: F401
    OaxacaBlinderResult,
    oaxaca_blinder,
)
from econtools.evaluation.matching import (  # noqa: F401
    PropensityResult,
    MatchResult,
    IPWResult,
    DoublyRobustResult,
    propensity_score,
    nearest_neighbor_match,
    ipw_estimate,
    doubly_robust,
    covariate_balance,
)

__all__ = [
    # binary
    "_BinaryMetrics",
    "_binary_metrics",
    "_marginal_effects",
    # heteroskedasticity
    "breusch_pagan",
    "goldfeld_quandt",
    "white_test",
    # hypothesis
    "TestResult",
    "wald_test",
    "f_test",
    "t_test_coeff",
    "lr_test",
    "score_test",
    "conf_int",
    # marginal effects
    "MarginalEffectsResult",
    "marginal_effects",
    "marginal_effects_at_values",
    # multicollinearity
    "compute_vif",
    "condition_number",
    # normality
    "jarque_bera",
    "omnibus_test",
    "shapiro_wilk",
    # serial correlation
    "autocorr_from_series",
    "box_pierce_from_autocorr",
    "box_pierce_q",
    "breusch_godfrey",
    "durbin_watson",
    "ljung_box_from_autocorr",
    "ljung_box_q",
    # specification
    "chow_test",
    "harvey_collier",
    "reset_test",
    # stationarity
    "adf_test",
    "kpss_test",
    "pp_test",
    # time series
    "granger_causality",
    "lead_exogeneity_test",
    "select_var_lag",
    # IV checks
    "basmann_f_test",
    "basmann_test",
    "run_iv_diagnostics",
    "sargan_test",
    "weak_instrument_tests",
    "wu_hausman_test",
    # panel checks
    "breusch_pagan_lm_test",
    "hausman_fe_re",
    "lead_test_strict_exogeneity",
    "run_panel_diagnostics",
    # structural breaks
    "cusum_test",
    "cusum_sq_test",
    "andrews_sup_wald",
    "chow_forecast_test",
    "recursive_residuals",
    # difference-in-differences
    "DIDResult",
    "EventStudyResult",
    "did_estimate",
    "event_study",
    "parallel_trends_test",
    # event study — finance-style CAR
    "EventStudyCARResult",
    "MarketModel",
    "bmp_car_test",
    "compute_abnormal_returns",
    "corrado_rank_test",
    "cowan_sign_test",
    "cumulative_abnormal_returns",
    "estimate_market_model",
    "event_study_car",
    "patell_car_test",
    # decomposition
    "OaxacaBlinderResult",
    "oaxaca_blinder",
    # matching & IPW
    "PropensityResult",
    "MatchResult",
    "IPWResult",
    "DoublyRobustResult",
    "propensity_score",
    "nearest_neighbor_match",
    "ipw_estimate",
    "doubly_robust",
    "covariate_balance",
]
