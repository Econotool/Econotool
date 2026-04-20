"""Binary model evaluation metrics and marginal effects.

Extracted from ``output.tables.reg_table`` so that statistical computation
lives in ``evaluation/`` and rendering stays in ``output/``.

Public API (package-internal)
------------------------------
_BinaryMetrics     — frozen dataclass of binary classification metrics
_binary_metrics    — compute _BinaryMetrics from a RegressionResult
_marginal_effects  — compute AME/MEM dict from a RegressionResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from econtools._core.types import RegressionResult


@dataclass(frozen=True)
class _BinaryMetrics:
    ybar: float
    acc_05: float
    acc_ybar: float
    acc_match_ybar: float
    bal_acc_05: float
    r2_corr: float
    r2_efron: float
    mcfadden_r2: float
    brier: float
    tp: int
    tn: int
    fp: int
    fn: int
    tp_star: int
    tn_star: int
    fp_star: int
    fn_star: int
    tpr: float
    tnr: float
    ppv: float
    npv: float
    tpr_star: float
    tnr_star: float
    ppv_star: float
    npv_star: float
    c_match_ybar: float
    auc: float


def _binary_metrics(result: RegressionResult) -> _BinaryMetrics:
    """Compute binary classification metrics from a fitted Probit/Logit result."""
    y = np.asarray(result.raw.model.endog).astype(float)
    p = np.asarray(result.raw.predict())

    ybar = float(np.mean(y)) if y.size else float("nan")

    def _confusion(y_true: np.ndarray, y_hat: np.ndarray) -> tuple[int, int, int, int]:
        tp = int(np.sum((y_hat == 1) & (y_true == 1)))
        tn = int(np.sum((y_hat == 0) & (y_true == 0)))
        fp = int(np.sum((y_hat == 1) & (y_true == 0)))
        fn = int(np.sum((y_hat == 0) & (y_true == 1)))
        return tp, tn, fp, fn

    yhat_05 = (p >= 0.5).astype(int)
    tp, tn, fp, fn = _confusion(y, yhat_05)
    acc_05 = float(np.mean(yhat_05 == y)) if y.size else float("nan")

    yhat_ybar = (p >= ybar).astype(int) if not math.isnan(ybar) else np.zeros_like(y)
    acc_ybar = float(np.mean(yhat_ybar == y)) if y.size else float("nan")

    if 0.0 < ybar < 1.0:
        c_match_ybar = float(np.quantile(p, 1.0 - ybar))
    else:
        c_match_ybar = 0.5
    yhat_match = (p >= c_match_ybar).astype(int)
    acc_match_ybar = float(np.mean(yhat_match == y)) if y.size else float("nan")
    tp_s, tn_s, fp_s, fn_s = _confusion(y, yhat_match)

    def _safe_rate(num: int, denom: int) -> float:
        return float(num / denom) if denom > 0 else float("nan")

    tpr = _safe_rate(tp, tp + fn)
    tnr = _safe_rate(tn, tn + fp)
    ppv = _safe_rate(tp, tp + fp)
    npv = _safe_rate(tn, tn + fn)
    bal_acc_05 = (tpr + tnr) / 2.0 if not math.isnan(tpr) and not math.isnan(tnr) else float("nan")
    tpr_s = _safe_rate(tp_s, tp_s + fn_s)
    tnr_s = _safe_rate(tn_s, tn_s + fp_s)
    ppv_s = _safe_rate(tp_s, tp_s + fp_s)
    npv_s = _safe_rate(tn_s, tn_s + fn_s)

    if np.std(p) > 0 and np.std(y) > 0:
        r2_corr = float(np.corrcoef(p, y)[0, 1] ** 2)
    else:
        r2_corr = float("nan")

    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - ybar) ** 2))
    r2_efron = 1.0 - sse / sst if sst > 0 else float("nan")
    brier = sse / float(y.size) if y.size else float("nan")

    # ROC AUC via rank statistic (handles ties)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos > 0 and n_neg > 0:
        ranks = rankdata(p)
        sum_ranks_pos = float(np.sum(ranks[y == 1]))
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    else:
        auc = float("nan")

    llf = float(getattr(result.raw, "llf", float("nan")))
    llnull = float(getattr(result.raw, "llnull", float("nan")))
    if not math.isnan(llf) and not math.isnan(llnull) and llnull != 0.0:
        mcfadden_r2 = 1.0 - llf / llnull
    else:
        mcfadden_r2 = float(result.fit.r_squared)

    return _BinaryMetrics(
        ybar=ybar,
        acc_05=acc_05,
        acc_ybar=acc_ybar,
        acc_match_ybar=acc_match_ybar,
        bal_acc_05=bal_acc_05,
        r2_corr=r2_corr,
        r2_efron=r2_efron,
        mcfadden_r2=mcfadden_r2,
        brier=brier,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        tp_star=tp_s,
        tn_star=tn_s,
        fp_star=fp_s,
        fn_star=fn_s,
        tpr=tpr,
        tnr=tnr,
        ppv=ppv,
        npv=npv,
        tpr_star=tpr_s,
        tnr_star=tnr_s,
        ppv_star=ppv_s,
        npv_star=npv_s,
        c_match_ybar=c_match_ybar,
        auc=auc,
    )


def _marginal_effects(
    result: RegressionResult,
    at: str,
) -> dict[str, tuple[float, float, float]]:
    """Compute average (AME) or marginal (MEM) effects for a binary model."""
    me = result.raw.get_margeff(at=at, dummy=True)
    frame = me.summary_frame()
    effects: dict[str, tuple[float, float, float]] = {}
    for name in frame.index:
        effects[str(name)] = (
            float(frame.loc[name, "dy/dx"]),
            float(frame.loc[name, "Std. Err."]),
            float(frame.loc[name, "Pr(>|z|)"]),
        )
    return effects
