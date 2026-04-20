"""Diagnostics for IV/2SLS models."""

from __future__ import annotations

from typing import Iterable

import math
import re

from econtools.inference.hypothesis import TestResult
from econtools.models._results import RegressionResult


def wu_hausman_test(result: RegressionResult) -> TestResult:
    """Wu-Hausman test for endogeneity.

    H0: Regressors are exogenous (OLS is consistent).
    """
    if not hasattr(result.raw, "wu_hausman"):
        raise ValueError("wu_hausman test is only available for IV results.")
    sm_test = result.raw.wu_hausman()
    stat, pval, df = _unwrap_test(sm_test)
    return TestResult(
        test_name="Wu-Hausman",
        statistic=stat,
        pvalue=pval,
        df=df,
        distribution="F",
        null_hypothesis="Regressors are exogenous (OLS consistent)",
        reject=pval < 0.05,
    )


def sargan_test(result: RegressionResult) -> TestResult:
    """Sargan test of overidentifying restrictions."""
    if not hasattr(result.raw, "sargan"):
        raise ValueError("sargan test is only available for IV results.")
    sm_test = result.raw.sargan
    stat, pval, df = _unwrap_test(sm_test)
    return TestResult(
        test_name="Sargan",
        statistic=stat,
        pvalue=pval,
        df=df,
        distribution="Chi2",
        null_hypothesis="Overidentifying restrictions are valid",
        reject=bool(pval < 0.05) if not math.isnan(pval) else False,
    )


def basmann_test(result: RegressionResult) -> TestResult:
    """Basmann test of overidentifying restrictions."""
    if not hasattr(result.raw, "basmann"):
        raise ValueError("basmann test is only available for IV results.")
    sm_test = result.raw.basmann
    stat, pval, df = _unwrap_test(sm_test)
    return TestResult(
        test_name="Basmann",
        statistic=stat,
        pvalue=pval,
        df=df,
        distribution="Chi2",
        null_hypothesis="Overidentifying restrictions are valid",
        reject=bool(pval < 0.05) if not math.isnan(pval) else False,
    )


def basmann_f_test(result: RegressionResult) -> TestResult:
    """Basmann F test of overidentifying restrictions."""
    if not hasattr(result.raw, "basmann_f"):
        raise ValueError("basmann_f test is only available for IV results.")
    sm_test = result.raw.basmann_f
    stat, pval, df = _unwrap_test(sm_test)
    return TestResult(
        test_name="Basmann F",
        statistic=stat,
        pvalue=pval,
        df=df,
        distribution="F",
        null_hypothesis="Overidentifying restrictions are valid",
        reject=bool(pval < 0.05) if not math.isnan(pval) else False,
    )


def weak_instrument_tests(result: RegressionResult) -> list[TestResult]:
    """First-stage weak instrument diagnostics (per endogenous regressor)."""
    if not hasattr(result.raw, "first_stage"):
        raise ValueError("weak instrument diagnostics require IV results.")
    diag = result.raw.first_stage.diagnostics
    outputs: list[TestResult] = []
    for name, row in diag.iterrows():
        stat = float(row.get("f.stat", float("nan")))
        pval = float(row.get("f.pval", float("nan")))
        df = _parse_f_dist(row.get("f.dist"))
        outputs.append(
            TestResult(
                test_name=f"Weak instr. F ({name})",
                statistic=stat,
                pvalue=pval,
                df=df,
                distribution="F",
                null_hypothesis="Instruments are irrelevant",
                reject=bool(pval < 0.05) if not math.isnan(pval) else False,
            )
        )
    return outputs


def j_test(result: RegressionResult) -> TestResult:
    """Hansen J test of overidentifying restrictions (GMM only).

    H0: All overidentifying restrictions are valid.
    """
    j_obj = getattr(result.raw, "j_stat", None)
    if j_obj is None:
        raise ValueError(
            "j_test is only available for GMM results (IVGMM / IVGMMCUE). "
            "Use sargan_test() for 2SLS."
        )
    stat = float(getattr(j_obj, "stat", float("nan")))
    pval = float(getattr(j_obj, "pval", float("nan")))
    df_val = getattr(j_obj, "df", None)
    df = float(df_val) if df_val is not None else None
    return TestResult(
        test_name="Hansen J",
        statistic=stat,
        pvalue=pval,
        df=df,
        distribution="Chi2",
        null_hypothesis="Overidentifying restrictions are valid",
        reject=bool(pval < 0.05) if not math.isnan(pval) else False,
    )


def c_statistic(
    result: RegressionResult,
    variables: list[str],
) -> TestResult:
    """C-statistic (difference-in-J) test for endogenous variable exogeneity.

    Tests whether one or more variables that were treated as endogenous are
    actually exogenous, using the difference between two J-statistics.  Only
    meaningful for overidentified GMM results.

    Parameters
    ----------
    result:
        A GMM :class:`~econtools._core.types.Estimate`.
    variables:
        Names of *endogenous* variables (``endog_vars`` in the original
        :class:`~econtools.model.spec.ModelSpec`) whose exogeneity is to be
        tested.
    """
    c_fn = getattr(result.raw, "c_stat", None)
    if c_fn is None:
        raise ValueError(
            "c_statistic is only available for GMM results (IVGMM / IVGMMCUE)."
        )
    c_obj = c_fn(variables)
    stat = float(getattr(c_obj, "stat", float("nan")))
    pval = float(getattr(c_obj, "pval", float("nan")))
    df_val = getattr(c_obj, "df", None)
    df = float(df_val) if df_val is not None else None
    return TestResult(
        test_name=f"C-statistic ({', '.join(variables)})",
        statistic=stat,
        pvalue=pval,
        df=df,
        distribution="Chi2",
        null_hypothesis=f"Instruments {variables} are exogenous",
        reject=bool(pval < 0.05) if not math.isnan(pval) else False,
    )


def run_iv_diagnostics(
    result: RegressionResult,
    tests: Iterable[str] | None = None,
) -> list[TestResult]:
    """Run IV diagnostics; filter by test names if provided.

    For GMM results, includes the Hansen J-test instead of Sargan/Basmann.
    """
    tests = [t.strip().lower() for t in tests] if tests else None
    is_gmm = getattr(result.raw, "j_stat", None) is not None
    outputs: list[TestResult] = []

    if tests is None or "wu_hausman" in tests or "wu-hausman" in tests:
        try:
            outputs.append(wu_hausman_test(result))
        except ValueError:
            pass

    if is_gmm:
        if tests is None or "j_stat" in tests or "j-stat" in tests or "hansen" in tests:
            try:
                outputs.append(j_test(result))
            except ValueError:
                pass
    else:
        if tests is None or "sargan" in tests:
            try:
                outputs.append(sargan_test(result))
            except ValueError:
                pass
        if tests is None or "basmann" in tests:
            try:
                outputs.append(basmann_test(result))
            except ValueError:
                pass
        if tests is None or "basmann_f" in tests or "basmann-f" in tests:
            try:
                outputs.append(basmann_f_test(result))
            except ValueError:
                pass

    if tests is None or "weak" in tests or "weak_f" in tests or "weak-f" in tests:
        try:
            outputs.extend(weak_instrument_tests(result))
        except ValueError:
            pass

    return outputs


def _unwrap_test(sm_test: object) -> tuple[float, float, float | None]:
    """Extract stat, pval, df from linearmodels test results."""
    stat = getattr(sm_test, "stat", None)
    pval = getattr(sm_test, "pval", None)
    df = getattr(sm_test, "df", None)
    if stat is not None and pval is not None:
        return float(stat), float(pval), float(df) if df is not None else None

    if isinstance(sm_test, tuple) and len(sm_test) >= 2:
        stat = float(sm_test[0])
        pval = float(sm_test[1])
        df = float(sm_test[2]) if len(sm_test) > 2 else None
        return stat, pval, df

    raise ValueError("Unable to parse test results.")


def _parse_f_dist(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"F\(([\d.]+),\s*([\d.]+)\)", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))
