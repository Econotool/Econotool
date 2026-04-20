"""Hypothesis tests — re-export from inference.hypothesis.

This module is the canonical evaluation location.  The inference module
re-exports from here for backward compatibility.
"""

from econtools.inference.hypothesis import (  # noqa: F401
    TestResult,
    wald_test,
    f_test,
    t_test_coeff,
    lr_test,
    score_test,
    conf_int,
)

__all__ = [
    "TestResult",
    "wald_test",
    "f_test",
    "t_test_coeff",
    "lr_test",
    "score_test",
    "conf_int",
]
