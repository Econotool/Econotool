"""Standard errors and hypothesis tests — re-export shim.

Statistical content now lives in ``econtools.evaluation.hypothesis`` and
``econtools._core.cov_mapping``.  This module re-exports everything for
backward compatibility.
"""

from econtools._core.cov_mapping import VALID_COV_TYPES  # noqa: F401
from econtools.inference.se_types import resolve_cov_args  # noqa: F401
from econtools._core.types import TestResult  # noqa: F401
from econtools.inference.hypothesis import (  # noqa: F401
    wald_test,
    f_test,
    t_test_coeff,
    lr_test,
    score_test,
    conf_int,
)

__all__ = [
    "VALID_COV_TYPES",
    "resolve_cov_args",
    "TestResult",
    "wald_test",
    "f_test",
    "t_test_coeff",
    "lr_test",
    "score_test",
    "conf_int",
]
