"""Chainable transform recorder used by :class:`Paper`.

A :class:`DataTransformer` records a sequence of data transformations to
apply to a named dataset.  The transforms are pure wrappers around
:mod:`econtools.data.transform` — all real work happens there, the
transformer only buffers the operations until :meth:`apply` is called.

Every applied step logs via :func:`econtools.data.provenance.log_step`
when a provenance log path is supplied; otherwise the log call is a
silent no-op.

Public API
----------
TransformStep      — dataclass describing one deferred transform
DataTransformer    — fluent-builder recorder
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from econtools.data import transform as _tx
from econtools.data.provenance import log_step


# ---------------------------------------------------------------------------
# Deferred step description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformStep:
    """One deferred transform operation.

    Attributes
    ----------
    function:
        Name of the function in :mod:`econtools.data.transform`.
    args:
        Positional arguments (excluding the DataFrame itself).
    kwargs:
        Keyword arguments passed through to the function.
    """

    function: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fluent recorder
# ---------------------------------------------------------------------------


class DataTransformer:
    """Fluent builder recording a chain of data transforms.

    Usage::

        paper.transform.log("gdp_pc").lag("unemp", entity="state", k=1)

    The transforms are executed lazily when :meth:`apply` is invoked —
    typically by :meth:`Paper.build`.  Each step records a call to
    :func:`econtools.data.provenance.log_step` when a provenance log path
    is supplied.
    """

    # Map public method names → real function in econtools.data.transform.
    _METHOD_FUNCS: dict[str, str] = {
        "log": "log_col",
        "log1p": "log1p_col",
        "lag": "lag",
        "lead": "lead",
        "diff": "diff_col",
        "growth": "growth_rate",
        "interact": "interact",
        "poly": "poly",
        "standardise": "standardise",
        "demean": "demean_within",
        "time_trend": "time_trend",
        "rolling_mean": "rolling_mean",
        "dummies": "dummies",
    }

    def __init__(self) -> None:
        self._steps: list[TransformStep] = []

    # ------------------------------------------------------------------ #
    # Public fluent methods                                              #
    # ------------------------------------------------------------------ #

    def log(self, col: str, name: str | None = None) -> "DataTransformer":
        """Record ``log_<col>`` transform."""
        self._steps.append(
            TransformStep("log_col", (col,), {"name": name} if name else {})
        )
        return self

    def log1p(self, col: str, name: str | None = None) -> "DataTransformer":
        self._steps.append(
            TransformStep("log1p_col", (col,), {"name": name} if name else {})
        )
        return self

    def lag(
        self,
        col: str,
        k: int = 1,
        *,
        entity: str | None = None,
        name: str | None = None,
    ) -> "DataTransformer":
        """Record a panel lag.  ``entity`` falls back to the Paper's panel id."""
        kwargs: dict[str, Any] = {"k": k}
        if entity is not None:
            kwargs["entity"] = entity
        if name is not None:
            kwargs["name"] = name
        self._steps.append(TransformStep("lag", (col,), kwargs))
        return self

    def lead(
        self,
        col: str,
        k: int = 1,
        *,
        entity: str | None = None,
        name: str | None = None,
    ) -> "DataTransformer":
        kwargs: dict[str, Any] = {"k": k}
        if entity is not None:
            kwargs["entity"] = entity
        if name is not None:
            kwargs["name"] = name
        self._steps.append(TransformStep("lead", (col,), kwargs))
        return self

    def diff(
        self,
        col: str,
        *,
        entity: str | None = None,
        k: int = 1,
        name: str | None = None,
    ) -> "DataTransformer":
        kwargs: dict[str, Any] = {"k": k}
        if entity is not None:
            kwargs["entity"] = entity
        if name is not None:
            kwargs["name"] = name
        self._steps.append(TransformStep("diff_col", (col,), kwargs))
        return self

    def growth(
        self,
        col: str,
        *,
        entity: str | None = None,
        k: int = 1,
        name: str | None = None,
    ) -> "DataTransformer":
        kwargs: dict[str, Any] = {"k": k}
        if entity is not None:
            kwargs["entity"] = entity
        if name is not None:
            kwargs["name"] = name
        self._steps.append(TransformStep("growth_rate", (col,), kwargs))
        return self

    def interact(
        self, col1: str, col2: str, name: str | None = None
    ) -> "DataTransformer":
        self._steps.append(
            TransformStep(
                "interact",
                (col1, col2),
                {"name": name} if name else {},
            )
        )
        return self

    def poly(
        self, col: str, degree: int = 2, name: str | None = None
    ) -> "DataTransformer":
        kwargs: dict[str, Any] = {"degree": degree}
        if name is not None:
            kwargs["name"] = name
        self._steps.append(TransformStep("poly", (col,), kwargs))
        return self

    def standardise(self, col: str, name: str | None = None) -> "DataTransformer":
        self._steps.append(
            TransformStep("standardise", (col,), {"name": name} if name else {})
        )
        return self

    def demean(
        self, col: str, *, entity: str | None = None, name: str | None = None
    ) -> "DataTransformer":
        kwargs: dict[str, Any] = {}
        if entity is not None:
            kwargs["entity"] = entity
        if name is not None:
            kwargs["name"] = name
        self._steps.append(TransformStep("demean_within", (col,), kwargs))
        return self

    def time_trend(
        self, *, entity: str | None = None, name: str = "t"
    ) -> "DataTransformer":
        kwargs: dict[str, Any] = {"name": name}
        if entity is not None:
            kwargs["entity"] = entity
        self._steps.append(TransformStep("time_trend", (), kwargs))
        return self

    def rolling_mean(
        self,
        col: str,
        k: int,
        *,
        entity: str | None = None,
        name: str | None = None,
    ) -> "DataTransformer":
        kwargs: dict[str, Any] = {"k": k}
        if entity is not None:
            kwargs["entity"] = entity
        if name is not None:
            kwargs["name"] = name
        self._steps.append(TransformStep("rolling_mean", (col,), kwargs))
        return self

    def dummies(
        self,
        col: str,
        *,
        drop_first: bool = True,
        prefix: str | None = None,
    ) -> "DataTransformer":
        self._steps.append(
            TransformStep(
                "dummies",
                (col,),
                {"drop_first": drop_first, "prefix": prefix},
            )
        )
        return self

    # ------------------------------------------------------------------ #
    # Introspection / execution                                          #
    # ------------------------------------------------------------------ #

    @property
    def steps(self) -> list[TransformStep]:
        """Return the recorded steps (read-only view)."""
        return list(self._steps)

    def __len__(self) -> int:  # pragma: no cover — trivial
        return len(self._steps)

    def apply(
        self,
        df: pd.DataFrame,
        *,
        entity: str | None = None,
        log_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Execute all recorded steps against *df* and return the result.

        Parameters
        ----------
        df:
            Input DataFrame.
        entity:
            Default panel entity column used by lag/lead/diff/etc when the
            step did not specify one.  Typically the Paper's ``entity_col``.
        log_path:
            If provided, every step is appended to this provenance log.
        """
        out = df
        for step in self._steps:
            fn = getattr(_tx, step.function)
            kwargs = dict(step.kwargs)
            # Supply default entity for panel-aware steps
            if (
                step.function in {"lag", "lead", "diff_col", "growth_rate",
                                    "demean_within", "time_trend", "rolling_mean"}
                and "entity" not in kwargs
                and entity is not None
            ):
                kwargs["entity"] = entity
            out = fn(out, *step.args, **kwargs)
            if log_path is not None:
                log_step(
                    log_path,
                    f"transform.{step.function}",
                    {"args": list(step.args), **kwargs},
                )
        return out
