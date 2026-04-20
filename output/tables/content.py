"""Intermediate representation for table content.

:class:`TableContent` decouples statistical computation (what goes in the
table) from rendering (how it is formatted).  Renderers receive a
``TableContent`` and produce the output string.

Public API
----------
TableRow
TableContent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TableRow:
    """A single data row in a table.

    Parameters
    ----------
    label:
        Row label (variable name, test name, etc.).
    values:
        Ordered cell values (one per model/column).
    is_secondary:
        True for SE rows, p-value sub-rows, etc.
    note:
        Optional inline note (e.g. null hypothesis text).
    """

    label: str
    values: list[Any]
    is_secondary: bool = False
    note: str | None = None


@dataclass
class TableContent:
    """Pre-computed content for any table type.

    This object travels from the statistical computation layer to the
    rendering layer.  Renderers should never call statistical functions.

    Parameters
    ----------
    headers:
        Column headers (e.g. model labels or statistic names).
    rows:
        Ordered list of :class:`TableRow` objects.
    footer_rows:
        Rows below the separator (N, R², etc.).
    notes:
        List of table notes / significance footnotes.
    title:
        Optional table title.
    table_type:
        One of ``'regression'``, ``'comparison'``, ``'diagnostic'``.
    metadata:
        Arbitrary extra data for advanced renderers.
    """

    headers: list[str]
    rows: list[TableRow] = field(default_factory=list)
    footer_rows: list[TableRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    title: str | None = None
    table_type: str = "regression"
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_row(
        self,
        label: str,
        values: list[Any],
        *,
        is_secondary: bool = False,
        note: str | None = None,
    ) -> None:
        """Append a :class:`TableRow` to ``rows``."""
        self.rows.append(TableRow(label=label, values=values, is_secondary=is_secondary, note=note))

    def add_footer(
        self,
        label: str,
        values: list[Any],
    ) -> None:
        """Append a :class:`TableRow` to ``footer_rows``."""
        self.footer_rows.append(TableRow(label=label, values=values))
