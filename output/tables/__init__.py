"""Table renderers — re-export from econtools.tables."""

from econtools.tables.compare_table import compare_table  # noqa: F401
from econtools.tables.diagnostic_table import diagnostic_table  # noqa: F401
from econtools.tables.reg_table import reg_table  # noqa: F401
from econtools.output.tables.pub_latex import (  # noqa: F401
    DiagnosticsTable,
    InfoTable,
    ResultsTable,
    SummaryTable,
)
from econtools.output.tables.panelled_page import (  # noqa: F401
    PanelEntry,
    PanelledPage,
)
from econtools.output.tables.inline import (  # noqa: F401
    diagnostics_panel,
    info_panel,
    results_panel,
    summary_panel,
)

__all__ = [
    "compare_table",
    "diagnostic_table",
    "reg_table",
    "DiagnosticsTable",
    "InfoTable",
    "ResultsTable",
    "SummaryTable",
    "PanelEntry",
    "PanelledPage",
    "diagnostics_panel",
    "info_panel",
    "results_panel",
    "summary_panel",
]
