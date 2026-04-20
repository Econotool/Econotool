"""``econtools.paper`` — top-level paper orchestrator.

Public API
----------
Paper               — fluent-builder entry point
BuildResult         — :meth:`Paper.build` return value
DataTransformer     — fluent transform recorder (also exposed on ``paper.transform``)
ColumnSpec          — re-exported from :mod:`econtools.replication.spec` for convenience

Typical use::

    from econtools.paper import Paper

    paper = Paper(
        id="my_paper",
        title="...",
        authors=["A. Author"],
        abstract="...",
        profile="econometrica",
    )
    paper.load("data.parquet", as_="main", entity_col="state", time_col="year")
    paper.transform.log("gdp").lag("unemp", 1)
    paper.table_summary("tab_desc", vars=["unemp", "gdp"])
    paper.table_results("tab_main", columns=[...])
    paper.figure_residuals("fig_resid", model="tab_main:FE")
    paper.section("Introduction", body="...")
    paper.section("Results", after=["tab_main", "fig_resid"])
    paper.build(output_dir="build")
"""

from econtools.paper.builder import (
    BuildResult,
    CompositeNode,
    DataLoadNode,
    FigureNode,
    Paper,
    SectionNode,
    TableNode,
    TransformNode,
)
from econtools.paper.transforms import DataTransformer, TransformStep
from econtools.replication.spec import ColumnSpec

__all__ = [
    "BuildResult",
    "ColumnSpec",
    "DataLoadNode",
    "DataTransformer",
    "FigureNode",
    "Paper",
    "SectionNode",
    "TableNode",
    "TransformNode",
    "TransformStep",
]
