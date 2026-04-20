"""Replication engine for academic econometrics papers.

Provides a declarative schema for specifying what a paper estimates,
a parser for Stata .do files, and an orchestration runner that maps
specs → estimates → comparisons → diagnostics → reports.

Public API
----------
ReplicationSpec     — full paper replication specification
TableSpec           — one table from a paper
ColumnSpec          — one regression column
CausalClaim         — a paper's causal claim
DataRef             — pointer to source data
VariableDef         — variable definition / construction
SampleRestriction   — sample selection rule
ComparisonResult    — result of comparing estimate to published value
ReplicationRunner   — orchestrate a full replication
"""

from econtools.replication.spec import (
    CausalClaim,
    ColumnSpec,
    ComparisonResult,
    DataRef,
    ReplicationSpec,
    SampleRestriction,
    TableSpec,
    VariableDef,
)
from econtools.replication.runner import ReplicationRunner

__all__ = [
    "CausalClaim",
    "ColumnSpec",
    "ComparisonResult",
    "DataRef",
    "ReplicationRunner",
    "ReplicationSpec",
    "SampleRestriction",
    "TableSpec",
    "VariableDef",
]
