"""Table generation (reg_table, compare_table, LaTeX/HTML) — Phase 1+."""

from econtools.tables.compare_table import compare_table
from econtools.tables.diagnostic_table import diagnostic_table
from econtools.tables.latex_utils import combine_tex, compile_tex_to_pdf, open_pdf
from econtools.tables.reg_table import reg_table

__all__ = [
    "combine_tex",
    "compile_tex_to_pdf",
    "open_pdf",
    "compare_table",
    "diagnostic_table",
    "reg_table",
]
