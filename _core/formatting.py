"""Shared formatting helpers for tables and LaTeX output.

Single authoritative copy — all table and LaTeX modules import from here.

Public API
----------
_star(pval) -> str
_fmt(x, digits) -> str
_latex_star(s) -> str
_latex_escape(text) -> str
"""

from __future__ import annotations


def _star(pval: float) -> str:
    """Return significance star string for a p-value."""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.1:
        return "*"
    return ""


def _fmt(x: float | None, digits: int) -> str:
    """Format a float to *digits* decimal places; return '' for None."""
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def _latex_star(s: str) -> str:
    """Convert a star string (e.g. '**') to a LaTeX superscript."""
    if not s:
        return ""
    count = s.count("*")
    return r"$^{" + r"\ast" * count + r"}$"


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in *text*."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out
