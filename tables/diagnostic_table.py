"""Diagnostic test table formatter."""

from __future__ import annotations

from econtools._core.formatting import _fmt
from econtools._core.types import TestResult


def diagnostic_table(
    results: list[TestResult],
    *,
    digits: int = 3,
    format: str = "text",
    title: str | None = None,
) -> str:
    if format == "text":
        return _text_table(results, digits, title)
    if format == "latex":
        return _latex_table(results, digits, title)
    if format == "html":
        return _html_table(results, digits, title)
    raise ValueError(
        f"Unknown format '{format}'. Choose from 'text', 'latex', 'html'."
    )


def _df_str(df: float | tuple[float, float] | None) -> str:
    if df is None:
        return ""
    if isinstance(df, tuple):
        return f"{df[0]:.0f}, {df[1]:.0f}"
    return f"{df:.0f}"


def _text_table(results: list[TestResult], digits: int, title: str | None) -> str:
    name_w = max(16, max((len(r.test_name) for r in results), default=0) + 2)
    col_w = 12
    sep = "-" * (name_w + col_w * 5)

    lines: list[str] = []
    if title:
        lines.append(title)
    lines += [
        sep,
        f"{'Test':<{name_w}}"
        f"{'Stat':>{col_w}}"
        f"{'p-value':>{col_w}}"
        f"{'df':>{col_w}}"
        f"{'Dist':>{col_w}}"
        f"{'Reject':>{col_w}}",
        sep,
    ]
    for r in results:
        lines.append(
            f"{r.test_name:<{name_w}}"
            f"{_fmt(r.statistic, digits):>{col_w}}"
            f"{_fmt(r.pvalue, digits):>{col_w}}"
            f"{_df_str(r.df):>{col_w}}"
            f"{r.distribution:>{col_w}}"
            f"{'Yes' if r.reject else 'No':>{col_w}}"
        )
        lines.append(f"{'H0:':<{name_w}} {r.null_hypothesis}")
    lines.append(sep)
    return "\n".join(lines)


def _latex_table(results: list[TestResult], digits: int, title: str | None) -> str:
    lines: list[str] = [
        r"\renewcommand{\arraystretch}{1.1}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{lrrrrl}",
        r"\hline",
    ]
    if title:
        lines.append(r"\multicolumn{6}{l}{\textit{" + title + r"}} \\")
        lines.append(r"\hline")
    lines += [
        r"Test & Stat & $p$ & df & Dist & Reject \\",
        r"\hline",
    ]
    for r in results:
        lines.append(
            f"{r.test_name} & {_fmt(r.statistic, digits)} & {_fmt(r.pvalue, digits)} "
            f"& {_df_str(r.df)} & {r.distribution} & {'Yes' if r.reject else 'No'} \\\\"
        )
        lines.append(
            r"\multicolumn{6}{l}{\footnotesize{H0: "
            + r.null_hypothesis
            + r"}} \\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _html_table(results: list[TestResult], digits: int, title: str | None) -> str:
    lines: list[str] = ["<table>"]
    if title:
        lines.append(f"<caption>{title}</caption>")
    lines.append(
        "<tr><th>Test</th><th>Stat</th><th>p</th><th>df</th><th>Dist</th><th>Reject</th></tr>"
    )
    for r in results:
        lines.append(
            "<tr>"
            f"<td>{r.test_name}</td>"
            f"<td>{_fmt(r.statistic, digits)}</td>"
            f"<td>{_fmt(r.pvalue, digits)}</td>"
            f"<td>{_df_str(r.df)}</td>"
            f"<td>{r.distribution}</td>"
            f"<td>{'Yes' if r.reject else 'No'}</td>"
            "</tr>"
        )
        lines.append(f"<tr><td colspan='6'><em>H0:</em> {r.null_hypothesis}</td></tr>")
    lines.append("</table>")
    return "\n".join(lines)
