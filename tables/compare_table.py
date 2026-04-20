"""Multi-model regression comparison table formatter."""

from __future__ import annotations

from econtools._core.formatting import _fmt, _latex_star, _star
from econtools._core.types import RegressionResult


def compare_table(
    results: list[RegressionResult],
    labels: list[str] | None = None,
    stars: bool = True,
    se_in_parens: bool = True,
    digits: int = 3,
    format: str = "text",
    title: str | None = None,
) -> str:
    if labels is not None and len(labels) != len(results):
        raise ValueError("labels length must match number of results.")
    if format == "text":
        return _text_table(results, labels, stars, se_in_parens, digits, title)
    if format == "latex":
        return _latex_table(results, labels, stars, se_in_parens, digits, title)
    if format == "html":
        return _html_table(results, labels, stars, se_in_parens, digits, title)
    raise ValueError(
        f"Unknown format '{format}'. Choose from 'text', 'latex', 'html'."
    )


def _default_labels(results: list[RegressionResult]) -> list[str]:
    return [f"Model {i + 1}" for i in range(len(results))]


def _collect_vars(results: list[RegressionResult]) -> list[str]:
    names: list[str] = []
    seen = set()
    for res in results:
        for name in res.params.index:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _text_table(
    results: list[RegressionResult],
    labels: list[str] | None,
    stars: bool,
    se_in_parens: bool,
    digits: int,
    title: str | None,
) -> str:
    labels = labels or _default_labels(results)
    var_names = _collect_vars(results)
    var_w = max(20, max(len(n) for n in var_names) + 2)
    col_w = max(14, max(len(l) for l in labels) + 2)
    sep = "-" * (var_w + col_w * len(results))

    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(sep)
    header = f"{'Variable':<{var_w}}" + "".join(
        f"{labels[i]:>{col_w}}" for i in range(len(results))
    )
    lines.append(header)
    lines.append(sep)

    for name in var_names:
        row = f"{name:<{var_w}}"
        for res in results:
            if name in res.params.index:
                coef = float(res.params[name])
                pval = float(res.pvalues[name])
                s = _star(pval) if stars else ""
                row += f"{(_fmt(coef, digits) + s):>{col_w}}"
            else:
                row += " " * col_w
        lines.append(row)
        if se_in_parens:
            row = f"{'':<{var_w}}"
            for res in results:
                if name in res.bse.index:
                    se = float(res.bse[name])
                    row += f"{('(' + _fmt(se, digits) + ')'):>{col_w}}"
                else:
                    row += " " * col_w
            lines.append(row)

    lines.append(sep)
    stats = [
        ("N", lambda r: str(int(r.fit.nobs))),
        ("R\u00b2", lambda r: _fmt(r.fit.r_squared, digits)),
        ("Adj. R\u00b2", lambda r: _fmt(r.fit.r_squared_adj, digits)),
        ("F-stat", lambda r: _fmt(r.fit.f_stat, digits)),
        ("RMSE", lambda r: _fmt(r.fit.rmse, digits)),
    ]
    for label, fn in stats:
        row = f"{label:<{var_w}}" + "".join(
            f"{fn(res):>{col_w}}" for res in results
        )
        lines.append(row)
    lines.append(sep)
    if stars:
        lines.append("* p<0.1  ** p<0.05  *** p<0.01")

    return "\n".join(lines)


def _latex_table(
    results: list[RegressionResult],
    labels: list[str] | None,
    stars: bool,
    se_in_parens: bool,
    digits: int,
    title: str | None,
) -> str:
    labels = labels or _default_labels(results)
    var_names = _collect_vars(results)
    cols = "l" + "r" * len(results)
    lines: list[str] = [
        r"\renewcommand{\arraystretch}{1.1}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{" + cols + r"}",
        r"\hline",
    ]
    if title:
        lines.append(
            r"\multicolumn{"
            + str(len(results) + 1)
            + r"}{l}{\textit{"
            + title
            + r"}} \\"
        )
        lines.append(r"\hline")
    header = "Variable & " + " & ".join(labels) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    for name in var_names:
        row = [name]
        for res in results:
            if name in res.params.index:
                coef = float(res.params[name])
                pval = float(res.pvalues[name])
                s = _star(pval) if stars else ""
                row.append(_fmt(coef, digits) + _latex_star(s))
            else:
                row.append("")
        lines.append(" & ".join(row) + r" \\")
        if se_in_parens:
            row = [""]
            for res in results:
                if name in res.bse.index:
                    se = float(res.bse[name])
                    row.append(r"$(" + _fmt(se, digits) + r")$")
                else:
                    row.append("")
            lines.append(" & ".join(row) + r" \\")

    lines.append(r"\hline")
    stats = [
        ("$N$", lambda r: str(int(r.fit.nobs))),
        ("$R^2$", lambda r: _fmt(r.fit.r_squared, digits)),
        ("Adj.\\ $R^2$", lambda r: _fmt(r.fit.r_squared_adj, digits)),
        ("$F$-stat", lambda r: _fmt(r.fit.f_stat, digits)),
        ("RMSE", lambda r: _fmt(r.fit.rmse, digits)),
    ]
    for label, fn in stats:
        row = [label] + [fn(res) for res in results]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    if stars:
        lines.append(r"\vspace{2pt}")
        lines.append(
            r"\begin{flushleft}\footnotesize{* $p<0.1$, ** $p<0.05$, *** $p<0.01$}\end{flushleft}"
        )
    return "\n".join(lines)


def _html_table(
    results: list[RegressionResult],
    labels: list[str] | None,
    stars: bool,
    se_in_parens: bool,
    digits: int,
    title: str | None,
) -> str:
    labels = labels or _default_labels(results)
    var_names = _collect_vars(results)
    lines: list[str] = ["<table>"]
    if title:
        lines.append(f"<caption>{title}</caption>")
    lines.append(
        "<tr><th>Variable</th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr>"
    )

    for name in var_names:
        row = f"<tr><td>{name}</td>"
        for res in results:
            if name in res.params.index:
                coef = float(res.params[name])
                pval = float(res.pvalues[name])
                s = _star(pval) if stars else ""
                coef_str = (
                    f"{_fmt(coef, digits)}<sup>{s}</sup>" if s else _fmt(coef, digits)
                )
                row += f"<td>{coef_str}</td>"
            else:
                row += "<td></td>"
        row += "</tr>"
        lines.append(row)
        if se_in_parens:
            row = "<tr><td></td>"
            for res in results:
                if name in res.bse.index:
                    se = float(res.bse[name])
                    row += f"<td>({_fmt(se, digits)})</td>"
                else:
                    row += "<td></td>"
            row += "</tr>"
            lines.append(row)

    lines.append("<tr><td colspan='{}'><hr></td></tr>".format(len(results) + 1))
    stats = [
        ("N", lambda r: str(int(r.fit.nobs))),
        ("R&sup2;", lambda r: _fmt(r.fit.r_squared, digits)),
        ("Adj. R&sup2;", lambda r: _fmt(r.fit.r_squared_adj, digits)),
        ("F-stat", lambda r: _fmt(r.fit.f_stat, digits)),
        ("RMSE", lambda r: _fmt(r.fit.rmse, digits)),
    ]
    for label, fn in stats:
        row = f"<tr><td>{label}</td>" + "".join(
            f"<td>{fn(res)}</td>" for res in results
        ) + "</tr>"
        lines.append(row)
    lines.append("</table>")
    if stars:
        lines.append(
            "<p><small>* p&lt;0.1, ** p&lt;0.05, *** p&lt;0.01</small></p>"
        )
    return "\n".join(lines)
