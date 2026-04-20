"""Single-model regression table formatter.

Supports text (fixed-width), LaTeX, and HTML output.

Public API
----------
reg_table(result, stars, se_in_parens, digits, format) -> str
"""

from __future__ import annotations

from econtools._core.formatting import _fmt, _latex_star, _star  # noqa: F401
from econtools._core.types import RegressionResult
from econtools.evaluation.binary_metrics import (
    _BinaryMetrics,
    _binary_metrics,
    _marginal_effects,
)


def reg_table(
    result: RegressionResult,
    stars: bool = True,
    se_in_parens: bool = True,
    digits: int = 3,
    format: str = "text",
    title: str | None = None,
) -> str:
    """Format a regression result as a printable table.

    Parameters
    ----------
    result:
        Fitted :class:`RegressionResult`.
    stars:
        Append significance stars (``*`` p<0.1, ``**`` p<0.05,
        ``***`` p<0.01).
    se_in_parens:
        Print the standard error in parentheses on the row below each
        coefficient.
    digits:
        Number of decimal places for all numeric values.
    format:
        ``'text'`` (fixed-width), ``'latex'`` (tabular env), or
        ``'html'`` (table tags).
    title:
        Optional table title (rendered above the table).

    Returns
    -------
    Formatted table string.

    Raises
    ------
    ValueError
        If ``format`` is not one of the supported values.
    """
    if format == "text":
        return _text_table(result, stars, se_in_parens, digits, title)
    if format == "latex":
        return _latex_table(result, stars, se_in_parens, digits, title)
    if format == "html":
        return _html_table(result, stars, se_in_parens, digits, title)
    raise ValueError(
        f"Unknown format '{format}'. Choose from 'text', 'latex', 'html'."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_binary_result(result: RegressionResult) -> bool:
    model = result.model_type.lower()
    if "probit" in model or "logit" in model:
        return True
    return hasattr(result.raw, "prsquared")


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

_COL_W = 12  # numeric column width


def _text_table(
    result: RegressionResult,
    stars: bool,
    se_in_parens: bool,
    digits: int,
    title: str | None,
) -> str:
    var_w = max(20, max(len(n) for n in result.params.index) + 2)
    sep = "-" * (var_w + _COL_W * 4)

    lines: list[str] = []
    if title:
        lines.append(title)
    lines += [
        sep,
        f"{'Dep. variable:':<{var_w}} {result.dep_var}",
        f"{'Estimator:':<{var_w}} {result.model_type}",
        f"{'SE type:':<{var_w}} {result.cov_type}",
        sep,
        (
            f"{'Variable':<{var_w}}"
            f"{'Coef':>{_COL_W}}"
            f"{'Std Err':>{_COL_W}}"
            f"{'t-stat':>{_COL_W}}"
            f"{'p-value':>{_COL_W}}"
        ),
        sep,
    ]

    for name in result.params.index:
        coef = float(result.params[name])
        se = float(result.bse[name])
        tval = float(result.tvalues[name])
        pval = float(result.pvalues[name])
        s = _star(pval) if stars else ""

        coef_str = _fmt(coef, digits) + s
        lines.append(
            f"{name:<{var_w}}"
            f"{coef_str:>{_COL_W}}"
            f"{_fmt(se, digits):>{_COL_W}}"
            f"{_fmt(tval, digits):>{_COL_W}}"
            f"{_fmt(pval, digits):>{_COL_W}}"
        )
        if se_in_parens:
            se_str = f"({_fmt(se, digits)})"
            lines.append(f"{'':<{var_w}}{se_str:>{_COL_W}}")

    fit = result.fit
    lines.append(sep)
    if _is_binary_result(result):
        metrics = _binary_metrics(result)
        llr = float(getattr(result.raw, "llr", float("nan")))
        llr_pvalue = float(getattr(result.raw, "llr_pvalue", float("nan")))
        lines += [
            f"{'N':<{var_w}}{fit.nobs:>{_COL_W}}",
            f"{'Pseudo R\u00b2 (McFadden)':<{var_w}}{_fmt(metrics.mcfadden_r2, digits):>{_COL_W}}",
            f"{'Corr\u00b2(y,p\u0302)':<{var_w}}{_fmt(metrics.r2_corr, digits):>{_COL_W}}",
            f"{'R\u00b2 (Efron)':<{var_w}}{_fmt(metrics.r2_efron, digits):>{_COL_W}}",
            f"{'AUC':<{var_w}}{_fmt(metrics.auc, digits):>{_COL_W}}",
            f"{'Log-likelihood':<{var_w}}{_fmt(fit.log_likelihood, digits):>{_COL_W}}",
            f"{'LR stat':<{var_w}}{_fmt(llr, digits):>{_COL_W}}",
            f"{'LR p-value':<{var_w}}{_fmt(llr_pvalue, digits):>{_COL_W}}",
            f"{'AIC':<{var_w}}{_fmt(fit.aic, digits):>{_COL_W}}",
            f"{'BIC':<{var_w}}{_fmt(fit.bic, digits):>{_COL_W}}",
            f"{'Brier score':<{var_w}}{_fmt(metrics.brier, digits):>{_COL_W}}",
            f"{'Percent correct (c=0.5)':<{var_w}}{_fmt(metrics.acc_05 * 100, digits):>{_COL_W}}",
            f"{'Balanced accuracy (c=0.5)':<{var_w}}{_fmt(metrics.bal_acc_05 * 100, digits):>{_COL_W}}",
            f"{'Percent correct (c=ybar)':<{var_w}}{_fmt(metrics.acc_ybar * 100, digits):>{_COL_W}}",
            f"{'Percent correct (c* match ybar)':<{var_w}}{_fmt(metrics.acc_match_ybar * 100, digits):>{_COL_W}}",
            f"{'c* (match ybar)':<{var_w}}{_fmt(metrics.c_match_ybar, digits):>{_COL_W}}",
        ]
    else:
        lines += [
            f"{'N':<{var_w}}{fit.nobs:>{_COL_W}}",
            f"{'R\u00b2':<{var_w}}{_fmt(fit.r_squared, digits):>{_COL_W}}",
            f"{'Adj. R\u00b2':<{var_w}}{_fmt(fit.r_squared_adj, digits):>{_COL_W}}",
            f"{'F-stat':<{var_w}}{_fmt(fit.f_stat, digits):>{_COL_W}}",
            f"{'RMSE':<{var_w}}{_fmt(fit.rmse, digits):>{_COL_W}}",
        ]
    lines.append(sep)

    if _is_binary_result(result):
        metrics = _binary_metrics(result)
        lines += [
            "",
            "Classification (c=0.5):",
            f"TP={metrics.tp}  TN={metrics.tn}  FP={metrics.fp}  FN={metrics.fn}",
            f"TPR={_fmt(metrics.tpr, digits)}  TNR={_fmt(metrics.tnr, digits)}  "
            f"PPV={_fmt(metrics.ppv, digits)}  NPV={_fmt(metrics.npv, digits)}",
            "",
            f"Classification (c* match ybar, c*={_fmt(metrics.c_match_ybar, digits)}):",
            f"TP={metrics.tp_star}  TN={metrics.tn_star}  FP={metrics.fp_star}  FN={metrics.fn_star}",
            f"TPR={_fmt(metrics.tpr_star, digits)}  TNR={_fmt(metrics.tnr_star, digits)}  "
            f"PPV={_fmt(metrics.ppv_star, digits)}  NPV={_fmt(metrics.npv_star, digits)}",
        ]
        try:
            ame = _marginal_effects(result, at="overall")
            mem = _marginal_effects(result, at="mean")
            lines.append("")
            lines.append("Marginal effects (AME vs MEM):")
            lines.append(
                f"{'Variable':<{var_w}}"
                f"{'AME':>{_COL_W}}"
                f"{'SE':>{_COL_W}}"
                f"{'p':>{_COL_W}}"
                f"{'MEM':>{_COL_W}}"
                f"{'SE':>{_COL_W}}"
                f"{'p':>{_COL_W}}"
            )
            for name in result.params.index:
                if name == "const":
                    continue
                if name in ame and name in mem:
                    lines.append(
                        f"{name:<{var_w}}"
                        f"{_fmt(ame[name][0], digits):>{_COL_W}}"
                        f"{_fmt(ame[name][1], digits):>{_COL_W}}"
                        f"{_fmt(ame[name][2], digits):>{_COL_W}}"
                        f"{_fmt(mem[name][0], digits):>{_COL_W}}"
                        f"{_fmt(mem[name][1], digits):>{_COL_W}}"
                        f"{_fmt(mem[name][2], digits):>{_COL_W}}"
                    )
        except Exception:
            lines.append("")
            lines.append("Marginal effects: unavailable")

    if stars:
        lines.append("* p<0.1  ** p<0.05  *** p<0.01")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX renderer
# ---------------------------------------------------------------------------


def _latex_table(
    result: RegressionResult,
    stars: bool,
    se_in_parens: bool,
    digits: int,
    title: str | None,
) -> str:
    lines: list[str] = [
        r"\renewcommand{\arraystretch}{1.1}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
    ]
    if title:
        lines.append(r"\multicolumn{5}{l}{\textit{" + title + r"}} \\")
        lines.append(r"\hline")
    lines += [
        r"Variable & Coef & Std Err & $t$ & $p$ \\",
        r"\hline",
    ]

    for name in result.params.index:
        coef = float(result.params[name])
        se = float(result.bse[name])
        tval = float(result.tvalues[name])
        pval = float(result.pvalues[name])
        s = _star(pval) if stars else ""

        coef_str = _fmt(coef, digits) + _latex_star(s)
        lines.append(
            f"{name} & {coef_str} & {_fmt(se, digits)} & "
            f"{_fmt(tval, digits)} & {_fmt(pval, digits)} \\\\"
        )
        if se_in_parens:
            lines.append(f" & $({_fmt(se, digits)})$ & & & \\\\")

    fit = result.fit
    lines.append(r"\hline")
    if _is_binary_result(result):
        metrics = _binary_metrics(result)
        llr = float(getattr(result.raw, "llr", float("nan")))
        llr_pvalue = float(getattr(result.raw, "llr_pvalue", float("nan")))
        lines += [
            f"$N$ & {fit.nobs} & & & \\\\",
            f"Pseudo $R^2$ (McFadden) & {_fmt(metrics.mcfadden_r2, digits)} & & & \\\\",
            f"Corr$^2(y,\\hat p)$ & {_fmt(metrics.r2_corr, digits)} & & & \\\\",
            f"Efron $R^2$ & {_fmt(metrics.r2_efron, digits)} & & & \\\\",
            f"AUC & {_fmt(metrics.auc, digits)} & & & \\\\",
            f"Log-likelihood & {_fmt(fit.log_likelihood, digits)} & & & \\\\",
            f"LR stat & {_fmt(llr, digits)} & & & \\\\",
            f"LR p-value & {_fmt(llr_pvalue, digits)} & & & \\\\",
            f"AIC & {_fmt(fit.aic, digits)} & & & \\\\",
            f"BIC & {_fmt(fit.bic, digits)} & & & \\\\",
            f"Brier score & {_fmt(metrics.brier, digits)} & & & \\\\",
            f"\\% correct (c=0.5) & {_fmt(metrics.acc_05 * 100, digits)} & & & \\\\",
            f"Balanced acc (c=0.5) & {_fmt(metrics.bal_acc_05 * 100, digits)} & & & \\\\",
            f"\\% correct ($c=\\bar{{y}}$) & {_fmt(metrics.acc_ybar * 100, digits)} & & & \\\\",
            f"\\% correct (c* match $\\bar{{y}}$) & {_fmt(metrics.acc_match_ybar * 100, digits)} & & & \\\\",
            f"c* (match $\\bar{{y}}$) & {_fmt(metrics.c_match_ybar, digits)} & & & \\\\",
        ]
    else:
        lines += [
            f"$N$ & {fit.nobs} & & & \\\\",
            f"$R^2$ & {_fmt(fit.r_squared, digits)} & & & \\\\",
            f"Adj.\\ $R^2$ & {_fmt(fit.r_squared_adj, digits)} & & & \\\\",
            f"$F$-stat & {_fmt(fit.f_stat, digits)} & & & \\\\",
            f"RMSE & {_fmt(fit.rmse, digits)} & & & \\\\",
        ]
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    if _is_binary_result(result):
        metrics = _binary_metrics(result)
        lines += [
            r"\vspace{4pt}",
            r"\begin{table}[H]",
            r"\centering",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Classification (c=0.5) & TP & TN & FP & FN \\",
            r"\midrule",
            f"Counts & {metrics.tp} & {metrics.tn} & {metrics.fp} & {metrics.fn} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\begin{table}[H]",
            r"\centering",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Rate & TPR & TNR & PPV & NPV \\",
            r"\midrule",
            f"Value & {_fmt(metrics.tpr, digits)} & {_fmt(metrics.tnr, digits)} & "
            f"{_fmt(metrics.ppv, digits)} & {_fmt(metrics.npv, digits)} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\begin{table}[H]",
            r"\centering",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Classification (c* match $\bar{y}$) & TP & TN & FP & FN \\",
            r"\midrule",
            f"Counts & {metrics.tp_star} & {metrics.tn_star} & {metrics.fp_star} & {metrics.fn_star} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\begin{table}[H]",
            r"\centering",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Rate (c* match $\bar{y}$) & TPR & TNR & PPV & NPV \\",
            r"\midrule",
            f"Value & {_fmt(metrics.tpr_star, digits)} & {_fmt(metrics.tnr_star, digits)} & "
            f"{_fmt(metrics.ppv_star, digits)} & {_fmt(metrics.npv_star, digits)} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        try:
            ame = _marginal_effects(result, at="overall")
            mem = _marginal_effects(result, at="mean")
            lines += [
                r"\begin{table}[H]",
                r"\centering",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                r"Variable & AME & SE & $p$ & MEM & SE & $p$ \\",
                r"\midrule",
            ]
            for name in result.params.index:
                if name == "const":
                    continue
                if name in ame and name in mem:
                    lines.append(
                        f"{name} & {_fmt(ame[name][0], digits)} & {_fmt(ame[name][1], digits)} & "
                        f"{_fmt(ame[name][2], digits)} & {_fmt(mem[name][0], digits)} & "
                        f"{_fmt(mem[name][1], digits)} & {_fmt(mem[name][2], digits)} \\\\"
                    )
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        except Exception:
            lines.append(r"\noindent Marginal effects unavailable.")

    if stars:
        lines.append(r"\vspace{2pt}")
        lines.append(
            r"\begin{flushleft}\footnotesize{* $p<0.1$, ** $p<0.05$, *** $p<0.01$}\end{flushleft}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------


def _html_table(
    result: RegressionResult,
    stars: bool,
    se_in_parens: bool,
    digits: int,
    title: str | None,
) -> str:
    lines: list[str] = [
        "<table>",
    ]
    if title:
        lines.append(f"<caption>{title}</caption>")
    lines += [
        "<tr>"
        "<th>Variable</th><th>Coef</th>"
        "<th>Std Err</th><th><i>t</i></th><th><i>p</i></th>"
        "</tr>",
    ]

    for name in result.params.index:
        coef = float(result.params[name])
        se = float(result.bse[name])
        tval = float(result.tvalues[name])
        pval = float(result.pvalues[name])
        s = _star(pval) if stars else ""

        coef_str = (
            f"{_fmt(coef, digits)}<sup>{s}</sup>" if s else _fmt(coef, digits)
        )
        lines.append(
            f"<tr><td>{name}</td><td>{coef_str}</td>"
            f"<td>{_fmt(se, digits)}</td>"
            f"<td>{_fmt(tval, digits)}</td>"
            f"<td>{_fmt(pval, digits)}</td></tr>"
        )
        if se_in_parens:
            lines.append(
                f"<tr><td></td><td>({_fmt(se, digits)})</td>"
                f"<td></td><td></td><td></td></tr>"
            )

    fit = result.fit
    lines.append("<tr><td colspan='5'><hr></td></tr>")
    if _is_binary_result(result):
        metrics = _binary_metrics(result)
        llr = float(getattr(result.raw, "llr", float("nan")))
        llr_pvalue = float(getattr(result.raw, "llr_pvalue", float("nan")))
        lines += [
            f"<tr><td>N</td><td>{fit.nobs}</td><td></td><td></td><td></td></tr>",
            f"<tr><td>Pseudo R&sup2; (McFadden)</td><td>{_fmt(metrics.mcfadden_r2, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>Corr&sup2;(y,p)</td><td>{_fmt(metrics.r2_corr, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>Efron R&sup2;</td><td>{_fmt(metrics.r2_efron, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>AUC</td><td>{_fmt(metrics.auc, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>Log-likelihood</td><td>{_fmt(fit.log_likelihood, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>LR stat</td><td>{_fmt(llr, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>LR p-value</td><td>{_fmt(llr_pvalue, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>AIC</td><td>{_fmt(fit.aic, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>BIC</td><td>{_fmt(fit.bic, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>Brier score</td><td>{_fmt(metrics.brier, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>% correct (c=0.5)</td><td>{_fmt(metrics.acc_05 * 100, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>Balanced acc (c=0.5)</td><td>{_fmt(metrics.bal_acc_05 * 100, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>% correct (c=ybar)</td><td>{_fmt(metrics.acc_ybar * 100, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>% correct (c* match ybar)</td><td>{_fmt(metrics.acc_match_ybar * 100, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>c* (match ybar)</td><td>{_fmt(metrics.c_match_ybar, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
        ]
    else:
        lines += [
            f"<tr><td>N</td><td>{fit.nobs}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>R&sup2;</td><td>{_fmt(fit.r_squared, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>Adj. R&sup2;</td><td>{_fmt(fit.r_squared_adj, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>F-stat</td><td>{_fmt(fit.f_stat, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
            f"<tr><td>RMSE</td><td>{_fmt(fit.rmse, digits)}</td>"
            f"<td></td><td></td><td></td></tr>",
        ]
    lines.append("</table>")

    if _is_binary_result(result):
        metrics = _binary_metrics(result)
        lines += [
            "<table>",
            "<tr><th>Classification (c=0.5)</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr>",
            f"<tr><td>Counts</td><td>{metrics.tp}</td><td>{metrics.tn}</td><td>{metrics.fp}</td><td>{metrics.fn}</td></tr>",
            "</table>",
            "<table>",
            "<tr><th>Rate</th><th>TPR</th><th>TNR</th><th>PPV</th><th>NPV</th></tr>",
            f"<tr><td>Value</td><td>{_fmt(metrics.tpr, digits)}</td><td>{_fmt(metrics.tnr, digits)}</td>"
            f"<td>{_fmt(metrics.ppv, digits)}</td><td>{_fmt(metrics.npv, digits)}</td></tr>",
            "</table>",
            "<table>",
            "<tr><th>Classification (c* match ybar)</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr>",
            f"<tr><td>Counts</td><td>{metrics.tp_star}</td><td>{metrics.tn_star}</td><td>{metrics.fp_star}</td><td>{metrics.fn_star}</td></tr>",
            "</table>",
            "<table>",
            "<tr><th>Rate (c* match ybar)</th><th>TPR</th><th>TNR</th><th>PPV</th><th>NPV</th></tr>",
            f"<tr><td>Value</td><td>{_fmt(metrics.tpr_star, digits)}</td><td>{_fmt(metrics.tnr_star, digits)}</td>"
            f"<td>{_fmt(metrics.ppv_star, digits)}</td><td>{_fmt(metrics.npv_star, digits)}</td></tr>",
            "</table>",
        ]
        try:
            ame = _marginal_effects(result, at="overall")
            mem = _marginal_effects(result, at="mean")
            lines += [
                "<table>",
                "<tr><th>Variable</th><th>AME</th><th>SE</th><th>p</th>"
                "<th>MEM</th><th>SE</th><th>p</th></tr>",
            ]
            for name in result.params.index:
                if name == "const":
                    continue
                if name in ame and name in mem:
                    lines.append(
                        f"<tr><td>{name}</td><td>{_fmt(ame[name][0], digits)}</td>"
                        f"<td>{_fmt(ame[name][1], digits)}</td>"
                        f"<td>{_fmt(ame[name][2], digits)}</td>"
                        f"<td>{_fmt(mem[name][0], digits)}</td>"
                        f"<td>{_fmt(mem[name][1], digits)}</td>"
                        f"<td>{_fmt(mem[name][2], digits)}</td></tr>"
                    )
            lines.append("</table>")
        except Exception:
            lines.append("<p>Marginal effects unavailable.</p>")

    if stars:
        lines.append(
            "<p><small>* p&lt;0.1, ** p&lt;0.05, *** p&lt;0.01</small></p>"
        )

    return "\n".join(lines)
