"""Command-line interface for econtools data workflows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from econtools.data.clean import snake_case
from econtools.data.io import load_raw, save_curated, versioned_path
from econtools.inference.se_types import VALID_COV_TYPES
from econtools.models.iv import fit_iv_2sls
from econtools.models.ols import fit_ols, fit_ols_formula
from econtools.models.panel import fit_first_difference
from econtools.models.probit import fit_probit
from econtools.diagnostics import run_iv_diagnostics, run_panel_diagnostics
from econtools.tables import (
    combine_tex,
    compare_table,
    compile_tex_to_pdf,
    diagnostic_table,
    open_pdf,
    reg_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RAW_ROOT = _REPO_ROOT / "data_lake" / "raw"
_CURATED_ROOT = _REPO_ROOT / "data_lake" / "curated"
_PROJECTS_ROOT = _REPO_ROOT / "projects"
_PROJECT_FILE = _REPO_ROOT / ".econ_project"


def _list_raw_datasets(source: str) -> list[str]:
    raw_dir = _RAW_ROOT / source
    if not raw_dir.exists():
        return []
    names = []
    for path in raw_dir.iterdir():
        if path.suffix.lower() in {".dta", ".csv"}:
            names.append(path.stem)
    return sorted(set(names))


def _find_datasets_with_columns(source: str, columns: list[str]) -> list[str]:
    raw_dir = _RAW_ROOT / source
    if not raw_dir.exists():
        return []
    need = {c.lower() for c in columns}
    matches: list[str] = []
    for path in raw_dir.iterdir():
        if path.suffix.lower() not in {".dta", ".csv"}:
            continue
        try:
            if path.suffix.lower() == ".dta":
                df = pd.read_stata(
                    path,
                    convert_categoricals=False,
                    preserve_dtypes=False,
                    convert_missing=False,
                )
            else:
                df = pd.read_csv(path, nrows=5)
            cols = {c.lower() for c in df.columns}
            if need.issubset(cols):
                matches.append(path.stem)
        except Exception:
            continue
    return sorted(set(matches))


def _load_curated(name: str, version: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet_path, meta_path = versioned_path(name, version, base_dir=_CURATED_ROOT)
    if not parquet_path.exists():
        raise FileNotFoundError(str(parquet_path))
    df = pd.read_parquet(parquet_path)
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return df, meta


def _resolve_version(version: str | int) -> str:
    if isinstance(version, int):
        return f"v{version}"
    if version.startswith("v"):
        return version
    return f"v{version}"


def _project_output_dir(project: str, session: str | None = None) -> Path:
    project_dir = _PROJECTS_ROOT / project
    output_dir = project_dir / "output"
    if session:
        output_dir = output_dir / session
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _active_project() -> str | None:
    env = os.environ.get("ECON_PROJECT")
    if env:
        return env
    if _PROJECT_FILE.exists():
        name = _PROJECT_FILE.read_text(encoding="utf-8").strip()
        return name if name else None
    return None


def _safe_label(label: str) -> str:
    out = []
    for ch in label.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    return "".join(out).strip("_") or "output"


def _print_describe(df: pd.DataFrame, meta: dict[str, Any], title: str) -> None:
    console = Console()
    console.print(f"[bold]{title}[/bold]")
    console.print(f"Observations: {len(df)}")
    console.print(f"Variables: {len(df.columns)}")

    labels = meta.get("variable_labels", {}) if meta else {}
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    table = Table(show_header=True, header_style="bold")
    table.add_column("variable", style="cyan", no_wrap=True)
    table.add_column("dtype", style="magenta")
    table.add_column("label")
    table.add_column("missing", justify="right")
    table.add_column("% missing", justify="right")

    for col in df.columns:
        label = labels.get(col, "")
        table.add_row(
            str(col),
            str(df[col].dtype),
            str(label),
            str(int(missing[col])),
            f"{missing_pct[col]:.2f}",
        )

    console.print(table)


def _print_summary(df: pd.DataFrame, cols: list[str] | None) -> None:
    console = Console()
    if cols is None:
        data = df.select_dtypes(include="number")
    else:
        data = df[cols]

    if data.shape[1] == 0:
        console.print("No numeric columns to summarise.")
        return

    summary = data.agg(["count", "mean", "std", "min", "max"]).T
    table = Table(show_header=True, header_style="bold")
    table.add_column("variable", style="cyan", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("sd", justify="right")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")

    for col, row in summary.iterrows():
        table.add_row(
            str(col),
            f"{int(row['count'])}",
            f"{row['mean']:.4f}" if pd.notna(row["mean"]) else "",
            f"{row['std']:.4f}" if pd.notna(row["std"]) else "",
            f"{row['min']:.4f}" if pd.notna(row["min"]) else "",
            f"{row['max']:.4f}" if pd.notna(row["max"]) else "",
        )

    console.print(table)


def _describe_table_data(df: pd.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    labels = meta.get("variable_labels", {}) if meta else {}
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        rows.append(
            {
                "variable": str(col),
                "dtype": str(df[col].dtype),
                "label": str(labels.get(col, "")),
                "missing": int(missing[col]),
                "missing_pct": float(missing_pct[col]),
            }
        )
    return rows


def _summary_table_data(df: pd.DataFrame, cols: list[str] | None) -> list[dict[str, Any]]:
    if cols is None:
        data = df.select_dtypes(include="number")
    else:
        data = df[cols]
    if data.shape[1] == 0:
        return []
    summary = data.agg(["count", "mean", "std", "min", "max"]).T
    rows: list[dict[str, Any]] = []
    for col, row in summary.iterrows():
        rows.append(
            {
                "variable": str(col),
                "n": int(row["count"]),
                "mean": float(row["mean"]) if pd.notna(row["mean"]) else None,
                "sd": float(row["std"]) if pd.notna(row["std"]) else None,
                "min": float(row["min"]) if pd.notna(row["min"]) else None,
                "max": float(row["max"]) if pd.notna(row["max"]) else None,
            }
        )
    return rows


def _render_table_text(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [sep]
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    lines.append(header_row)
    lines.append(sep)
    for row in rows:
        line = "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
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


def _render_table_latex(headers: list[str], rows: list[list[str]]) -> str:
    if headers == ["variable", "dtype", "label", "missing", "% missing"]:
        cols = (
            r">{\raggedright\arraybackslash}p{0.20\textwidth} "
            r">{\raggedleft\arraybackslash}r "
            r">{\raggedright\arraybackslash}p{0.45\textwidth} "
            r">{\raggedleft\arraybackslash}r "
            r">{\raggedleft\arraybackslash}r"
        )
    elif headers == ["variable", "N", "mean", "sd", "min", "max"]:
        cols = (
            r">{\raggedright\arraybackslash}p{0.24\textwidth} "
            r">{\raggedleft\arraybackslash}r "
            r">{\raggedleft\arraybackslash}r "
            r">{\raggedleft\arraybackslash}r "
            r">{\raggedleft\arraybackslash}r "
            r">{\raggedleft\arraybackslash}r"
        )
    else:
        cols = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{" + cols + r"}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(c) for c in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _render_table_html(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["<table>"]
    lines.append("<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _format_describe(
    df: pd.DataFrame, meta: dict[str, Any], format: str
) -> str:
    rows = _describe_table_data(df, meta)
    headers = ["variable", "dtype", "label", "missing", "% missing"]
    data_rows = [
        [
            r["variable"],
            r["dtype"],
            r["label"],
            str(r["missing"]),
            f"{r['missing_pct']:.2f}",
        ]
        for r in rows
    ]
    if format == "text":
        return _render_table_text(headers, data_rows)
    if format == "latex":
        return _render_table_latex(headers, data_rows)
    if format == "html":
        return _render_table_html(headers, data_rows)
    raise ValueError(f"Unknown format '{format}'. Choose from 'text', 'latex', 'html'.")


def _format_summary(
    df: pd.DataFrame, cols: list[str] | None, format: str
) -> str:
    rows = _summary_table_data(df, cols)
    headers = ["variable", "N", "mean", "sd", "min", "max"]
    data_rows = []
    for r in rows:
        data_rows.append(
            [
                r["variable"],
                str(r["n"]),
                "" if r["mean"] is None else f"{r['mean']:.4f}",
                "" if r["sd"] is None else f"{r['sd']:.4f}",
                "" if r["min"] is None else f"{r['min']:.4f}",
                "" if r["max"] is None else f"{r['max']:.4f}",
            ]
        )
    if format == "text":
        return _render_table_text(headers, data_rows)
    if format == "latex":
        return _render_table_latex(headers, data_rows)
    if format == "html":
        return _render_table_html(headers, data_rows)
    raise ValueError(f"Unknown format '{format}'. Choose from 'text', 'latex', 'html'.")


def _write_output(
    content: str,
    project: str | None,
    out_path: str | None,
    default_name: str,
    format: str,
    session: str | None = None,
) -> Path:
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
    if project:
        output_dir = _project_output_dir(project, session=session)
        ext = "tex" if format == "latex" else ("html" if format == "html" else "txt")
        path = output_dir / f"{default_name}.{ext}"
        path.write_text(content, encoding="utf-8")
        return path
    raise ValueError("Provide --project or --out to export output.")


def _compile_latex_to_pdf(tex_path: Path, engine: str | None = None) -> Path:
    tex_path = tex_path.resolve()
    engine = engine or os.environ.get("ECON_LATEX_ENGINE") or ""
    candidates = [engine] if engine else ["pdflatex", "latexmk", "tectonic"]
    cmd = None
    for name in candidates:
        if not name:
            continue
        exe = shutil.which(name)
        if exe:
            cmd = name
            break
    if not cmd:
        raise RuntimeError(
            "No LaTeX engine found. Install latexmk/pdflatex/tectonic or set ECON_LATEX_ENGINE."
        )
    workdir = tex_path.parent
    content = tex_path.read_text(encoding="utf-8")
    needs_wrapper = "\\begin{document}" not in content and "\\documentclass" not in content
    compile_path = tex_path
    wrapper_path = None
    if needs_wrapper:
        wrapper_path = tex_path.with_name(tex_path.stem + "_wrapper.tex")
        wrapper = (
            "\\documentclass{article}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\usepackage{array}\n"
            "\\usepackage{longtable}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{float}\n"
            "\\begin{document}\n"
            "\\footnotesize\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            "\\renewcommand{\\arraystretch}{0.9}\n"
            f"{content}\n"
            "\\end{document}\n"
        )
        wrapper_path.write_text(wrapper, encoding="utf-8")
        compile_path = wrapper_path
    def _run_engine(name: str) -> None:
        if name == "latexmk":
            argv = [name, "-pdf", "-interaction=nonstopmode", compile_path.name]
        elif name == "tectonic":
            argv = [name, compile_path.name]
        else:
            argv = [name, "-interaction=nonstopmode", compile_path.name]
        import subprocess
        subprocess.run(argv, cwd=str(workdir), check=False)

    try:
        _run_engine(cmd)
    except Exception:
        # Fallback: try pdflatex directly if latexmk/tectonic fails
        if cmd != "pdflatex" and shutil.which("pdflatex"):
            _run_engine("pdflatex")
        else:
            raise

    pdf_path = compile_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError("PDF compilation failed; see LaTeX log for details.")
    target_pdf = tex_path.with_suffix(".pdf")
    if pdf_path != target_pdf:
        pdf_path.replace(target_pdf)
    if wrapper_path and wrapper_path.exists():
        try:
            wrapper_path.unlink()
        except OSError:
            pass
    return target_pdf


def cmd_curate(args: argparse.Namespace) -> int:
    df, meta = load_raw(args.name, source=args.source, verify=not args.no_verify)

    transformations = []
    if args.snake_case:
        rename_map = {c: snake_case(c) for c in df.columns}
        df = df.rename(columns=rename_map)
        transformations.append("rename_snake")
        if meta.get("variable_labels"):
            meta["variable_labels"] = {
                rename_map.get(k, k): v for k, v in meta["variable_labels"].items()
            }
        if meta.get("stata_dtypes"):
            meta["stata_dtypes"] = {
                rename_map.get(k, k): v for k, v in meta["stata_dtypes"].items()
            }
        meta["column_rename_map"] = rename_map

    if transformations:
        meta["transformations"] = transformations

    version = _resolve_version(args.version)
    parquet_path, meta_path = save_curated(df, args.name, version, meta)

    console = Console()
    console.print("Curated dataset written:")
    console.print(f"  {parquet_path}")
    console.print(f"  {meta_path}")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    title = f"{args.name}"
    try:
        if args.curated:
            version = _resolve_version(args.version)
            df, meta = _load_curated(args.name, version)
            title = f"{args.name} ({version}, curated)"
        else:
            df, meta = load_raw(args.name, source=args.source, verify=not args.no_verify)
            title = f"{args.name} (raw)"
    except FileNotFoundError:
        if args.curated:
            Console().print("Curated dataset not found.")
            return 1
        available = _list_raw_datasets(args.source)
        Console().print(f"Dataset '{args.name}' not found in raw/{args.source}.")
        if available:
            Console().print("Available datasets:")
            Console().print(", ".join(available))
        return 1

    project = args.project or _active_project()
    export_format = args.format
    if args.export and args.export in {"text", "latex", "html"}:
        export_format = args.export
    if args.export:
        content = _format_describe(df, meta, export_format)
        if args.label:
            if export_format == "latex":
                content = f"\\section*{{{args.label}}}\n{content}"
            elif export_format == "html":
                content = f"<h3>{args.label}</h3>\n{content}"
            else:
                content = f"{args.label}\n\n{content}"
        label_slug = _safe_label(args.label) if args.label else None
        path = _write_output(
            content,
            project=project,
            out_path=args.out,
            default_name=f"{args.name}_{label_slug}_des" if label_slug else f"{args.name}_des",
            format=export_format,
            session=args.session,
        )
        if args.pdf:
            if export_format != "latex":
                raise ValueError("--pdf requires --format latex (or --export latex).")
            pdf_path = _compile_latex_to_pdf(path)
            Console().print(f"Wrote {pdf_path}")
        Console().print(f"Wrote {path}")
        return 0
    _print_describe(df, meta, title)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    try:
        if args.curated:
            version = _resolve_version(args.version)
            df, _meta = _load_curated(args.name, version)
        else:
            df, _meta = load_raw(args.name, source=args.source, verify=not args.no_verify)
    except FileNotFoundError:
        if args.curated:
            Console().print("Curated dataset not found.")
            return 1
        available = _list_raw_datasets(args.source)
        Console().print(f"Dataset '{args.name}' not found in raw/{args.source}.")
        if available:
            Console().print("Available datasets:")
            Console().print(", ".join(available))
        return 1

    cols = None
    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    project = args.project or _active_project()
    export_format = args.format
    if args.export and args.export in {"text", "latex", "html"}:
        export_format = args.export
    if args.export:
        content = _format_summary(df, cols, export_format)
        if args.label:
            if export_format == "latex":
                content = f"\\section*{{{args.label}}}\n{content}"
            elif export_format == "html":
                content = f"<h3>{args.label}</h3>\n{content}"
            else:
                content = f"{args.label}\n\n{content}"
        label_slug = _safe_label(args.label) if args.label else None
        path = _write_output(
            content,
            project=project,
            out_path=args.out,
            default_name=f"{args.name}_{label_slug}_summ" if label_slug else f"{args.name}_summ",
            format=export_format,
            session=args.session,
        )
        if args.pdf:
            if export_format != "latex":
                raise ValueError("--pdf requires --format latex (or --export latex).")
            pdf_path = _compile_latex_to_pdf(path)
            Console().print(f"Wrote {pdf_path}")
        Console().print(f"Wrote {path}")
        return 0
    _print_summary(df, cols)
    return 0


def cmd_regress(args: argparse.Namespace) -> int:
    try:
        if args.curated:
            version = _resolve_version(args.version)
            df, _meta = _load_curated(args.name, version)
        else:
            df, _meta = load_raw(args.name, source=args.source, verify=not args.no_verify)
    except FileNotFoundError:
        if args.curated:
            Console().print("Curated dataset not found.")
            return 1
        available = _list_raw_datasets(args.source)
        Console().print(f"Dataset '{args.name}' not found in raw/{args.source}.")
        if available:
            Console().print("Available datasets:")
            Console().print(", ".join(available))
        return 1

    groups = None
    if args.cov_type == "cluster":
        if not args.cluster:
            Console().print("cov_type='cluster' requires --cluster.")
            return 1
        if args.cluster not in df.columns:
            Console().print(f"Cluster variable '{args.cluster}' not found.")
            return 1
        groups = df[args.cluster]

    endog_vars: list[str] = []
    instr_vars: list[str] = []

    if args.panel == "fd":
        if args.formula:
            Console().print("--panel fd does not support --formula.")
            return 1
        if not args.entity or not args.time:
            Console().print("--panel fd requires --entity and --time.")
            return 1
        if args.endog or args.instruments or args.compare_ols:
            Console().print("--panel fd does not support IV or compare-ols.")
            return 1
        if not args.exog_vars:
            Console().print("Provide at least one regressor variable.")
            return 1
        missing = [c for c in [args.dep_var, *args.exog_vars, args.entity, args.time] if c not in df.columns]
        if missing:
            Console().print("Missing columns: " + ", ".join(missing))
            return 1
        result = fit_first_difference(
            df,
            args.dep_var,
            args.exog_vars,
            entity=args.entity,
            time=args.time,
            add_constant=False,
            cov_type="unadjusted",
        )
    else:
        if args.endog:
            endog_vars = [c.strip() for c in args.endog.split(",") if c.strip()]
        if args.instruments:
            instr_vars = [c.strip() for c in args.instruments.split(",") if c.strip()]

        if (endog_vars or instr_vars) and args.formula:
            Console().print("IV mode does not support --formula (use dep_var/exog_vars instead).")
            return 1
        if (endog_vars and not instr_vars) or (instr_vars and not endog_vars):
            Console().print("IV mode requires both --endog and --instruments.")
            return 1

        if args.formula:
            result = fit_ols_formula(
                df,
                args.formula,
                cov_type=args.cov_type,
                maxlags=args.maxlags,
                groups=groups,
            )
        elif endog_vars:
            if not args.exog_vars:
                Console().print("Provide at least one exogenous regressor.")
                return 1
            missing = [
                c for c in [args.dep_var, *args.exog_vars, *endog_vars, *instr_vars]
                if c not in df.columns
            ]
            if missing:
                Console().print("Missing columns: " + ", ".join(missing))
                return 1
            result = fit_iv_2sls(
                df,
                args.dep_var,
                args.exog_vars,
                endog_vars,
                instr_vars,
                add_constant=args.add_constant,
                cov_type=args.cov_type,
                maxlags=args.maxlags,
                groups=groups,
            )
        else:
            if not args.exog_vars:
                Console().print("Provide at least one regressor variable.")
                return 1
            missing = [c for c in [args.dep_var, *args.exog_vars] if c not in df.columns]
            if missing:
                Console().print("Missing columns: " + ", ".join(missing))
                return 1
            result = fit_ols(
                df,
                args.dep_var,
                args.exog_vars,
                add_constant=args.add_constant,
                cov_type=args.cov_type,
                maxlags=args.maxlags,
                groups=groups,
            )

    project = args.project or _active_project()
    export_format = args.format
    if args.export and args.export in {"text", "latex", "html"}:
        export_format = args.export
    if args.compare_ols:
        if not endog_vars:
            Console().print("--compare-ols requires IV mode (--endog and --instruments).")
            return 1
        ols_result = fit_ols(
            df,
            args.dep_var,
            args.exog_vars + endog_vars,
            add_constant=args.add_constant,
            cov_type=args.cov_type,
            maxlags=args.maxlags,
            groups=groups,
        )
        labels = ["OLS", "2SLS"]
        content = compare_table(
            [ols_result, result],
            labels=labels,
            stars=not args.no_stars,
            se_in_parens=not args.no_se_parens,
            digits=args.digits,
            format=export_format,
            title=args.label,
        )
    else:
        content = reg_table(
            result,
            stars=not args.no_stars,
            se_in_parens=not args.no_se_parens,
            digits=args.digits,
            format=export_format,
            title=args.label,
        )
    if args.export:
        label_slug = _safe_label(args.label) if args.label else None
        path = _write_output(
            content,
            project=project,
            out_path=args.out,
            default_name=(
                f"{args.name}_{label_slug}_reg" if label_slug else f"{args.name}_reg"
            ),
            format=export_format,
            session=args.session,
        )
        if args.pdf:
            if export_format != "latex":
                raise ValueError("--pdf requires --format latex (or --export latex).")
            pdf_path = _compile_latex_to_pdf(path)
            Console().print(f"Wrote {pdf_path}")
        Console().print(f"Wrote {path}")
    else:
        Console().print(content)

    if args.diagnostics or args.diag_tests:
        if not endog_vars:
            if args.panel == "fd":
                if not args.entity or not args.time:
                    Console().print("--panel fd diagnostics require --entity and --time.")
                    return 1
                tests = None
                if args.diag_tests:
                    tests = [t.strip() for t in args.diag_tests.split(",") if t.strip()]
                diag_results = run_panel_diagnostics(
                    df,
                    args.dep_var,
                    args.exog_vars,
                    entity=args.entity,
                    time=args.time,
                    tests=tests,
                    leads=args.lead_k,
                )
            else:
                Console().print("Diagnostics currently only supported for IV/2SLS.")
                return 1
        else:
            tests = None
            if args.diag_tests:
                tests = [t.strip() for t in args.diag_tests.split(",") if t.strip()]
            diag_results = run_iv_diagnostics(result, tests=tests)
        if not diag_results:
            Console().print("No diagnostics run (check test names).")
            return 1
        diag_content = diagnostic_table(
            diag_results,
            digits=args.digits,
            format=export_format,
            title=args.label if args.label else "Diagnostics",
        )
        if args.export:
            label_slug = _safe_label(args.label) if args.label else None
            diag_path = _write_output(
                diag_content,
                project=project,
                out_path=None,
                default_name=(
                    f"{args.name}_{label_slug}_diag" if label_slug else f"{args.name}_diag"
                ),
                format=export_format,
                session=args.session,
            )
            if args.pdf:
                if export_format != "latex":
                    raise ValueError("--pdf requires --format latex (or --export latex).")
                pdf_path = _compile_latex_to_pdf(diag_path)
                Console().print(f"Wrote {pdf_path}")
            Console().print(f"Wrote {diag_path}")
        else:
            Console().print(diag_content)

    return 0


def cmd_probit(args: argparse.Namespace) -> int:
    try:
        if args.curated:
            version = _resolve_version(args.version)
            df, _meta = _load_curated(args.name, version)
        else:
            df, _meta = load_raw(args.name, source=args.source, verify=not args.no_verify)
    except FileNotFoundError:
        if args.curated:
            Console().print("Curated dataset not found.")
            return 1
        available = _list_raw_datasets(args.source)
        Console().print(f"Dataset '{args.name}' not found in raw/{args.source}.")
        if available:
            Console().print("Available datasets:")
            Console().print(", ".join(available))
        return 1

    groups = None
    if args.cov_type == "cluster":
        if not args.cluster:
            Console().print("cov_type='cluster' requires --cluster.")
            return 1
        if args.cluster not in df.columns:
            Console().print(f"Cluster variable '{args.cluster}' not found.")
            return 1
        groups = df[args.cluster]

    if not args.exog_vars:
        Console().print("Provide at least one regressor variable.")
        return 1
    missing = [c for c in [args.dep_var, *args.exog_vars] if c not in df.columns]
    if missing:
        Console().print("Missing columns: " + ", ".join(missing))
        return 1

    result = fit_probit(
        df,
        args.dep_var,
        args.exog_vars,
        add_constant=args.add_constant,
        cov_type=args.cov_type,
        maxlags=args.maxlags,
        groups=groups,
    )

    project = args.project or _active_project()
    export_format = args.format
    if args.export and args.export in {"text", "latex", "html"}:
        export_format = args.export

    content = reg_table(
        result,
        stars=not args.no_stars,
        se_in_parens=not args.no_se_parens,
        digits=args.digits,
        format=export_format,
        title=args.label,
    )
    if args.export:
        label_slug = _safe_label(args.label) if args.label else None
        path = _write_output(
            content,
            project=project,
            out_path=args.out,
            default_name=(
                f"{args.name}_{label_slug}_probit" if label_slug else f"{args.name}_probit"
            ),
            format=export_format,
            session=args.session,
        )
        if args.pdf:
            if export_format != "latex":
                raise ValueError("--pdf requires --format latex (or --export latex).")
            pdf_path = _compile_latex_to_pdf(path)
            Console().print(f"Wrote {pdf_path}")
        Console().print(f"Wrote {path}")
    else:
        Console().print(content)

    return 0


def cmd_findcols(args: argparse.Namespace) -> int:
    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    if not cols:
        Console().print("Provide one or more columns (comma-separated).")
        return 1
    matches = _find_datasets_with_columns(args.source, cols)
    if not matches:
        Console().print("No datasets matched.")
        return 0
    Console().print("Matches:")
    Console().print(", ".join(matches))
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    console = Console()
    if args.name:
        _PROJECT_FILE.write_text(args.name.strip(), encoding="utf-8")
        console.print(f"Active project: {args.name.strip()}")
        return 0
    active = _active_project()
    if active:
        console.print(f"Active project: {active}")
    else:
        console.print("Active project: (none)")
    return 0


def cmd_texpdf(args: argparse.Namespace) -> int:
    tex_paths = [Path(p) for p in args.files]
    for path in tex_paths:
        if not path.exists():
            Console().print(f"File not found: {path}")
            return 1
    out_path = Path(args.out)
    if out_path.suffix.lower() != ".pdf":
        out_path = out_path.with_suffix(".pdf")
    out_tex = out_path.with_suffix(".tex")

    combine_tex(tex_paths, out_tex)
    pdf_path = compile_tex_to_pdf(out_tex, engine=args.engine, workdir=out_tex.parent)
    Console().print(f"Wrote {pdf_path}")
    if not args.no_open:
        open_pdf(pdf_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="econ",
        description="econtools CLI for data pipelines and inspection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    curate = sub.add_parser("curate", help="Convert raw dataset to curated Parquet.")
    curate.add_argument("name", help="Dataset name (e.g. crime4)")
    curate.add_argument("--source", default="wooldridge_and_oleg", help="Raw source folder")
    curate.add_argument("--version", default="1", help="Curated version (default: 1)")
    curate.add_argument("--no-verify", action="store_true", help="Skip manifest hash verification")
    curate.add_argument(
        "--no-snake",
        dest="snake_case",
        action="store_false",
        help="Do not rename columns to snake_case",
    )
    curate.set_defaults(func=cmd_curate, snake_case=True)

    describe = sub.add_parser("des", help="Describe dataset (Stata-like).")
    describe.add_argument("name", help="Dataset name")
    describe.add_argument("--source", default="wooldridge_and_oleg", help="Raw source folder")
    describe.add_argument("--curated", action="store_true", help="Load from curated")
    describe.add_argument("--version", default="1", help="Curated version (default: 1)")
    describe.add_argument("--no-verify", action="store_true", help="Skip manifest hash verification")
    describe.add_argument(
        "--export",
        nargs="?",
        const="text",
        help="Write output to a file (optionally specify format: text, latex, html)",
    )
    describe.add_argument("--format", default="text", help="Output format: text, latex, html")
    describe.add_argument("--out", help="Output filepath (overrides --project)")
    describe.add_argument("--project", help="Project name (writes to projects/<name>/output/)")
    describe.add_argument("--session", help="Output session subfolder under project output")
    describe.add_argument("--label", help="Optional label/title for exported output")
    describe.add_argument("--pdf", action="store_true", help="Compile LaTeX to PDF (requires LaTeX engine)")
    describe.set_defaults(func=cmd_describe)

    summary = sub.add_parser("summ", help="Summarise numeric columns (Stata-like).")
    summary.add_argument("name", help="Dataset name")
    summary.add_argument("--source", default="wooldridge_and_oleg", help="Raw source folder")
    summary.add_argument("--curated", action="store_true", help="Load from curated")
    summary.add_argument("--version", default="1", help="Curated version (default: 1)")
    summary.add_argument("--no-verify", action="store_true", help="Skip manifest hash verification")
    summary.add_argument("--cols", help="Comma-separated list of columns to summarise")
    summary.add_argument(
        "--export",
        nargs="?",
        const="text",
        help="Write output to a file (optionally specify format: text, latex, html)",
    )
    summary.add_argument("--format", default="text", help="Output format: text, latex, html")
    summary.add_argument("--out", help="Output filepath (overrides --project)")
    summary.add_argument("--project", help="Project name (writes to projects/<name>/output/)")
    summary.add_argument("--session", help="Output session subfolder under project output")
    summary.add_argument("--label", help="Optional label/title for exported output")
    summary.add_argument("--pdf", action="store_true", help="Compile LaTeX to PDF (requires LaTeX engine)")
    summary.set_defaults(func=cmd_summary)

    regress = sub.add_parser("reg", help="Run OLS regression (Stata-like).")
    regress.add_argument("name", help="Dataset name")
    regress.add_argument("dep_var", help="Dependent variable")
    regress.add_argument("exog_vars", nargs="*", help="Regressor variables")
    regress.add_argument("--formula", help="Patsy formula (overrides dep_var/exog_vars)")
    regress.add_argument("--endog", help="Comma-separated endogenous regressors (IV/2SLS)")
    regress.add_argument("--instruments", help="Comma-separated instruments (IV/2SLS)")
    regress.add_argument("--panel", choices=["fd"], help="Panel estimator (fd = first differences)")
    regress.add_argument("--entity", help="Panel entity column (required for --panel fd)")
    regress.add_argument("--time", help="Panel time column (required for --panel fd)")
    regress.add_argument("--source", default="wooldridge_and_oleg", help="Raw source folder")
    regress.add_argument("--curated", action="store_true", help="Load from curated")
    regress.add_argument("--version", default="1", help="Curated version (default: 1)")
    regress.add_argument("--no-verify", action="store_true", help="Skip manifest hash verification")
    regress.add_argument(
        "--no-const",
        dest="add_constant",
        action="store_false",
        help="Do not add an intercept",
    )
    regress.add_argument(
        "--cov-type",
        default="classical",
        choices=VALID_COV_TYPES,
        help="Covariance type",
    )
    regress.add_argument("--maxlags", type=int, help="Max lags for HAC/Newey-West SEs")
    regress.add_argument("--cluster", help="Cluster variable (requires cov-type cluster)")
    regress.add_argument("--digits", type=int, default=3, help="Decimal places")
    regress.add_argument("--no-stars", action="store_true", help="Disable significance stars")
    regress.add_argument(
        "--no-se-parens",
        action="store_true",
        help="Do not print standard errors in parentheses",
    )
    regress.add_argument(
        "--compare-ols",
        action="store_true",
        help="When using IV, also compute OLS and output a comparison table",
    )
    regress.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run IV diagnostics and output a diagnostics table",
    )
    regress.add_argument(
        "--diag-tests",
        help="Comma-separated diagnostic tests to run (e.g., wu_hausman)",
    )
    regress.add_argument(
        "--lead-k",
        type=int,
        default=1,
        help="Number of leads for panel lead-test diagnostics",
    )
    regress.add_argument(
        "--export",
        nargs="?",
        const="text",
        help="Write output to a file (optionally specify format: text, latex, html)",
    )
    regress.add_argument("--format", default="text", help="Output format: text, latex, html")
    regress.add_argument("--out", help="Output filepath (overrides --project)")
    regress.add_argument("--project", help="Project name (writes to projects/<name>/output/)")
    regress.add_argument("--session", help="Output session subfolder under project output")
    regress.add_argument("--label", help="Optional label/title for output")
    regress.add_argument("--pdf", action="store_true", help="Compile LaTeX to PDF (requires LaTeX engine)")
    regress.set_defaults(func=cmd_regress, add_constant=True)

    probit = sub.add_parser("probit", help="Run Probit regression (binary outcome).")
    probit.add_argument("name", help="Dataset name")
    probit.add_argument("dep_var", help="Dependent variable")
    probit.add_argument("exog_vars", nargs="*", help="Regressor variables")
    probit.add_argument("--source", default="wooldridge_and_oleg", help="Raw source folder")
    probit.add_argument("--curated", action="store_true", help="Load from curated")
    probit.add_argument("--version", default="1", help="Curated version (default: 1)")
    probit.add_argument("--no-verify", action="store_true", help="Skip manifest hash verification")
    probit.add_argument(
        "--no-const",
        dest="add_constant",
        action="store_false",
        help="Do not add an intercept",
    )
    probit.add_argument(
        "--cov-type",
        default="classical",
        choices=VALID_COV_TYPES,
        help="Covariance type",
    )
    probit.add_argument("--maxlags", type=int, help="Max lags for HAC/Newey-West SEs")
    probit.add_argument("--cluster", help="Cluster variable (requires cov-type cluster)")
    probit.add_argument("--digits", type=int, default=3, help="Decimal places")
    probit.add_argument("--no-stars", action="store_true", help="Disable significance stars")
    probit.add_argument(
        "--no-se-parens",
        action="store_true",
        help="Do not print standard errors in parentheses",
    )
    probit.add_argument(
        "--export",
        nargs="?",
        const="text",
        help="Write output to a file (optionally specify format: text, latex, html)",
    )
    probit.add_argument("--format", default="text", help="Output format: text, latex, html")
    probit.add_argument("--out", help="Output filepath (overrides --project)")
    probit.add_argument("--project", help="Project name (writes to projects/<name>/output/)")
    probit.add_argument("--session", help="Output session subfolder under project output")
    probit.add_argument("--label", help="Optional label/title for output")
    probit.add_argument("--pdf", action="store_true", help="Compile LaTeX to PDF (requires LaTeX engine)")
    probit.set_defaults(func=cmd_probit, add_constant=True)

    findcols = sub.add_parser("findcols", help="Find raw datasets by column names.")
    findcols.add_argument("columns", help="Comma-separated columns to match")
    findcols.add_argument("--source", default="wooldridge_and_oleg", help="Raw source folder")
    findcols.set_defaults(func=cmd_findcols)

    project = sub.add_parser("project", help="Show or set active project.")
    project.add_argument("name", nargs="?", help="Project name to set as active")
    project.set_defaults(func=cmd_project)

    texpdf = sub.add_parser("texpdf", help="Combine .tex files, compile to PDF, and optionally open.")
    texpdf.add_argument("files", nargs="+", help="Input .tex files (paths)")
    texpdf.add_argument("--out", required=True, help="Output PDF path (or base path without .pdf)")
    texpdf.add_argument("--engine", help="LaTeX engine override (pdflatex/latexmk/tectonic)")
    texpdf.add_argument("--no-open", action="store_true", help="Do not open PDF after compiling")
    texpdf.set_defaults(func=cmd_texpdf)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
