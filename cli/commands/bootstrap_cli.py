"""CLI command: econtools bootstrap --config path/to/config.yaml

Reads a YAML configuration file, loads data, runs run_bootstrap, and writes:
- results table (CSV and optionally LaTeX)
- manifest.json
- draws file (.npy) if configured

Usage
-----
    econtools bootstrap --config analysis/bootstrap_config.yaml

See ``bootstrap_default_config.yaml`` in the repo root for a fully
annotated example configuration file.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def add_bootstrap_parser(subparsers: Any) -> None:  # noqa: ANN401
    """Register the ``bootstrap`` subcommand on *subparsers*."""
    p = subparsers.add_parser(
        "bootstrap",
        help="Run bootstrap inference from a YAML config file.",
        description=__doc__,
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="CONFIG",
        help="Path to the YAML configuration file.",
    )
    p.add_argument(
        "--manifest-path",
        default=None,
        metavar="PATH",
        help="Override manifest output path (default: from config or manifest.json).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    p.set_defaults(func=run_bootstrap_command)


def run_bootstrap_command(args: Any) -> int:  # noqa: ANN401
    """Entry point for ``econtools bootstrap``."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: PyYAML is required for the bootstrap CLI. "
            "Install it with: pip install pyyaml",
            file=sys.stderr,
        )
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not args.quiet:
        print(f"Loaded config from {config_path}")

    # --- load data ---
    data_path = Path(cfg["data_path"])
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        return 1

    import pandas as pd

    suffix = data_path.suffix.lower()
    if suffix == ".dta":
        df = pd.read_stata(
            data_path,
            convert_categoricals=False,
            preserve_dtypes=False,
            convert_missing=False,
        )
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(data_path)
    else:
        print(f"ERROR: Unsupported data format: {suffix}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns from {data_path}")

    # --- run bootstrap ---
    from econtools.uncertainty.bootstrap import run_bootstrap

    kwargs: dict[str, Any] = {
        "data": df,
        "y": cfg["y"],
        "X": cfg["X"],
        "estimator": cfg.get("estimator", "ols"),
        "bootstrap_method": cfg.get("bootstrap_method", "iid_pairs"),
        "B": cfg.get("B", 1999),
        "seed": cfg.get("seed", 12345),
        "Z": cfg.get("Z") or None,
        "endog": cfg.get("endog") or None,
        "cluster": cfg.get("cluster") or None,
        "id_col": cfg.get("id_col") or None,
        "time_col": cfg.get("time_col") or None,
        "add_intercept": cfg.get("add_intercept", True),
        "ci_level": cfg.get("ci_level", 0.95),
        "ci_methods": cfg.get("ci_methods", ["percentile", "basic"]),
        "wild_dist": cfg.get("wild_dist", "rademacher"),
        "block_length": cfg.get("block_length"),
        "save_draws_path": cfg.get("save_draws_path"),
        "n_jobs": cfg.get("n_jobs", 1),
    }

    if not args.quiet:
        print(
            f"Running {kwargs['bootstrap_method']} bootstrap "
            f"({kwargs['B']} replications, seed={kwargs['seed']}) …"
        )

    result = run_bootstrap(**kwargs)

    if not args.quiet:
        print("Bootstrap complete.")

    # --- write manifest ---
    manifest_path = (
        args.manifest_path
        or cfg.get("manifest_path", "manifest.json")
    )
    from econtools.uncertainty._bootstrap_manifest import write_manifest
    write_manifest(result["_manifest"], manifest_path)
    if not args.quiet:
        print(f"Manifest written to {manifest_path}")

    # --- write CSV results ---
    output_csv = cfg.get("output_csv")
    if output_csv:
        _write_results_csv(result, Path(output_csv))
        if not args.quiet:
            print(f"Results table written to {output_csv}")

    # --- write LaTeX results (optional) ---
    output_latex = cfg.get("output_latex")
    if output_latex:
        _write_results_latex(result, Path(output_latex))
        if not args.quiet:
            print(f"LaTeX table written to {output_latex}")

    # --- print summary to stdout ---
    if not args.quiet:
        _print_summary(result)

    return 0


def _write_results_csv(result: dict[str, Any], path: Path) -> None:
    """Write bootstrap results as a CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pe = result["point_estimate"]
    bs = result["bootstrap"]
    coef_names = pe["coef_names"]

    rows = []
    for name in coef_names:
        row: dict[str, Any] = {
            "coef": name,
            "point_estimate": pe["params"][name],
            "bootstrap_se": bs["se"][name],
            "pvalue": bs["pvalues"][name],
            "bagged_mean": bs["bagged_mean"][name],
            "bagged_median": bs["bagged_median"][name],
        }
        for ci_method, ci_data in bs["ci"].items():
            row[f"ci_{ci_method}_lower"] = ci_data["lower"][name]
            row[f"ci_{ci_method}_upper"] = ci_data["upper"][name]
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_results_latex(result: dict[str, Any], path: Path) -> None:
    """Write a minimal LaTeX tabular of bootstrap results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pe = result["point_estimate"]
    bs = result["bootstrap"]
    coef_names = pe["coef_names"]
    ci_methods = list(bs["ci"].keys())

    lines = [
        r"\begin{tabular}{l r r r}",
        r"\hline\hline",
        r"Coefficient & Estimate & Bootstrap SE & p-value \\",
        r"\hline",
    ]
    for name in coef_names:
        est = pe["params"][name]
        se = bs["se"][name]
        pv = bs["pvalues"][name]
        stars = _stars(pv)
        lines.append(
            f"{name} & {est:.4f}{stars} & ({se:.4f}) & {pv:.4f} \\\\"
        )
    lines += [
        r"\hline",
        f"$N$ & {result['metadata']['n_obs']} & & \\\\",
        f"$B$ & {result['metadata']['B']} & & \\\\",
        r"\hline\hline",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def _print_summary(result: dict[str, Any]) -> None:
    """Print a readable summary to stdout."""
    meta = result["metadata"]
    pe = result["point_estimate"]
    bs = result["bootstrap"]
    coef_names = pe["coef_names"]

    print(
        f"\n{'='*60}\n"
        f"Bootstrap Results\n"
        f"  Estimator : {meta['estimator'].upper()}\n"
        f"  Method    : {meta['method']}\n"
        f"  B         : {meta['B']}\n"
        f"  Seed      : {meta['seed']}\n"
        f"  N obs     : {meta['n_obs']}\n"
        f"  N dropped : {meta['n_dropped']}\n"
        f"{'='*60}"
    )
    header = f"{'Coef':<20} {'Estimate':>12} {'Boot SE':>12} {'p-value':>10}"
    print(header)
    print("-" * 56)
    for name in coef_names:
        est = pe["params"][name]
        se = bs["se"][name]
        pv = bs["pvalues"][name]
        print(f"{name:<20} {est:>12.4f} {se:>12.4f} {pv:>10.4f}")
    print("=" * 60)
    if meta["warnings"]:
        print("WARNINGS:")
        for w in meta["warnings"]:
            print(f"  ! {w}")
        print()
