"""CLI commands for the sieve subsystem.

Usage::

    econtools sieve --config sieve.yaml [--output ./sieve_runs/run1]
    econtools sieve-report --run ./sieve_runs/run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fall back to JSON
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def cmd_sieve(args: argparse.Namespace) -> int:
    """Run a sieve from a YAML/JSON configuration file."""
    from econtools.sieve.api import run_sieve
    from econtools.sieve.reporting import leaderboard_summary
    import pandas as pd

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    cfg = _load_yaml(config_path)

    # Load data
    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("path") or args.data
    if not data_path:
        print("Error: data path not specified (use 'data.path' in config or --data flag).",
              file=sys.stderr)
        return 1

    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        return 1

    # Load data based on extension
    ext = data_path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(str(data_path))
    elif ext in (".csv", ".tsv"):
        df = pd.read_csv(str(data_path))
    elif ext == ".dta":
        df = pd.read_stata(str(data_path))
    else:
        print(f"Error: unsupported data format '{ext}'. Use .parquet, .csv, or .dta.",
              file=sys.stderr)
        return 1

    # Apply filters
    filters = data_cfg.get("filters", {})
    for col, val in filters.items():
        if col in df.columns:
            df = df[df[col] == val]

    # Target spec
    target = cfg.get("target", {})
    y = target.get("y") or args.y
    base_X = target.get("base_X") or []
    estimator = target.get("estimator", "ols")
    endog = target.get("endog")
    base_Z = target.get("base_Z")

    if not y:
        print("Error: dependent variable 'y' not specified in config or --y flag.",
              file=sys.stderr)
        return 1

    output_dir = args.output or cfg.get("output", {}).get("path", "./sieve_output")
    seed = args.seed or cfg.get("reproducibility", {}).get("seed", 12345)
    n_jobs = args.jobs or cfg.get("reproducibility", {}).get("n_jobs", 1)

    print(f"Running sieve: y={y}, estimator={estimator}, n_base_X={len(base_X)}", flush=True)
    print(f"  Data: {len(df)} rows × {len(df.columns)} cols", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Seed: {seed}", flush=True)

    result = run_sieve(
        data=df,
        y=y,
        base_X=base_X,
        estimator=estimator,
        endog=endog,
        base_Z=base_Z,
        sieve_spec=cfg,
        seed=seed,
        n_jobs=n_jobs,
        output_dir=output_dir,
    )

    meta = result["run_metadata"]
    print(f"\nSieve complete — {meta['n_candidates']} candidates evaluated.")
    print(f"  Selected: {meta['n_selected']}")
    print(f"  Protocol: {meta['protocol']}")
    if meta["exploratory_only"]:
        print("  WARNING: EXPLORATORY ONLY — in-sample selection used.")
    print(f"  Run ID: {meta['run_id']}")

    summary = leaderboard_summary(
        result["leaderboard"],
        n_show=10,
        primary_metric=cfg.get("selection", {}).get("primary_metric", "rmse"),
        exploratory_only=meta["exploratory_only"],
    )
    print("\n" + summary)

    if result["violations"]:
        print(f"\n{len(result['violations'])} guardrail violations:")
        for v in result["violations"][:5]:
            print(f"  [{v.reason_code}] {v.details[:80]}")

    print(f"\nArtifacts written to: {output_dir}")
    return 0


def cmd_sieve_report(args: argparse.Namespace) -> int:
    """Print a summary of a previously saved sieve run."""
    from econtools.sieve.api import load_sieve_results

    run_dir = args.run
    if not Path(run_dir).exists():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    results = load_sieve_results(run_dir)
    manifest = results.get("manifest")
    leaderboard = results.get("leaderboard")
    cards = results.get("model_cards", [])

    if manifest:
        print("=" * 60)
        print(f"Run ID:       {manifest.get('run_id')}")
        print(f"Timestamp:    {manifest.get('timestamp')}")
        print(f"Protocol:     {manifest.get('protocol_mode')}")
        print(f"Candidates:   {manifest.get('n_candidates')}")
        print(f"Selected:     {manifest.get('n_selected')}")
        print(f"Config hash:  {manifest.get('config_hash')}")
        print(f"Data hash:    {manifest.get('dataset_fingerprint')}")
        if manifest.get("exploratory_only"):
            print("\n  WARNING: EXPLORATORY ONLY")
        print("=" * 60)

    if leaderboard is not None:
        print(f"\nLeaderboard ({len(leaderboard)} rows, top 10):")
        show_cols = [c for c in ["candidate_hash", "estimator", "n_X_terms",
                                   "mean_rmse", "rejected"] if c in leaderboard.columns]
        print(leaderboard[show_cols].head(10).to_string(index=False))

    if cards:
        print(f"\nSelected model cards ({len(cards)}):")
        for card in cards:
            h = card.get("candidate_hash", "")[:8]
            est = card.get("model", {}).get("estimator", "")
            n = card.get("n_obs", "?")
            print(f"  [{h}] {est} — N={n}")

    return 0


def register_sieve_commands(subparsers: Any) -> None:
    """Register 'sieve' and 'sieve-report' sub-commands."""
    # sieve
    p_sieve = subparsers.add_parser("sieve", help="Run a model specification sieve.")
    p_sieve.add_argument("--config", required=True, help="Path to sieve YAML/JSON config.")
    p_sieve.add_argument("--data", default=None, help="Path to data file (overrides config).")
    p_sieve.add_argument("--y", default=None, help="Dependent variable (overrides config).")
    p_sieve.add_argument("--output", default=None, help="Output directory.")
    p_sieve.add_argument("--seed", type=int, default=None, help="Random seed.")
    p_sieve.add_argument("--jobs", type=int, default=None, help="Parallel workers.")
    p_sieve.set_defaults(func=cmd_sieve)

    # sieve-report
    p_report = subparsers.add_parser("sieve-report", help="Summarise a previous sieve run.")
    p_report.add_argument("--run", required=True, help="Path to sieve run output directory.")
    p_report.set_defaults(func=cmd_sieve_report)


# typing shim
from typing import Any
