"""Manifest writing and config hashing for bootstrap runs.

A manifest is a JSON file capturing the full reproducibility record:
seed, method, sample stats, package versions, git commit, and a stable
config hash so downstream consumers can detect configuration drift.

Internal API.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_config_hash(config: dict[str, Any]) -> str:
    """Return SHA-256 of a canonically serialised config dict.

    Keys are sorted recursively so the hash is stable regardless of
    dict insertion order.

    Returns
    -------
    str
        Hex digest prefixed with ``'sha256:'``.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"sha256:{digest}"


def _git_commit_hash() -> str:
    """Return short git commit hash, or ``'unknown'`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _package_versions() -> dict[str, str]:
    """Return versions of key packages used by the bootstrap module."""
    packages = ["econtools", "numpy", "pandas", "scipy", "statsmodels"]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    versions["python"] = sys.version.split()[0]
    return versions


def build_manifest(
    *,
    config: dict[str, Any],
    n_obs: int,
    n_dropped: int,
    y_col: str,
    x_cols: list[str],
    endog_cols: list[str],
    z_cols: list[str],
    cluster_col: str | None,
    id_col: str | None,
    cluster_count: int | None,
    warnings_list: list[str],
) -> dict[str, Any]:
    """Assemble the full manifest dict.

    Parameters
    ----------
    config:
        The normalised configuration dict (everything passed to run_bootstrap).
    n_obs:
        Number of observations used in estimation (after listwise deletion).
    n_dropped:
        Number of rows dropped due to missing values.
    y_col, x_cols, endog_cols, z_cols:
        Column lists used.
    cluster_col, id_col:
        Optional grouping columns.
    cluster_count:
        Number of distinct clusters / ids in the estimation sample.
    warnings_list:
        Any warnings generated during the run (e.g. small cluster count).

    Returns
    -------
    dict
        Fully populated manifest.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "package_versions": _package_versions(),
        "config": config,
        "config_hash": compute_config_hash(config),
        "sample": {
            "n_obs": n_obs,
            "n_dropped": n_dropped,
            "y_col": y_col,
            "x_cols": x_cols,
            "endog_cols": endog_cols,
            "z_cols": z_cols,
            "cluster_col": cluster_col,
            "id_col": id_col,
            "cluster_count": cluster_count,
        },
        "warnings": warnings_list,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write *manifest* as indented JSON to *path*.

    Creates parent directories as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
