"""Ingestion and persistence for the data pipeline.

Public API
----------
verify_hash(path, expected_sha256)
    Raise if file hash doesn't match.

load_dta(path, manifest_path=None, verify=True)
    Load a Stata .dta file; extract all metadata; optionally verify hash.
    Returns (DataFrame, meta_dict).

load_csv(path, **kwargs)
    Load a CSV with nullable-int backend.
    Returns DataFrame.

load_raw(name, source='wooldridge_and_oleg', base_dir=None, verify=True)
    Convenience wrapper: resolve path from data_lake/raw/, load, verify.
    Returns (DataFrame, meta_dict).

save_curated(df, name, version, meta, base_dir=None)
    Write versioned Parquet + sidecar _meta.json to data_lake/curated/.

versioned_path(name, version, base_dir=None)
    Return (parquet_path, meta_path) for a given name/version.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # C:\Econometrics
_RAW_ROOT = _REPO_ROOT / "data_lake" / "raw"
_CURATED_ROOT = _REPO_ROOT / "data_lake" / "curated"


def _raw_dir(source: str, base_dir: str | pathlib.Path | None) -> pathlib.Path:
    if base_dir is not None:
        return pathlib.Path(base_dir)
    return _RAW_ROOT / source


def _curated_dir(base_dir: str | pathlib.Path | None) -> pathlib.Path:
    if base_dir is not None:
        return pathlib.Path(base_dir)
    return _CURATED_ROOT


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------


def verify_hash(path: str | pathlib.Path, expected_sha256: str) -> None:
    """Raise ``ValueError`` if the SHA-256 of *path* doesn't match *expected_sha256*.

    Parameters
    ----------
    path:
        Path to the file to verify.
    expected_sha256:
        Expected hex digest (case-insensitive).
    """
    path = pathlib.Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Hash mismatch for {path.name}\n"
            f"  expected: {expected_sha256.lower()}\n"
            f"  actual:   {actual}"
        )


def _load_manifest(manifest_path: str | pathlib.Path) -> dict[str, Any]:
    with pathlib.Path(manifest_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _lookup_hash(filename: str, manifest_path: str | pathlib.Path) -> str | None:
    """Return expected SHA-256 for *filename* from the manifest, or None."""
    manifest = _load_manifest(manifest_path)
    # Support both list-of-records and dict-keyed manifests
    if isinstance(manifest, list):
        for entry in manifest:
            if entry.get("filename") == filename or entry.get("name") == filename:
                return entry.get("sha256") or entry.get("sha256_hex")
    elif isinstance(manifest, dict):
        entry = manifest.get(filename)
        if entry:
            return entry.get("sha256") or entry.get("sha256_hex")
    return None


# ---------------------------------------------------------------------------
# Stata .dta ingestion
# ---------------------------------------------------------------------------


def load_dta(
    path: str | pathlib.Path,
    manifest_path: str | pathlib.Path | None = None,
    verify: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a Stata ``.dta`` file and extract all metadata.

    Parameters
    ----------
    path:
        Path to the ``.dta`` file.
    manifest_path:
        Path to ``manifest.json``.  If *verify* is True and this is
        provided, the file hash is verified before loading.
    verify:
        Whether to verify the SHA-256 hash (requires *manifest_path*).

    Returns
    -------
    (df, meta)
        ``df`` is the loaded ``pd.DataFrame``.
        ``meta`` is a dict with all extracted Stata metadata.
    """
    path = pathlib.Path(path)

    if verify and manifest_path is not None:
        manifest_path = pathlib.Path(manifest_path)
        expected = _lookup_hash(path.name, manifest_path)
        if expected is not None:
            verify_hash(path, expected)
        else:
            # Hash not in manifest — warn but don't block
            import warnings
            warnings.warn(
                f"{path.name} not found in manifest; skipping hash check.",
                stacklevel=2,
            )

    # Extract metadata via StataReader before loading the full DataFrame
    meta: dict[str, Any] = {
        "source_file": path.name,
        "source_type": "dta",
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    with pd.io.stata.StataReader(str(path)) as reader:
        meta["data_label"] = reader.data_label
        meta["variable_labels"] = reader.variable_labels()
        meta["value_labels"] = reader.value_labels()
        # stata_dtypes: map colname → Stata storage type string (if available)
        varlist = getattr(reader, "varlist", None)
        dtyplist = getattr(reader, "dtyplist", None)
        if varlist is not None and dtyplist is not None:
            meta["stata_dtypes"] = dict(zip(varlist, dtyplist))

    if manifest_path is not None:
        manifest = _load_manifest(manifest_path)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        meta["source_sha256"] = actual_hash

    df: pd.DataFrame = pd.read_stata(
        str(path),
        convert_dates=True,
        convert_categoricals=True,
        preserve_dtypes=True,
        convert_missing=False,
    )

    return df, meta


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------


def load_csv(
    path: str | pathlib.Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV file using nullable-integer dtype backend.

    Parameters
    ----------
    path:
        Path to the CSV file.
    **kwargs:
        Forwarded to ``pd.read_csv``.  ``dtype_backend`` defaults to
        ``'numpy_nullable'`` unless overridden.

    Returns
    -------
    pd.DataFrame
    """
    kwargs.setdefault("dtype_backend", "numpy_nullable")
    return pd.read_csv(str(path), **kwargs)


# ---------------------------------------------------------------------------
# Convenience: load_raw
# ---------------------------------------------------------------------------


def load_raw(
    name: str,
    source: str = "wooldridge_and_oleg",
    base_dir: str | pathlib.Path | None = None,
    verify: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a raw dataset by name from ``data_lake/raw/<source>/``.

    Automatically detects ``.dta`` or ``.csv`` extension.
    For ``.dta`` files the manifest at the same directory is consulted
    for hash verification.

    Parameters
    ----------
    name:
        Filename without extension, or full filename (e.g. ``'wage1'``
        or ``'wage1.dta'``).
    source:
        Sub-directory of ``data_lake/raw/`` (default: ``'wooldridge_and_oleg'``).
    base_dir:
        Override the raw data directory root.
    verify:
        Verify SHA-256 hash against manifest.json for .dta files.

    Returns
    -------
    (df, meta)
    """
    raw_dir = _raw_dir(source, base_dir)

    # Resolve filename — try with and without extension
    candidates = [
        raw_dir / name,
        raw_dir / (name + ".dta"),
        raw_dir / (name + ".csv"),
    ]
    path: pathlib.Path | None = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        raise FileNotFoundError(
            f"Could not find '{name}' in {raw_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    if path.suffix.lower() == ".dta":
        manifest_path = raw_dir / "manifest.json"
        return load_dta(
            path,
            manifest_path=manifest_path if (verify and manifest_path.exists()) else None,
            verify=verify,
        )
    elif path.suffix.lower() == ".csv":
        df = load_csv(path)
        meta: dict[str, Any] = {
            "source_file": path.name,
            "source_type": "csv",
            "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return df, meta
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


# ---------------------------------------------------------------------------
# Versioned paths
# ---------------------------------------------------------------------------


def versioned_path(
    name: str,
    version: str | int,
    base_dir: str | pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Return ``(parquet_path, meta_path)`` for a curated dataset.

    Parameters
    ----------
    name:
        Dataset base name (e.g. ``'wages_panel'``).
    version:
        Version string or integer (e.g. ``'v1'`` or ``1``).
    base_dir:
        Override the curated output directory.

    Returns
    -------
    (parquet_path, meta_path)
        e.g. ``(…/wages_panel_v1.parquet, …/wages_panel_v1_meta.json)``
    """
    curated_dir = _curated_dir(base_dir)
    if isinstance(version, int):
        version = f"v{version}"
    stem = f"{name}_{version}"
    return curated_dir / f"{stem}.parquet", curated_dir / f"{stem}_meta.json"


# ---------------------------------------------------------------------------
# Save curated
# ---------------------------------------------------------------------------


def save_curated(
    df: pd.DataFrame,
    name: str,
    version: str | int,
    meta: dict[str, Any],
    base_dir: str | pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write versioned Parquet + sidecar ``_meta.json`` to ``data_lake/curated/``.

    Parameters
    ----------
    df:
        DataFrame to save.
    name:
        Dataset base name (e.g. ``'wages_panel'``).
    version:
        Version string or integer.
    meta:
        Metadata dict to write to the sidecar JSON.  Common fields:
        ``source_file``, ``source_sha256``, ``variable_labels``, etc.
        Extra fields are written as-is.  ``curated_version``,
        ``saved_at``, ``shape`` are auto-added.
    base_dir:
        Override the curated output directory.

    Returns
    -------
    (parquet_path, meta_path)
    """
    parquet_path, meta_path = versioned_path(name, version, base_dir)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Write Parquet — pyarrow engine, snappy compression, preserve index
    df.to_parquet(str(parquet_path), engine="pyarrow", compression="snappy", index=True)

    # Augment and write sidecar metadata
    if isinstance(version, int):
        version = f"v{version}"
    full_meta = {
        **meta,
        "curated_version": version,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shape": list(df.shape),
        "columns": list(df.columns),
    }

    # Coerce non-JSON-serialisable values (e.g. numpy dtypes) to strings
    def _coerce(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_coerce(v) for v in obj]
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(_coerce(full_meta), fh, indent=2, ensure_ascii=False)

    return parquet_path, meta_path
