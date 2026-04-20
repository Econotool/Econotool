"""Append-only provenance log for pipeline steps.

Each dataset transformation appends a record to a sidecar
``<name>_provenance.json`` file in ``data_lake/curated/``.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_step(
    log_path: str | pathlib.Path,
    function_name: str,
    kwargs: dict[str, Any],
    note: str = "",
) -> None:
    """Append one pipeline step to an existing or new provenance log.

    Parameters
    ----------
    log_path:
        Path to the ``_provenance.json`` file (created if absent).
    function_name:
        Name of the pipeline function (e.g. ``'clean.winsorise'``).
    kwargs:
        The keyword arguments passed to the function — logged as-is.
        Values must be JSON-serialisable; non-serialisable values are
        coerced to their ``repr()``.
    note:
        Optional free-text annotation.
    """
    log_path = pathlib.Path(log_path)

    # Coerce non-serialisable kwargs values to repr strings
    safe_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        try:
            json.dumps(v)
            safe_kwargs[k] = v
        except (TypeError, ValueError):
            safe_kwargs[k] = repr(v)

    entry = {
        "timestamp": _now_iso(),
        "function": function_name,
        "kwargs": safe_kwargs,
    }
    if note:
        entry["note"] = note

    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as fh:
            records: list[dict[str, Any]] = json.load(fh)
    else:
        records = []

    records.append(entry)

    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)


def read_log(log_path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Return all provenance records from *log_path*.

    Returns an empty list if the file does not exist.
    """
    log_path = pathlib.Path(log_path)
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
