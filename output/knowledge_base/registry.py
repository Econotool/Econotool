"""Knowledge base registry.

Loads YAML concept entries from ``entries/`` and provides rendering with
variable substitution.

Public API
----------
KBEntry     — dataclass for a single concept entry
load_entry  — load a YAML entry by ID
render_entry — render an entry's definition with variable substitution
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_ENTRIES_DIR = Path(__file__).parent / "entries"


@dataclass(frozen=True)
class KBEntry:
    """A single knowledge-base concept entry.

    Parameters
    ----------
    id:
        Unique slug (e.g. ``'breusch_pagan'``).
    name:
        Display name (e.g. ``'Breusch-Pagan Test'``).
    category:
        Category tag (e.g. ``'heteroskedasticity'``).
    definition:
        LaTeX/plain-text definition string.  Supports ``{placeholder}``
        substitution via :func:`render_entry`.
    hypotheses:
        Dict with keys ``'null'`` and ``'alternative'``.
    interpretation:
        Dict with keys ``'reject'`` and ``'fail_to_reject'``.
    references:
        Optional list of citation strings.
    """

    id: str
    name: str
    category: str
    definition: str
    hypotheses: dict = field(default_factory=dict)
    interpretation: dict = field(default_factory=dict)
    references: list[str] = field(default_factory=list)


def load_entry(entry_id: str) -> KBEntry:
    """Load a :class:`KBEntry` from ``entries/{entry_id}.yaml``.

    Parameters
    ----------
    entry_id:
        Slug matching a ``*.yaml`` file in the entries directory.

    Returns
    -------
    KBEntry

    Raises
    ------
    FileNotFoundError
        If no YAML file for *entry_id* exists.
    ImportError
        If PyYAML is not installed.
    """
    if not _HAS_YAML:
        raise ImportError(
            "PyYAML is required for the knowledge base.  "
            "Install it with: pip install pyyaml"
        )
    path = _ENTRIES_DIR / f"{entry_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No knowledge base entry for '{entry_id}'. "
            f"Expected file: {path}"
        )
    with path.open(encoding="utf-8") as f:
        data: dict = yaml.safe_load(f)
    return KBEntry(
        id=data["id"],
        name=data["name"],
        category=data["category"],
        definition=data["definition"],
        hypotheses=data.get("hypotheses", {}),
        interpretation=data.get("interpretation", {}),
        references=data.get("references", []),
    )


def render_entry(entry: KBEntry, **substitutions: str) -> str:
    """Render the definition of *entry* with variable substitution.

    Parameters
    ----------
    entry:
        :class:`KBEntry` to render.
    **substitutions:
        Keyword arguments replace ``{placeholder}`` tokens in the
        definition string (e.g. ``dep_var='lwage'``).

    Returns
    -------
    str
        Rendered definition.
    """
    text = entry.definition
    for key, value in substitutions.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def list_entries() -> list[str]:
    """Return sorted list of all available entry IDs."""
    if not _ENTRIES_DIR.exists():
        return []
    return sorted(p.stem for p in _ENTRIES_DIR.glob("*.yaml"))
