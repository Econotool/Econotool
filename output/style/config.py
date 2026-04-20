"""Style configuration dataclasses and YAML loader.

A :class:`StyleConfig` is an immutable bundle of defaults consulted by
the :class:`~econtools.paper.Paper` execution layer when a user has not
overridden a parameter at the call site.  Style files are small YAML
documents (or equivalent mappings) that override one or more fields of
``DEFAULT_STYLE``; any field not mentioned falls back to the default.

Public API
----------
PanelStyle       — defaults for multi-panel figure composition
TableStyle       — defaults for publication-grade tables
StyleConfig      — top-level container (panel, table)
DEFAULT_STYLE    — built-in baseline
load_style(path) — load YAML overrides
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PanelStyle:
    """Defaults for multi-panel (scatter/line/composite) figures.

    Parameters
    ----------
    per_panel_width, per_panel_height:
        Inches per panel.  Total figsize is ``(ncols*w, nrows*h + figsize_row_pad)``.
    figsize_row_pad:
        Extra inches added once to the figure height (to accommodate a
        shared caption / super-title / notes strip).
    hspace, wspace:
        ``subplots_adjust`` spacing when the layout has multiple rows /
        columns.  ``None`` for a given axis means "let matplotlib decide"
        (via ``tight_layout``).
    panel_labels:
        Default label style: ``"none" | "lower_alpha" | "upper_alpha" | "panel"``.
    group_frame:
        When ``True``, draw a thin box around the composite figure to
        group the panels visually.
    frame_linewidth, frame_color:
        Line width / color of the group frame.
    """

    per_panel_width: float = 5.4
    per_panel_height: float = 3.1
    figsize_row_pad: float = 0.3
    hspace: float | None = 0.42
    wspace: float | None = 0.30
    panel_labels: str = "none"
    group_frame: bool = False
    frame_linewidth: float = 0.6
    frame_color: str = "#444444"


@dataclass(frozen=True)
class TableStyle:
    """Defaults for publication-grade tables (results/summary/diagnostic).

    Parameters
    ----------
    missing_cell:
        Placeholder for blank coefficient cells (em-dash by default).
    double_rule:
        ``True`` for ``\\hline\\hline`` top/bottom; ``False`` for
        ``\\toprule`` / ``\\bottomrule`` (booktabs).  Overridden by the
        journal profile when that profile declares a preference.
    """

    missing_cell: str = "---"
    double_rule: bool = False


@dataclass(frozen=True)
class StyleConfig:
    """Top-level style container."""

    panel: PanelStyle = field(default_factory=PanelStyle)
    table: TableStyle = field(default_factory=TableStyle)


DEFAULT_STYLE = StyleConfig()


def _merge_dataclass(instance, overrides: Mapping[str, Any]):
    """Return a copy of *instance* with any matching keys from *overrides* applied."""
    valid = {f.name for f in fields(instance)}
    kwargs = {k: v for k, v in overrides.items() if k in valid}
    return replace(instance, **kwargs) if kwargs else instance


def load_style(source: str | Path | Mapping[str, Any] | None) -> StyleConfig:
    """Load a :class:`StyleConfig`, optionally merging YAML / dict overrides.

    Parameters
    ----------
    source:
        - ``None``: return :data:`DEFAULT_STYLE`.
        - ``str`` or :class:`Path`: path to a YAML file (see format below).
        - ``Mapping``: nested dict with the same shape as the YAML file.

    YAML format
    -----------
    ::

        panel:
          per_panel_width: 5.0
          hspace: 0.3
          group_frame: true
        table:
          missing_cell: "-"
          double_rule: false
    """
    if source is None:
        return DEFAULT_STYLE

    if isinstance(source, Mapping):
        data: Mapping[str, Any] = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Style file not found: {path}")
        try:
            import yaml
        except ImportError as err:  # pragma: no cover — pyyaml in deps
            raise ImportError(
                "PyYAML is required to load style files from YAML."
            ) from err
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    panel_over = data.get("panel", {}) or {}
    table_over = data.get("table", {}) or {}
    return StyleConfig(
        panel=_merge_dataclass(DEFAULT_STYLE.panel, panel_over),
        table=_merge_dataclass(DEFAULT_STYLE.table, table_over),
    )
