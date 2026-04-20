"""Econtools publication style configuration.

Public API
----------
StyleConfig      — frozen dataclass of style defaults (figures, panels, tables)
load_style(path) — load a YAML override file; merges into ``DEFAULT_STYLE``
DEFAULT_STYLE    — the built-in default :class:`StyleConfig`
"""

from econtools.output.style.config import (
    DEFAULT_STYLE,
    PanelStyle,
    StyleConfig,
    TableStyle,
    load_style,
)

__all__ = [
    "DEFAULT_STYLE",
    "PanelStyle",
    "StyleConfig",
    "TableStyle",
    "load_style",
]
