"""Top-level adapter: ``CompositeNode`` → composite model → LaTeX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from econtools.output.composite.grid import render_grid
from econtools.output.composite.model import Composite, FigurePanel, TablePanel
from econtools.output.composite.stacked import render_stacked


def render_composite(
    cnode: Any,                       # CompositeNode from econtools.paper.composition
    paper: Any,                       # Paper — exposes table_ids(), figure_ids()
    bare_fragments: dict[str, str],
    figure_paths: dict[str, Path],
) -> str:
    """Resolve *cnode* into a :class:`Composite`, then dispatch to a renderer.

    The caller owns rendering of individual panels — we receive the bare
    tabular strings via *bare_fragments* and the filesystem paths of
    pre-rendered figures via *figure_paths*, so this layer does no
    statistical work.
    """
    table_ids = set(paper.table_ids())
    figure_ids = set(paper.figure_ids())
    n = len(cnode.panels)

    if cnode.layout is not None:
        nrows, ncols = cnode.layout
    elif all(pid in table_ids for pid in cnode.panels):
        nrows, ncols = n, 1
    else:
        ncols = min(n, 2) if n > 1 else 1
        nrows = (n + ncols - 1) // ncols

    subcaps = list(cnode.subcaptions or [""] * n)

    def _label(i: int, pid: str) -> str:
        letter = chr(ord("A") + i)
        subcap = subcaps[i] if i < len(subcaps) else ""
        return f"Panel {letter}." + (f" {subcap}" if subcap else "")

    panels: list[TablePanel | FigurePanel] = []
    for i, pid in enumerate(cnode.panels):
        label = _label(i, pid)
        if pid in table_ids:
            src = bare_fragments.get(pid)
            if src is None:
                raise KeyError(f"no bare rendering for table '{pid}'")
            panels.append(TablePanel(panel_id=pid, label=label, tabular_src=src))
        elif pid in figure_ids:
            fpath = figure_paths.get(pid)
            if fpath is None:
                raise KeyError(f"figure '{pid}' was not rendered")
            rel = (Path("figures") / fpath.name).as_posix()
            panels.append(FigurePanel(panel_id=pid, label=label, image_path=rel))
        else:
            raise KeyError(
                f"composite panel '{pid}' is neither a table nor a figure"
            )

    all_tables = all(isinstance(p, TablePanel) for p in panels)
    composite = Composite(
        composite_id=cnode.composite_id,
        panels=panels,
        layout=(nrows, ncols),
        caption=cnode.caption,
        notes=cnode.notes,
        float_kind="table" if all_tables else "figure",
        column_fraction=cnode.column_fraction,
    )

    if ncols == 1 and all_tables:
        return render_stacked(composite)
    return render_grid(composite, nrows=nrows, ncols=ncols)
