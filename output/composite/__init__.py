"""Composite multi-panel float rendering.

A composite is a single LaTeX float that packs several logical panels (tables,
figures, or a mix) under one caption/label.  The two supported geometries:

* ``stacked`` — all panels are concatenated into **one** ``tabular`` with a
  shared column grid.  Panels with fewer native columns use ``\\multicolumn``
  to span proportionally-wider outer columns, so every panel sits inside the
  same rigid envelope (equal width, aligned rules).

* ``grid`` — panels tile a rigid ``p x q`` lattice.  The whole composite is
  one outer ``tabular`` with fixed-width column spec; each cell is a minipage
  containing a bare panel body.  The outer tabular's ``\\toprule`` /
  ``\\midrule`` / ``\\bottomrule`` provide all horizontal demarcation — panel
  bodies do **not** carry their own top/bottom rules.

The ``probes`` module reads the compiled PDF with ``pdfplumber`` and returns
geometric facts (panel bboxes, row y-coordinates, header baselines) that the
test suite in ``tests/output/composite/`` asserts against.
"""

from econtools.output.composite.model import (
    Composite,
    CompositeStyle,
    FigurePanel,
    TablePanel,
)
from econtools.output.composite.render import render_composite
from econtools.output.composite.stacked import render_stacked
from econtools.output.composite.grid import render_grid

__all__ = [
    "Composite",
    "CompositeStyle",
    "FigurePanel",
    "TablePanel",
    "render_composite",
    "render_stacked",
    "render_grid",
]
