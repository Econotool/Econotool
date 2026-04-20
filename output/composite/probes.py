"""Geometric probes over a compiled composite-float PDF.

The probes extract panel bounding boxes and rule y-coordinates from a
compiled PDF page so the geometric tests in ``tests/output/composite/``
can verify that the rendered composite satisfies the fundamentals
(rectangular outer contour, aligned headers, bounded whitespace).

The shapes on the page are:

* **text** lines from the panel bodies and labels;
* **rect** primitives that implement booktabs rules (``\\toprule`` etc.)
  in pdfTeX.  Rules have height <= 1 pt and width equal to the span of
  the underlying columns.

Coordinates use the PDF convention (y increases upward).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pdfplumber


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float   # bottom edge (PDF convention: y grows upward)
    x1: float
    y1: float   # top edge

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


@dataclass
class RuleSegment:
    """A horizontal rule on the page (a thin rectangle drawn by booktabs)."""

    x0: float
    x1: float
    y: float     # vertical centre of the rule
    thickness: float


@dataclass
class PageGeometry:
    """All the geometric facts we extract about one PDF page."""

    page_number: int
    page_width: float
    page_height: float
    rules: list[RuleSegment]
    text_bbox: BBox | None   # union of every character bbox on the page


def _rect_is_horizontal_rule(rect: dict) -> bool:
    """True if the rectangle behaves like a booktabs horizontal rule."""
    h = abs(rect["y1"] - rect["y0"])
    w = abs(rect["x1"] - rect["x0"])
    # booktabs rules are thin (<= 1.2 pt) and non-trivial width
    return h <= 1.5 and w >= 10.0


def _chars_bbox(chars: Iterable[dict]) -> BBox | None:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for c in chars:
        xs0.append(c["x0"])
        ys0.append(c["y0"])
        xs1.append(c["x1"])
        ys1.append(c["y1"])
    if not xs0:
        return None
    return BBox(min(xs0), min(ys0), max(xs1), max(ys1))


def _line_is_horizontal_rule(line: dict) -> bool:
    """True if a ``page.lines`` entry is a horizontal booktabs-style rule."""
    h = abs(line["y1"] - line["y0"])
    w = abs(line["x1"] - line["x0"])
    return h <= 0.5 and w >= 10.0


def extract_page_geometry(pdf_path: Path, page_index: int = 0) -> PageGeometry:
    """Open *pdf_path* and pull the geometric facts for one page.

    booktabs rules are emitted by pdfTeX as thin stroked lines (pdfplumber
    ``page.lines``) rather than filled rectangles, so we collect both and
    treat any zero-height line as a rule candidate.
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_index]
        rules: list[RuleSegment] = []

        for r in page.rects:
            if _rect_is_horizontal_rule(r):
                rules.append(
                    RuleSegment(
                        x0=min(r["x0"], r["x1"]),
                        x1=max(r["x0"], r["x1"]),
                        y=(r["y0"] + r["y1"]) / 2.0,
                        thickness=abs(r["y1"] - r["y0"]),
                    )
                )

        for ln in page.lines:
            if _line_is_horizontal_rule(ln):
                rules.append(
                    RuleSegment(
                        x0=min(ln["x0"], ln["x1"]),
                        x1=max(ln["x0"], ln["x1"]),
                        y=(ln["y0"] + ln["y1"]) / 2.0,
                        thickness=float(ln.get("linewidth", 0.4) or 0.4),
                    )
                )

        chars = list(page.chars)
        text_bbox = _chars_bbox(chars)
        return PageGeometry(
            page_number=page_index + 1,
            page_width=float(page.width),
            page_height=float(page.height),
            rules=sorted(rules, key=lambda s: (-s.y, s.x0)),
            text_bbox=text_bbox,
        )


def rules_clustered_by_y(rules: list[RuleSegment], tol: float = 0.5) -> list[list[RuleSegment]]:
    """Cluster rules whose y-coordinates agree within *tol* points."""
    if not rules:
        return []
    # Sort descending (top of page first) then cluster.
    ordered = sorted(rules, key=lambda r: -r.y)
    clusters: list[list[RuleSegment]] = [[ordered[0]]]
    for r in ordered[1:]:
        if abs(r.y - clusters[-1][0].y) <= tol:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    return clusters


def horizontal_span_of_rules(rules: list[RuleSegment]) -> tuple[float, float] | None:
    """``(x0, x1)`` of the union of all supplied rule segments."""
    if not rules:
        return None
    return (min(r.x0 for r in rules), max(r.x1 for r in rules))


def largest_empty_rect(
    occupancy: Iterable[BBox],
    frame: BBox,
    *,
    grid: int = 80,
) -> BBox:
    """Approximate largest axis-aligned empty rectangle inside *frame*.

    Uses a coarse grid-based occupancy map for robustness against LaTeX's
    sub-pixel jitter — a pixel is "filled" if any supplied bbox covers
    its centre.  Returns the maximum all-ones rectangle.  Accuracy is
    bounded by ``frame.width / grid`` horizontally and ``frame.height /
    grid`` vertically, which is fine for the whitespace test (~2 pt
    resolution on a letter page).
    """
    bboxes = list(occupancy)
    W = H = grid
    cell_w = frame.width / W
    cell_h = frame.height / H
    if cell_w <= 0 or cell_h <= 0:
        return BBox(0, 0, 0, 0)

    # mask[row][col] == 1 if cell is empty.
    mask = [[1] * W for _ in range(H)]
    for bb in bboxes:
        # Map bbox to cell ranges.  PDF y grows upward; row 0 of mask is
        # the top of the frame.
        c0 = max(0, int((bb.x0 - frame.x0) / cell_w))
        c1 = min(W, int((bb.x1 - frame.x0) / cell_w) + 1)
        r0 = max(0, int((frame.y1 - bb.y1) / cell_h))
        r1 = min(H, int((frame.y1 - bb.y0) / cell_h) + 1)
        for r in range(r0, r1):
            row = mask[r]
            for c in range(c0, c1):
                row[c] = 0

    # Classic O(W*H) largest-rectangle-in-histogram over row heights.
    heights = [0] * W
    best_area = 0.0
    best = BBox(0, 0, 0, 0)
    for r in range(H):
        row = mask[r]
        for c in range(W):
            heights[c] = heights[c] + 1 if row[c] else 0
        # Histogram max for this row.
        stack: list[int] = []
        for c in range(W + 1):
            cur = heights[c] if c < W else 0
            start = c
            while stack and heights[stack[-1]] > cur:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                width_cells = c - left
                height_cells = heights[top]
                area_pts = width_cells * cell_w * height_cells * cell_h
                if area_pts > best_area:
                    best_area = area_pts
                    best = BBox(
                        x0=frame.x0 + left * cell_w,
                        y0=frame.y1 - (r + 1) * cell_h,
                        x1=frame.x0 + c * cell_w,
                        y1=frame.y1 - (r + 1 - height_cells) * cell_h,
                    )
                start = left
            stack.append(c)
    return best
