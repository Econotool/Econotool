r"""Section → fragment ordering for a :class:`Paper`.

Given a :class:`Paper` and a dict of already-rendered fragments (tables
and figures), :func:`compose_body` produces the ordered list of LaTeX
fragments that make up the main body, and :func:`compose_appendix`
returns the appendix fragments.

Ordering rules
--------------
1. Sections appear in insertion order.
2. Each section is emitted as a ``\section*{title}`` followed by its
   body text.
3. After a section's body, any tables/figures named in ``after`` are
   inserted in the order they were listed.
4. Tables and figures that are never referenced by a section are
   appended at the end of the body (still in the order they were
   registered) — except those routed to the appendix.
5. Appendix fragments are one-per-page (handled by
   :func:`econtools.output.latex.document.assemble_appendix`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from econtools.paper.builder import Paper


def _section_header(title: str, *, numbered: bool = False) -> str:
    """LaTeX section header; numbered or starred based on *numbered*."""
    if not title:
        return ""
    cmd = "section" if numbered else "section*"
    return rf"\{cmd}{{{title}}}"


def compose_body(
    paper: "Paper",
    fragments: dict[str, str],
) -> list[str]:
    """Return the ordered list of main-body LaTeX fragments.

    Parameters
    ----------
    paper:
        The :class:`Paper` whose plan is being rendered.
    fragments:
        Dict keyed by table/figure id → LaTeX fragment string.
    """
    appendix_set = set(paper.appendix_ids)
    referenced: set[str] = set()
    parts: list[str] = []

    numbered = bool(getattr(paper, "numbered_sections", False))
    for section in paper.sections:
        if section.title:
            parts.append(_section_header(section.title, numbered=numbered))
        if section.body:
            parts.append(section.body)
        for ref_id in section.after:
            if ref_id in appendix_set:
                continue
            if ref_id in fragments:
                parts.append(fragments[ref_id])
                referenced.add(ref_id)

    # Trailing un-referenced tables and figures go to the end of the body
    for tid in paper.table_ids():
        if tid in referenced or tid in appendix_set:
            continue
        if tid in fragments:
            parts.append(fragments[tid])
    for fid in paper.figure_ids():
        if fid in referenced or fid in appendix_set:
            continue
        if fid in fragments:
            parts.append(fragments[fid])
    for cid in paper.composite_ids():
        if cid in referenced or cid in appendix_set:
            continue
        if cid in fragments:
            parts.append(fragments[cid])

    return parts


def compose_appendix(
    paper: "Paper",
    fragments: dict[str, str],
) -> list[str]:
    """Return the list of appendix LaTeX fragments in registration order."""
    parts: list[str] = []
    for tid in paper.table_ids():
        if tid in paper.appendix_ids and tid in fragments:
            parts.append(fragments[tid])
    for fid in paper.figure_ids():
        if fid in paper.appendix_ids and fid in fragments:
            parts.append(fragments[fid])
    return parts
