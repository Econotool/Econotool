"""Full LaTeX document assembly from fragments.

Public API
----------
PageSetup           — paper/margin/landscape configuration
assemble_document   — build a complete .tex document from fragments
assemble_appendix   — build an appendix with one fragment per page
write_document      — assemble + write to file
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from econtools.output.latex.journal_profiles import JournalProfile


@dataclass(frozen=True)
class PageSetup:
    """Page geometry and layout configuration.

    Parameters
    ----------
    paper:
        Paper size (e.g. ``"a4paper"``, ``"letterpaper"``).
    margin:
        Uniform margin (e.g. ``"1in"``, ``"2.5cm"``).  If ``margins`` is
        also provided, ``margin`` is ignored.
    margins:
        Per-side margins dict with keys ``"top"``, ``"bottom"``, ``"left"``,
        ``"right"`` (LaTeX length strings).
    landscape:
        Set the entire document to landscape orientation.
    """

    paper: str = "a4paper"
    margin: str = "1in"
    margins: dict[str, str] = field(default_factory=dict)
    landscape: bool = False

    def to_geometry_options(self) -> str:
        """Return the options string for ``\\usepackage[...]{geometry}``."""
        parts = [self.paper]
        if self.margins:
            for side, val in self.margins.items():
                parts.append(f"{side}={val}")
        else:
            parts.append(f"margin={self.margin}")
        if self.landscape:
            parts.append("landscape")
        return ",".join(parts)


def assemble_document(
    fragments: list[str],
    profile: JournalProfile,
    *,
    title: str | None = None,
    author: str | None = None,
    abstract: str | None = None,
    page_setup: PageSetup | None = None,
) -> str:
    """Assemble a full LaTeX document from table/figure fragments.

    Parameters
    ----------
    fragments:
        List of LaTeX fragment strings (tables, figures).
    profile:
        :class:`JournalProfile` controlling document style.
    title:
        Optional document title.
    author:
        Optional author string.
    abstract:
        Optional abstract text.
    page_setup:
        Optional :class:`PageSetup` for geometry configuration.  If None,
        no ``geometry`` package is loaded (uses document class defaults).

    Returns
    -------
    str
        Complete LaTeX document source.
    """
    opts = "[" + ",".join(profile.class_options) + "]" if profile.class_options else ""
    lines: list[str] = [
        rf"\documentclass{opts}{{{profile.document_class}}}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{graphicx}",
        r"\usepackage{threeparttable}",
        r"\usepackage{caption}",
        r"\usepackage{subcaption}",
        # longtable supports tables that break across pages.  Zero-cost
        # when unused; required whenever a ResultsTable or SummaryTable
        # is rendered with ``long_table=True``.
        r"\usepackage{longtable}",
        # adjustbox lets composite tabulars shrink to panel width without
        # stretching narrow tables vertically (as \resizebox{..}{!} would).
        r"\usepackage{adjustbox}",
    ]
    if profile.float_specifier:
        lines.append(r"\usepackage{float}")

    # Geometry
    if page_setup is not None:
        geom_opts = page_setup.to_geometry_options()
        lines.append(rf"\usepackage[{geom_opts}]{{geometry}}")

    # Landscape support (per-page, not whole document)
    lines.append(r"\usepackage{pdflscape}")

    # Caption font styling (journal-specific; e.g. AER uses small caps).
    labelfont = getattr(profile, "caption_labelfont", "") or ""
    textfont = getattr(profile, "caption_textfont", "") or ""
    if labelfont or textfont:
        parts = []
        if labelfont:
            parts.append(f"labelfont={labelfont}")
        if textfont:
            parts.append(f"textfont={textfont}")
        lines.append(rf"\captionsetup{{{','.join(parts)}}}")

    # Profile-specific extra packages (e.g. hyperref for econsocart).
    for pkg in getattr(profile, "packages", []) or []:
        # Accept either a bare name ("natbib") or a full tail
        # ("[opts]{hyperref}").  Bare names are wrapped in braces.
        tail = pkg.strip()
        if tail.startswith("[") or tail.startswith("{"):
            lines.append(rf"\usepackage{tail}")
        else:
            lines.append(rf"\usepackage{{{tail}}}")

    # Stop LaTeX from vertically justifying the last page of content.
    lines.append(r"\raggedbottom")
    # Force floats on float-only pages to stick to the top instead of
    # being centered vertically (the default \@fpsep glue has "fil"
    # stretch at the top, pushing floats to the middle of the page).
    lines.append(r"\makeatletter")
    lines.append(r"\setlength{\@fptop}{0pt}")
    lines.append(r"\setlength{\@fpsep}{12pt plus 0fil}")
    lines.append(r"\setlength{\@fpbot}{0pt plus 1fil}")
    lines.append(r"\makeatother")

    lines.append(r"\begin{document}")

    # Journal-specific preamble hooks (e.g. \journalname{...}).
    for hook in getattr(profile, "preamble_hooks", []) or []:
        lines.append(hook)

    if title:
        lines.append(rf"\title{{{title}}}")
    if author:
        lines.append(rf"\author{{{author}}}")
    if title or author:
        lines.append(r"\maketitle")
    if abstract:
        lines.append(r"\begin{abstract}")
        lines.append(abstract)
        lines.append(r"\end{abstract}")
    for fragment in fragments:
        lines.append("")
        lines.append(fragment)
    lines.append(r"\end{document}")
    return "\n".join(lines)


def assemble_appendix(
    fragments: list[str],
    profile: JournalProfile,
    *,
    page_setup: PageSetup | None = None,
    title: str | None = "Appendix",
) -> str:
    """Assemble an appendix document with one fragment per page.

    Each fragment is separated by ``\\clearpage`` to ensure one table/figure
    per page — ideal for referee appendices.

    Parameters
    ----------
    fragments:
        List of LaTeX fragment strings.
    profile:
        :class:`JournalProfile` controlling document style.
    page_setup:
        Optional :class:`PageSetup` for geometry configuration.
    title:
        Optional appendix title.

    Returns
    -------
    str
        Complete LaTeX document source.
    """
    separated = []
    for i, frag in enumerate(fragments):
        if i > 0:
            separated.append(r"\clearpage")
        separated.append(frag)

    # Build as a single joined fragment so assemble_document handles preamble
    body = "\n".join(separated)
    return assemble_document(
        [body],
        profile,
        title=title,
        page_setup=page_setup,
    )


def write_document(
    fragments: list[str],
    out_path: Path,
    profile: JournalProfile,
    **kwargs,
) -> Path:
    """Write a complete LaTeX document to *out_path*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = assemble_document(fragments, profile, **kwargs)
    out_path.write_text(source, encoding="utf-8")
    return out_path
