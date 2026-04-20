"""Journal-specific LaTeX profiles for typesetting regression output.

Public API
----------
JournalProfile
ECONOMETRICA  (alias for ECTA; retained for backward compatibility)
ECTA          (Econometrica)
QE            (Quantitative Economics)
TE            (Theoretical Economics)
AER           (American Economic Review)
DEFAULT       (plain ``article`` class)
resolve_profile(name)  — case-insensitive string lookup
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JournalProfile:
    """Parameters controlling LaTeX output style for a specific journal.

    Parameters
    ----------
    name:
        Journal name (human-readable).
    document_class:
        LaTeX document class (e.g. ``'article'``, ``'econsocart'``).
    class_options:
        Options passed to the document class.
    table_numbering:
        ``'roman'`` (TABLE I) or ``'arabic'`` (Table 1).
    caption_position:
        ``'above'`` or ``'below'`` the table body.
    notes_command:
        LaTeX command for table notes
        (``'legend'`` for Econometrica, ``'footnotesize'`` for others).
    float_specifier:
        Float placement specifier (e.g. ``'[H]'``); ``None`` omits it.
    star_symbols:
        Ordered list of star symbols from least to most significant.
    significance_levels:
        Ordered p-value thresholds matching ``star_symbols``.
    bst_file:
        BibTeX style file.
    packages:
        Extra ``\\usepackage{...}`` lines required by the profile beyond
        the base set already emitted by :func:`assemble_document`
        (``booktabs``, ``array``, ``graphicx``, ``threeparttable``,
        ``caption``, ``float``, ``pdflscape``). Each entry is a bare
        package name or the full tail of a ``\\usepackage`` command
        (e.g. ``"[colorlinks]{hyperref}"``).
    fallback_class:
        Document class used when the primary ``document_class`` is not
        installed on the compiling system. Defaults to ``'article'``.
    fallback_options:
        Class options paired with ``fallback_class``.
    preamble_hooks:
        Extra raw LaTeX commands inserted after ``\\begin{document}`` but
        before ``\\maketitle``. Used for journal front-matter commands
        such as ``\\journalname{...}`` or ``\\articletype{...}``.
    notes:
        Free-form human notes (install instructions, links, caveats).
    """

    name: str
    document_class: str = "article"
    class_options: list[str] = field(default_factory=list)
    table_numbering: str = "arabic"   # "roman" | "arabic"
    caption_position: str = "above"   # "above" | "below"
    notes_command: str = "footnotesize"
    float_specifier: str | None = "[H]"
    star_symbols: list[str] = field(default_factory=lambda: ["*", "**", "***"])
    significance_levels: list[float] = field(default_factory=lambda: [0.1, 0.05, 0.01])
    bst_file: str = "plain"
    packages: list[str] = field(default_factory=list)
    fallback_class: str = "article"
    fallback_options: list[str] = field(default_factory=lambda: ["11pt"])
    preamble_hooks: list[str] = field(default_factory=list)
    notes: str = ""
    # Table aesthetics
    table_double_rule: bool = False           # AER/ILR style: \hline\hline
    caption_labelfont: str = "bf"             # "bf" | "sc" | "it" | ""
    caption_textfont: str = ""                # applied to caption text

    # TODO(econtools): profile — validate significance_levels / star_symbols lengths match

    @property
    def sample_documentclass_line(self) -> str:
        """Return the exact ``\\documentclass`` line this profile would emit.

        Useful for docs, debugging, and quick sanity checks.
        """
        opts = "[" + ",".join(self.class_options) + "]" if self.class_options else ""
        return rf"\documentclass{opts}{{{self.document_class}}}"

    def info(self) -> str:
        """Return a short multi-line human-readable profile summary.

        Includes install instructions when present.
        """
        lines = [
            f"JournalProfile: {self.name}",
            f"  document class : {self.document_class}"
            + (f" [{','.join(self.class_options)}]" if self.class_options else ""),
            f"  bib style      : {self.bst_file}",
            f"  fallback class : {self.fallback_class}",
        ]
        if self.packages:
            lines.append("  extra packages : " + ", ".join(self.packages))
        if self.preamble_hooks:
            lines.append("  preamble hooks : " + "; ".join(self.preamble_hooks))
        if self.notes:
            lines.append("  notes          : " + self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_ECONSOC_HYPERREF = (
    "[colorlinks,citecolor=blue,linkcolor=blue,urlcolor=blue,pagebackref]{hyperref}"
)


ECTA = JournalProfile(
    name="Econometrica",
    document_class="econsocart",
    # Source: ecta_template.tex line 1 — \documentclass[ecta,nameyear,draft]{econsocart}
    class_options=["ecta", "nameyear"],
    table_numbering="roman",
    caption_position="above",
    notes_command="legend",
    float_specifier=None,                 # Econometrica disallows [H]
    star_symbols=["*", "**", "***"],
    significance_levels=[0.1, 0.05, 0.01],
    # Source: https://github.com/vtex-soft/texsupport.econometricsociety-ecta — ships econsoc.bst
    bst_file="econsoc",
    # Source: ecta_template.tex preamble — one \RequirePackage line before frontmatter
    packages=[_ECONSOC_HYPERREF],
    fallback_class="article",
    fallback_options=["11pt"],
    preamble_hooks=[],
    notes=(
        "Install via https://github.com/vtex-soft/texsupport.econometricsociety-ecta "
        "(econsocart.cls, econsocart.cfg, econsoc.bst)."
    ),
)

# Retain ECONOMETRICA as the documented public name; identical to ECTA.
ECONOMETRICA = ECTA


QE = JournalProfile(
    name="Quantitative Economics",
    document_class="econsocart",
    # Source: qe_template.tex line 1 — \documentclass[qe,nameyear,draft]{econsocart}
    class_options=["qe", "nameyear"],
    table_numbering="arabic",
    caption_position="above",
    notes_command="legend",
    float_specifier=None,
    star_symbols=["*", "**", "***"],
    significance_levels=[0.1, 0.05, 0.01],
    # Source: https://github.com/vtex-soft/texsupport.econometricsociety-qe — ships qe.bst
    bst_file="qe",
    packages=[_ECONSOC_HYPERREF],
    fallback_class="article",
    fallback_options=["11pt"],
    preamble_hooks=[],
    notes=(
        "Install via https://github.com/vtex-soft/texsupport.econometricsociety-qe "
        "(econsocart.cls, econsocart.cfg, qe.bst)."
    ),
)


TE = JournalProfile(
    name="Theoretical Economics",
    document_class="econsocart",
    # Source: te_template.tex line 1 — \documentclass[te,nameyear,draft]{econsocart}
    class_options=["te", "nameyear"],
    table_numbering="arabic",
    caption_position="above",
    notes_command="legend",
    float_specifier=None,
    star_symbols=["*", "**", "***"],
    significance_levels=[0.1, 0.05, 0.01],
    # Source: https://github.com/vtex-soft/texsupport.econometricsociety-te — ships te.bst
    bst_file="te",
    packages=[_ECONSOC_HYPERREF],
    fallback_class="article",
    fallback_options=["11pt"],
    preamble_hooks=[],
    notes=(
        "Install via https://github.com/vtex-soft/texsupport.econometricsociety-te "
        "(econsocart.cls, econsocart.cfg, te.bst)."
    ),
)


AER = JournalProfile(
    name="American Economic Review",
    document_class="aea",
    class_options=[],
    table_numbering="arabic",
    caption_position="above",
    notes_command="footnotesize",
    float_specifier="[H]",
    star_symbols=["*", "**", "***"],
    significance_levels=[0.1, 0.05, 0.01],
    bst_file="aea",
    packages=[],
    fallback_class="article",
    fallback_options=["11pt"],
    preamble_hooks=[],
    notes="Install via the AEA LaTeX template distribution (aea.cls, aea.bst).",
    table_double_rule=True,
    caption_labelfont="sc",
    caption_textfont="sc",
)


DEFAULT = JournalProfile(
    name="Default",
    document_class="article",
    class_options=["11pt"],
    bst_file="plain",
    fallback_class="article",
    fallback_options=["11pt"],
)


# ---------------------------------------------------------------------------
# String lookup (case-insensitive)
# ---------------------------------------------------------------------------

_PROFILE_REGISTRY: dict[str, "JournalProfile"] = {
    "ecta": ECTA,
    "econometrica": ECONOMETRICA,
    "qe": QE,
    "quantitative_economics": QE,
    "te": TE,
    "theoretical_economics": TE,
    "aer": AER,
    "american_economic_review": AER,
    "article": DEFAULT,
    "default": DEFAULT,
}


def resolve_profile(name: "str | JournalProfile") -> "JournalProfile":
    """Resolve *name* to a :class:`JournalProfile`.

    Accepts either a :class:`JournalProfile` (returned as-is) or a
    case-insensitive string matching one of the registered journal keys
    (``"qe"``, ``"te"``, ``"ecta"``, ``"econometrica"``, ``"aer"``,
    ``"american_economic_review"``, ``"article"``, ``"default"``,
    ``"quantitative_economics"``, ``"theoretical_economics"``).

    Raises
    ------
    ValueError
        When the name does not match any registered profile.
    """
    if isinstance(name, JournalProfile):
        return name
    key = str(name).strip().lower()
    if key not in _PROFILE_REGISTRY:
        raise ValueError(
            f"Unknown journal profile '{name}'. "
            f"Available: {sorted(_PROFILE_REGISTRY)}"
        )
    return _PROFILE_REGISTRY[key]


# TODO(econtools): profile — add QJE, RES, JPE profiles
