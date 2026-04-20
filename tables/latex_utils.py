"""LaTeX helpers for compiling and combining .tex outputs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from econtools._core.formatting import _latex_escape


def combine_tex(
    tex_paths: list[Path],
    out_tex: Path,
    *,
    margin: str = "1in",
) -> Path:
    """Combine multiple LaTeX fragments into a single wrapper .tex file."""
    header = [
        r"\documentclass{article}",
        rf"\usepackage[margin={margin}]{{geometry}}",
        r"\usepackage{array}",
        r"\usepackage{longtable}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.9}",
    ]
    footer = [r"\end{document}"]

    contents: list[str] = []
    for idx, path in enumerate(tex_paths):
        title = _title_from_path(Path(path))
        if idx > 0:
            contents.append(r"\clearpage")
        contents.append(r"\section*{" + _latex_escape(title) + r"}")
        contents.append("")
        contents.extend(Path(path).read_text(encoding="utf-8").splitlines())
        contents.append("")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(header + contents + footer), encoding="utf-8")
    return out_tex


def compile_tex_to_pdf(
    tex_path: Path,
    *,
    engine: str | None = None,
    workdir: Path | None = None,
) -> Path:
    """Compile a .tex file to PDF.

    Engine resolution order:
    1. ``engine`` argument
    2. ``ECON_LATEX_ENGINE`` environment variable
    3. ``pdflatex``, ``latexmk``, ``tectonic`` via ``shutil.which``
    4. Path in ``ECON_LATEX_FALLBACK_PATH`` environment variable
    """
    tex_path = Path(tex_path)
    workdir = Path(workdir) if workdir else tex_path.parent
    engine = engine or os.environ.get("ECON_LATEX_ENGINE") or ""
    fallback_path = os.environ.get("ECON_LATEX_FALLBACK_PATH", "")
    candidates = [engine] if engine else [
        "pdflatex",
        "latexmk",
        "tectonic",
        fallback_path,
    ]
    cmd = None
    for name in candidates:
        if not name:
            continue
        if Path(name).exists():
            cmd = str(name)
            break
        exe = shutil.which(name)
        if exe:
            cmd = exe
            break
    if not cmd:
        raise RuntimeError(
            "No LaTeX engine found. Install latexmk/pdflatex/tectonic, "
            "set ECON_LATEX_ENGINE, or set ECON_LATEX_FALLBACK_PATH."
        )

    if cmd.endswith("latexmk") or cmd.endswith("latexmk.exe"):
        argv = [cmd, "-pdf", "-interaction=nonstopmode", tex_path.name]
    elif cmd.endswith("tectonic") or cmd.endswith("tectonic.exe"):
        argv = [cmd, tex_path.name]
    else:
        argv = [cmd, "-interaction=nonstopmode", tex_path.name]

    pdf_path = tex_path.with_suffix(".pdf")
    # Remove stale PDF from a prior run so a silent compile failure cannot
    # masquerade as success by leaving the old artifact in place.
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError:
            pass

    subprocess.run(argv, cwd=str(workdir), check=False)
    if not pdf_path.exists():
        raise RuntimeError("PDF compilation failed; see LaTeX log for details.")
    # Second pass to resolve \ref / \label cross-references.  latexmk and
    # tectonic handle this internally; for bare pdflatex we rerun once.
    if not (cmd.endswith("latexmk") or cmd.endswith("latexmk.exe")
            or cmd.endswith("tectonic") or cmd.endswith("tectonic.exe")):
        try:
            source = tex_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            source = ""
        if r"\ref{" in source or r"\pageref{" in source or r"\cite{" in source:
            subprocess.run(argv, cwd=str(workdir), check=False)
    return pdf_path


def open_pdf(path: Path) -> None:
    """Open a PDF with the OS default viewer."""
    os.startfile(str(path))


def _title_from_path(path: Path) -> str:
    name = path.stem.replace("_", " ").strip()
    if not name:
        return "Output"
    return " ".join(word.capitalize() for word in name.split())
