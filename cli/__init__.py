"""CLI package.

During the transition period, the full implementation lives in
``econtools._cli_monolith``.  In Phase D Step 15, command logic will
be fully split into ``cli/commands/``.

The ``econ`` script entry point (pyproject.toml) points here:
    ``econ = "econtools.cli:main"``
"""

from econtools._cli_monolith import main  # noqa: F401

__all__ = ["main"]
