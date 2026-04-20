"""CLI entry point (canonical location).

During the transition period, delegates to ``econtools._cli_monolith``.
In Phase D Step 15, this file will contain the full ``build_parser()``
and ``main()`` with commands dispatched from ``commands/``.

TODO(econtools): cli-cmd — migrate each subcommand to its own file in cli/commands/
"""

from __future__ import annotations

import sys

from econtools._cli_monolith import main as _main  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    """Dispatch CLI commands."""
    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
