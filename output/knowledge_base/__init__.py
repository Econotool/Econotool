"""Knowledge base — concept entries and registry."""

from econtools.output.knowledge_base.registry import (  # noqa: F401
    KBEntry,
    load_entry,
    render_entry,
)

__all__ = ["KBEntry", "load_entry", "render_entry"]
