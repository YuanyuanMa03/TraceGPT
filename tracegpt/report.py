"""
report.py — Report generation utilities for TraceGPT.

Provides helper functions to generate, save, and format Markdown reports
from Tracer objects.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from tracegpt.tracer import Tracer


def export_report(
    tracer: Tracer,
    path: str,
    title: str = "TraceGPT Execution Trace",
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Generate a full Markdown report from a Tracer.

    Parameters
    ----------
    tracer : Tracer
        The tracer containing recorded operations.
    path : str
        Output file path.
    title : str
        Report title.
    metadata : dict, optional
        Additional metadata to include in the report header.

    Returns
    -------
    str
        The Markdown content written to disk.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    content = tracer.export_markdown(path, title=title)

    # If metadata provided, prepend it
    if metadata:
        meta_lines = ["## Metadata", ""]
        for k, v in metadata.items():
            meta_lines.append(f"- **{k}**: {v}")
        meta_lines.append("")
        meta_lines.append("---")
        meta_lines.append("")

        with open(path, "r", encoding="utf-8") as f:
            existing = f.read()

        # Insert after the title block
        header_end = existing.find("---")
        if header_end != -1:
            after_header = existing.find("\n", header_end + 3)
            new_content = existing[:after_header] + "\n" + "\n".join(meta_lines) + existing[after_header:]
        else:
            new_content = "\n".join(meta_lines) + existing

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        content = new_content

    return content


def print_trace_summary(tracer: Tracer) -> None:
    """
    Print a compact summary of the trace to stdout.

    Useful for quick debugging during development.
    """
    print(f"\n{'='*60}")
    print(f"  TraceGPT Trace Summary: {len(tracer)} operations")
    print(f"{'='*60}")
    for i, unit in enumerate(tracer.units, 1):
        print(f"\n  Step {i}: {unit.name}")
        print(f"    Formula: {unit.formula}")
        shapes_str = ", ".join(f"{k}={v}" for k, v in unit.shapes.items())
        print(f"    Shapes: {shapes_str}")
        print(f"    Why: {unit.explanation}")
    print(f"\n{'='*60}\n")
