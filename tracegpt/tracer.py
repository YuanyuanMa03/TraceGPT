"""
tracer.py — Core tracing infrastructure for TraceGPT.

Every operation in TraceGPT is recorded as a TraceUnit, capturing:
  - name: human-readable operation name
  - formula: mathematical formula (LaTeX-friendly string)
  - inputs: dict of input tensors (stored as lists for JSON safety)
  - output: the result tensor (stored as list)
  - shapes: dict mapping tensor names to their shapes
  - explanation: plain-language description of what happened

Usage:
    tracer = Tracer()
    with tracer.trace("softmax", formula="softmax(x) = exp(x) / sum(exp(x))",
                      inputs={"x": x}, output=result,
                      explanation="Convert logits to probabilities"):
        ...
    tracer.export_markdown("reports/my_report.md")
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TraceUnit:
    """A single recorded operation in the trace."""

    name: str
    formula: str
    inputs: dict[str, Any]
    output: Any
    shapes: dict[str, tuple[int, ...]]
    explanation: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3])

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "formula": self.formula,
            "inputs": {k: _tensor_to_list(v) for k, v in self.inputs.items()},
            "output": _tensor_to_list(self.output),
            "shapes": self.shapes,
            "explanation": self.explanation,
            "timestamp": self.timestamp,
        }


def _tensor_to_list(obj: Any) -> Any:
    """Safely convert numpy arrays to nested lists; leave scalars and lists alone."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def _shape_of(obj: Any) -> tuple[int, ...]:
    """Return shape of a numpy array, scalar, or nested list."""
    if isinstance(obj, np.ndarray):
        return obj.shape
    if isinstance(obj, (list, tuple)):
        arr = np.array(obj)
        return arr.shape
    return ()


class Tracer:
    """
    Collects TraceUnits to form a complete execution trace.

    Methods
    -------
    trace(name, formula, inputs, output, explanation)
        Record a single operation step.

    export_markdown(path)
        Write the full trace as a human-readable Markdown report.

    export_dict()
        Return the trace as a list of dicts (for programmatic access).
    """

    def __init__(self) -> None:
        self.units: list[TraceUnit] = []

    def trace(
        self,
        name: str,
        formula: str,
        inputs: dict[str, Any],
        output: Any,
        explanation: str,
    ) -> TraceUnit:
        """
        Record a single traced operation.

        Parameters
        ----------
        name : str
            Short operation name, e.g. "softmax", "linear".
        formula : str
            Mathematical formula string (LaTeX-compatible).
        inputs : dict[str, Any]
            Named input tensors.
        output : Any
            The output tensor.
        explanation : str
            Human-readable description of what happened and why.

        Returns
        -------
        TraceUnit
            The recorded trace unit (also appended to self.units).
        """
        # Build shapes dict from inputs + output
        shapes: dict[str, tuple[int, ...]] = {}
        for k, v in inputs.items():
            shapes[k] = _shape_of(v)
        shapes["output"] = _shape_of(output)

        unit = TraceUnit(
            name=name,
            formula=formula,
            inputs=inputs,
            output=output,
            shapes=shapes,
            explanation=explanation,
        )
        self.units.append(unit)
        return unit

    def export_markdown(self, path: str, title: str = "TraceGPT Execution Trace") -> str:
        """
        Export the full trace as a Markdown report.

        Parameters
        ----------
        path : str
            File path to write the report.
        title : str
            Title for the report.

        Returns
        -------
        str
            The Markdown content that was written.
        """
        lines: list[str] = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total operations:** {len(self.units)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for i, unit in enumerate(self.units, 1):
            lines.append(f"## Step {i}: {unit.name}")
            lines.append("")
            lines.append(f"**Formula:** `{unit.formula}`")
            lines.append("")
            lines.append(f"**Explanation:** {unit.explanation}")
            lines.append("")

            # Input shapes
            lines.append("### Shapes")
            lines.append("")
            for tensor_name, shape in unit.shapes.items():
                lines.append(f"- **{tensor_name}**: {shape}")
            lines.append("")

            # Input values
            lines.append("### Inputs")
            lines.append("")
            for k, v in unit.inputs.items():
                formatted = _format_array(v)
                lines.append(f"**{k}:**")
                lines.append(f"```")
                lines.append(formatted)
                lines.append("```")
                lines.append("")

            # Output
            lines.append("### Output")
            lines.append("")
            formatted_out = _format_array(unit.output)
            lines.append(f"```")
            lines.append(formatted_out)
            lines.append("```")
            lines.append("")

            lines.append("---")
            lines.append("")

        content = "\n".join(lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return content

    def export_dict(self) -> list[dict]:
        """Return all trace units as a list of dicts."""
        return [u.to_dict() for u in self.units]

    def __len__(self) -> int:
        return len(self.units)

    def __repr__(self) -> str:
        return f"Tracer(steps={len(self.units)})"


def _format_array(obj: Any, precision: int = 6) -> str:
    """Pretty-format a numpy array or nested list for Markdown."""
    if isinstance(obj, np.ndarray):
        # Handle string arrays
        if obj.dtype.kind in ('U', 'S', 'O'):
            return str(obj.tolist())
        if obj.ndim == 0:
            return str(float(obj))
        return _format_array(obj.tolist(), precision=precision)
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return "[]"
        # Check if it's a flat list of numbers
        if isinstance(obj[0], (int, float, np.integer, np.floating)):
            return "[" + ", ".join(f"{float(v):.{precision}f}" for v in obj) + "]"
        # Check if it's strings
        if isinstance(obj[0], str):
            return str(obj)
        # Nested list (2D+)
        rows = []
        for row in obj:
            rows.append("  " + _format_array(row, precision=precision))
        return "[\n" + ",\n".join(rows) + "\n]"
    if isinstance(obj, (str, int, float)):
        return str(obj)
    return str(obj)
