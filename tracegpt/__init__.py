"""
TraceGPT — A calculator-verifiable Transformer learning framework.

Turns every hidden tensor operation into a readable, auditable, and reproducible trace.

formula → hand calculation → NumPy code → tensor trace → Markdown report
"""

__version__ = "0.1.0"

from tracegpt.tracer import Tracer, TraceUnit
from tracegpt.ops import softmax, causal_mask, layer_norm, linear, relu, sinusoidal_position_encoding, multi_head_attention
from tracegpt.report import export_report
