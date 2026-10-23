"""Lifecycle-boundary helpers for measurable request timings.

This module intentionally records only those request timing boundaries that are
already observable in the current project without inventing internal vLLM stages.
The reference client and reference server can both measure total elapsed time with
monotonic clocks; internal GPU prefill/decode stages remain unavailable in the
current implementation and are therefore not assumed or fabricated here.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from .request_trace import RequestTrace
from .stage_timing import validate_stage_name


def elapsed_seconds(start_s: float, end_s: float, *, label: str = "elapsed") -> float:
    """Return (end_s - start_s) in seconds using a monotonic-style interval.

    This helper intentionally does not use wall-clock timestamps for measurement.
    Callers are expected to pass values obtained from monotonic clocks.
    """
    if isinstance(start_s, bool) or isinstance(end_s, bool):
        raise TypeError(f"{label} start and end values must be real numbers.")

    try:
        start_value = float(start_s)
        end_value = float(end_s)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} values must be real numbers.") from exc

    if not math.isfinite(start_value) or not math.isfinite(end_value):
        raise ValueError(f"{label} values must be finite.")
    if end_value < start_value:
        raise ValueError(f"{label} end time cannot be earlier than start time.")

    duration = end_value - start_value
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"{label} duration cannot be negative or non-finite.")
    return duration


def create_request_trace_from_interval(
    request_id: str,
    start_s: float,
    end_s: float,
    *,
    stage_name: Optional[str] = None,
    clock: Optional[Callable[[], float]] = None,
) -> RequestTrace:
    """Create a trace from a monotonic interval and optionally record an explicit stage.

    The current implementation measures only actual request boundaries available in
    the project. This means we can record total request duration and optionally a
    tokenized-stage label only when a caller explicitly chooses to record one.
    Internal GPU inference stages remain unavailable and must not be invented.
    """
    duration = elapsed_seconds(start_s, end_s, label="request interval")
    trace = RequestTrace(request_id=request_id, total_duration_s=duration)

    if stage_name is not None:
        normalized = validate_stage_name(stage_name)
        trace.record_stage_duration(normalized, duration)

    return trace


def create_client_total_trace(request_id: str, start_s: float, end_s: float) -> RequestTrace:
    """Record client-observed end-to-end request duration for one request."""
    return create_request_trace_from_interval(request_id, start_s, end_s)


def create_server_total_trace(request_id: str, start_s: float, end_s: float) -> RequestTrace:
    """Record server-side processing duration inside the reference request handler."""
    return create_request_trace_from_interval(request_id, start_s, end_s)
