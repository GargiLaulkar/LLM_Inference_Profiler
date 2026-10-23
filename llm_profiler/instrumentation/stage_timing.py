"""Reusable timing primitives for instrumented request stages.

This module provides a canonical ordered set of stage names and a monotonic-clock
based stopwatch utility. It intentionally does not claim to integrate with any
runtime lifecycle or exporter; it is a reusable measurement primitive that can be
wired into real request boundaries later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Callable, Dict, Iterable, Optional

STAGE_NAMES = [
    "tokenization",
    "queue_wait",
    "prefill",
    "decode",
    "detokenization",
    "network",
    "total",
]

STAGE_NAME_SET = set(STAGE_NAMES)


def validate_stage_name(stage_name: str) -> str:
    """Return a canonical stage name or raise ValueError for unknown stages."""
    if not isinstance(stage_name, str):
        raise ValueError("Stage name must be a string.")

    normalized = stage_name.strip()
    if normalized not in STAGE_NAME_SET:
        raise ValueError(f"Unknown stage name: {stage_name!r}")
    return normalized


@dataclass
class StageTimer:
    """Measure elapsed time for a named request stage with a monotonic clock.

    The class is intended to be a reusable primitive for instrumentation. It does
    not assume a lifecycle hook, a runtime, or a request trace collector. Instead,
    callers decide when to start and stop the stage timer.
    """

    stage_name: str
    clock: Optional[Callable[[], float]] = None
    _start_ts: Optional[float] = field(default=None, init=False, repr=False)
    _stop_ts: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.stage_name = validate_stage_name(self.stage_name)
        if self.clock is None:
            self.clock = monotonic

    def start(self) -> "StageTimer":
        """Start the timer using a monotonic clock and return self."""
        self._start_ts = float(self.clock())
        self._stop_ts = None
        return self

    def stop(self) -> float:
        """Stop the timer and return the elapsed duration in seconds."""
        if self._start_ts is None:
            raise RuntimeError("Cannot stop a timer before calling start().")

        self._stop_ts = float(self.clock())
        elapsed = self._stop_ts - self._start_ts
        if elapsed < 0:
            raise ValueError("Elapsed duration cannot be negative.")
        return elapsed

    @property
    def elapsed_seconds(self) -> float:
        """Return the elapsed duration in seconds if the timer has been stopped."""
        if self._start_ts is None or self._stop_ts is None:
            raise RuntimeError("Timer has not been stopped yet.")
        elapsed = self._stop_ts - self._start_ts
        if elapsed < 0:
            raise ValueError("Elapsed duration cannot be negative.")
        return elapsed

    def __enter__(self) -> "StageTimer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop the timer even when the context exits due to an exception."""
        self.stop()


def measure_stage_duration(
    stage_name: str,
    *,
    clock: Optional[Callable[[], float]] = None,
    duration: Optional[float] = None,
) -> float:
    """Measure a stage duration and return seconds.

    Parameters:
        stage_name: canonical stage name to validate.
        clock: optional injectable monotonic clock for deterministic testing.
        duration: optional duration to return without sleeping; for tests only.

    This helper is a small primitive useful for instrumentation integration code.
    The function does not automatically attach to vLLM lifecycle events; callers
    control when measurement begins and ends.
    """
    validate_stage_name(stage_name)

    if duration is not None:
        if duration < 0:
            raise ValueError("Duration cannot be negative.")
        return float(duration)

    if clock is None:
        clock = monotonic

    start = float(clock())
    end = float(clock())
    elapsed = end - start
    if elapsed < 0:
        raise ValueError("Elapsed duration cannot be negative.")
    return elapsed
