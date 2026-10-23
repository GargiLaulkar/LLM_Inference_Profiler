"""Request trace data model for stage-duration recording.

This module stores request-level metadata and per-stage timings. It intentionally
operates only on explicit values supplied by calling code; it does not measure or
infer lifecycle stages automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Dict, Optional

from .stage_timing import STAGE_NAMES, validate_stage_name


def _validate_duration(value: float, label: str) -> float:
    """Validate a stage or total duration represented in seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number in seconds.")

    duration = float(value)
    if not isfinite(duration):
        raise ValueError(f"{label} must be finite.")
    if duration < 0:
        raise ValueError(f"{label} cannot be negative.")
    return duration


@dataclass
class RequestTrace:
    """Store one request's request ID and optional per-stage timings.

    The total request duration is kept conceptually separate from the individual
    stage durations. The trace does not attempt to infer or validate that stage
    durations sum to total time.
    """

    request_id: str
    total_duration_s: Optional[float] = None
    stage_durations: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError("request_id must be a string.")

        self.request_id = self.request_id.strip()
        if not self.request_id:
            raise ValueError("request_id must be non-empty.")

        if self.total_duration_s is not None:
            self.total_duration_s = _validate_duration(self.total_duration_s, "total_duration_s")

        if self.stage_durations:
            for stage_name, duration in list(self.stage_durations.items()):
                self.record_stage_duration(stage_name, duration)

    def set_total_duration(self, total_duration_s: float) -> float:
        """Set the total request duration in seconds and return it."""
        validated = _validate_duration(total_duration_s, "total_duration_s")
        self.total_duration_s = validated
        return validated

    def record_stage_duration(self, stage_name: str, duration_s: float) -> float:
        """Record a validated stage duration for this request.

        Re-recording the same stage replaces the previous value rather than
        creating multiple entries for the same stage.
        """
        normalized = validate_stage_name(stage_name)
        validated = _validate_duration(duration_s, f"duration for stage '{normalized}'")
        self.stage_durations[normalized] = validated
        return validated

    def get_stage_duration(self, stage_name: str) -> Optional[float]:
        """Return the recorded duration for a stage, or None if not present."""
        normalized = validate_stage_name(stage_name)
        return self.stage_durations.get(normalized)

    def ordered_stage_durations(self) -> Dict[str, float]:
        """Return all recorded stage durations in canonical stage order."""
        ordered: Dict[str, float] = {}
        for stage_name in STAGE_NAMES:
            if stage_name in self.stage_durations:
                ordered[stage_name] = self.stage_durations[stage_name]
        return ordered

    def copy(self) -> "RequestTrace":
        """Return a shallow copy of the trace and its recorded stage values."""
        return RequestTrace(
            request_id=self.request_id,
            total_duration_s=self.total_duration_s,
            stage_durations=dict(self.stage_durations),
        )
