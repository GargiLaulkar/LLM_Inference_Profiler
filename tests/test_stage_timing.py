"""Focused tests for stage definitions and monotonic timing utilities."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from llm_profiler.instrumentation import (
    STAGE_NAMES,
    StageTimer,
    measure_stage_duration,
)


EXPECTED_STAGE_NAMES = [
    "tokenization",
    "queue_wait",
    "prefill",
    "decode",
    "detokenization",
    "network",
    "total",
]


def test_stage_names_exact_order():
    """The canonical stage list must match the shared project contract exactly."""
    assert STAGE_NAMES == EXPECTED_STAGE_NAMES


def test_stage_timer_tracks_valid_stage_duration():
    """A valid stage timer should report a positive elapsed duration."""
    clock_values = iter([10.0, 15.25])

    timer = StageTimer("prefill", clock=lambda: next(clock_values))
    timer.start()
    elapsed = timer.stop()

    assert elapsed == pytest.approx(5.25)


def test_unknown_stage_name_is_rejected():
    """Unknown stage names should be rejected immediately."""
    with pytest.raises(ValueError, match="Unknown stage name"):
        StageTimer("bogus_stage")


def test_elapsed_duration_is_non_negative():
    """Elapsed time must never be negative, even when a fake clock is misordered."""
    clock_values = iter([12.0, 11.5])
    timer = StageTimer("network", clock=lambda: next(clock_values))
    timer.start()

    with pytest.raises(ValueError, match="cannot be negative"):
        timer.stop()


def test_context_manager_stops_timer_after_exception():
    """The timer should still be stopped if the context exits via an exception."""
    clock_values = iter([100.0, 110.0])

    with pytest.raises(RuntimeError, match="boom"):
        with StageTimer("decode", clock=lambda: next(clock_values)) as timer:
            raise RuntimeError("boom")

    assert timer._stop_ts == pytest.approx(110.0)
    assert timer.elapsed_seconds == pytest.approx(10.0)


def test_measure_stage_duration_supports_injectable_clock():
    """Deterministic testing should not require real sleeping or wall-clock time."""
    values = iter([1.0, 3.5])
    elapsed = measure_stage_duration("tokenization", clock=lambda: next(values))
    assert elapsed == pytest.approx(2.5)


def test_measure_stage_duration_rejects_unknown_stage_name():
    """The helper should guard against incorrect stage names before measuring."""
    with pytest.raises(ValueError, match="Unknown stage name"):
        measure_stage_duration("not_a_stage")
