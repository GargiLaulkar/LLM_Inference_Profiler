"""Tests for measurable request lifecycle boundaries available in the current codebase."""

import math

import pytest

from llm_profiler.instrumentation import (
    RequestTrace,
    create_client_total_trace,
    create_request_trace_from_interval,
    create_server_total_trace,
    elapsed_seconds,
)


def test_create_client_total_trace_records_total_duration_only():
    """The client can measure end-to-end request duration without inventing stages."""
    trace = create_client_total_trace("req-client", 10.0, 15.25)

    assert trace.request_id == "req-client"
    assert trace.total_duration_s == pytest.approx(5.25)
    assert trace.stage_durations == {}


def test_create_server_total_trace_records_server_elapsed_time():
    """The reference server can measure its own processing interval without stage guesses."""
    trace = create_server_total_trace("req-server", 4.0, 9.5)

    assert trace.request_id == "req-server"
    assert trace.total_duration_s == pytest.approx(5.5)
    assert trace.stage_durations == {}


def test_elapsed_seconds_rejects_invalid_bounds():
    """Elapsed boundaries must be finite and end must not precede start."""
    with pytest.raises(ValueError, match="finite"):
        elapsed_seconds(float("nan"), 2.0)

    with pytest.raises(ValueError, match="earlier"):
        elapsed_seconds(3.0, 2.0)

    with pytest.raises(TypeError):
        elapsed_seconds("bad", 1.0)


def test_request_trace_from_interval_can_record_a_known_stage_only_when_explicitly_provided():
    """This helper must not fabricate stages unless the caller explicitly supplies a known one."""
    trace = create_request_trace_from_interval("req-explicit", 3.0, 8.0, stage_name="total")

    assert trace.total_duration_s == pytest.approx(5.0)
    assert trace.get_stage_duration("total") == pytest.approx(5.0)


def test_unknown_stage_name_is_rejected_when_recorded_in_lifecycle_helper():
    """The lifecycle helper should reuse the canonical stage validation rules."""
    with pytest.raises(ValueError, match="Unknown stage name"):
        create_request_trace_from_interval("req-bad", 1.0, 2.0, stage_name="not_a_stage")


def test_total_duration_remains_separate_from_stage_measurements():
    """This helper should never assume the stage total equals the overall request duration."""
    trace = create_request_trace_from_interval("req-separated", 0.0, 10.0)
    trace.record_stage_duration("decode", 4.0)
    trace.record_stage_duration("network", 1.0)

    assert trace.total_duration_s == pytest.approx(10.0)
    assert trace.get_stage_duration("decode") == pytest.approx(4.0)
    assert trace.get_stage_duration("network") == pytest.approx(1.0)
    assert trace.total_duration_s != sum(trace.stage_durations.values())
