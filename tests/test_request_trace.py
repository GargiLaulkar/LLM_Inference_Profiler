"""Unit tests for request trace storage and per-stage timing data."""

import math

import pytest

from llm_profiler.instrumentation import RequestTrace, STAGE_NAMES


def test_trace_created_with_request_id():
    """A request trace must identify a request and start with empty stage data."""
    trace = RequestTrace("req-123")

    assert trace.request_id == "req-123"
    assert trace.total_duration_s is None
    assert trace.stage_durations == {}


def test_record_and_retrieve_valid_stage_durations():
    """Stages should be stored in a predictable dictionary keyed by canonical names."""
    trace = RequestTrace("req-456")

    trace.record_stage_duration("prefill", 0.25)
    trace.record_stage_duration("decode", 1.5)
    trace.record_stage_duration("network", 0.1)

    assert trace.get_stage_duration("prefill") == pytest.approx(0.25)
    assert trace.get_stage_duration("decode") == pytest.approx(1.5)
    assert trace.get_stage_duration("network") == pytest.approx(0.1)
    assert trace.ordered_stage_durations() == {
        "prefill": 0.25,
        "decode": 1.5,
        "network": 0.1,
    }


def test_unknown_stage_name_is_rejected():
    """Only canonical stage names are accepted."""
    trace = RequestTrace("req-unknown")

    with pytest.raises(ValueError, match="Unknown stage name"):
        trace.record_stage_duration("invalid_stage", 0.1)

    with pytest.raises(ValueError, match="Unknown stage name"):
        trace.get_stage_duration("invalid_stage")


@pytest.mark.parametrize("bad_duration", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_reject_invalid_duration_values(bad_duration):
    """Negative, NaN, and infinite durations are invalid in seconds."""
    trace = RequestTrace("req-invalid")

    with pytest.raises((ValueError, TypeError)):
        trace.record_stage_duration("tokenization", bad_duration)

    with pytest.raises((ValueError, TypeError)):
        trace.set_total_duration(bad_duration)


def test_re_recording_stage_replaces_previous_value():
    """Re-recording a stage should overwrite prior data for that same stage."""
    trace = RequestTrace("req-replace")

    trace.record_stage_duration("queue_wait", 0.05)
    trace.record_stage_duration("queue_wait", 0.20)

    assert trace.get_stage_duration("queue_wait") == pytest.approx(0.20)
    assert len(trace.stage_durations) == 1


def test_total_duration_is_separate_from_stage_durations():
    """Total request duration is stored separately and not inferred from stage values."""
    trace = RequestTrace("req-total", total_duration_s=2.75)
    trace.record_stage_duration("tokenization", 0.10)
    trace.record_stage_duration("prefill", 0.90)
    trace.record_stage_duration("decode", 1.50)

    assert trace.total_duration_s == pytest.approx(2.75)
    assert trace.total_duration_s != sum(trace.stage_durations.values())


def test_total_duration_rejection_and_update():
    """The total duration must be a valid non-negative finite float in seconds."""
    trace = RequestTrace("req-total-update")

    with pytest.raises(ValueError):
        trace.set_total_duration(-1.0)

    updated = trace.set_total_duration(3.5)
    assert updated == pytest.approx(3.5)
    assert trace.total_duration_s == pytest.approx(3.5)


def test_copy_preserves_request_id_and_stage_values():
    """A copy should preserve the trace payload without aliasing stage dictionaries."""
    trace = RequestTrace("req-copy", total_duration_s=1.0)
    trace.record_stage_duration("decode", 0.4)

    cloned = trace.copy()
    assert cloned.request_id == trace.request_id
    assert cloned.total_duration_s == trace.total_duration_s
    assert cloned.stage_durations == trace.stage_durations
    assert cloned.stage_durations is not trace.stage_durations


def test_stage_names_are_the_canonical_m2_names_in_order():
    """The trace model should only use the existing canonical M2.2 stage definitions."""
    assert STAGE_NAMES == [
        "tokenization",
        "queue_wait",
        "prefill",
        "decode",
        "detokenization",
        "network",
        "total",
    ]
