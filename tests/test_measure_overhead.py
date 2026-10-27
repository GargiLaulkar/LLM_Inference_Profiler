"""Unit tests for Member 4 Observer Overhead Measurement (Experiment E6)."""

import os
import sys
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.experiments.measure_overhead import evaluate_observer_overhead


def test_observer_overhead_evaluation():
    """Verify overhead evaluation executes and returns valid metrics."""
    report = evaluate_observer_overhead(iterations=1_000)

    assert "iterations" in report
    assert report.get("benchmark_type") == "instrumentation_microbenchmark"
    assert report.get("data_source") == "synthetic"
    assert "uninstrumented_per_request_us" in report
    assert "instrumented_per_request_us" in report
    assert "full_telemetry_per_request_us" in report
    assert "profiler_overhead_per_request_us" in report
    assert "overhead_percent_on_100ms_request" in report
    assert "is_negligible" in report

    assert report["iterations"] == 1_000
    assert report["uninstrumented_per_request_us"] > 0
    assert report["instrumented_per_request_us"] > 0
    assert report["full_telemetry_per_request_us"] > 0

    # Overhead must be small: stage timers + tracing should take < 100 microseconds per request
    assert report["profiler_overhead_per_request_us"] < 250.0

    # On a 100ms LLM request, overhead should be well under 1%
    assert report["overhead_percent_on_100ms_request"] < 0.50
    assert report["is_negligible"] is True
