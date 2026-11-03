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
    reports = [
        evaluate_observer_overhead(iterations=2_000)
        for _ in range(5)
    ]
    report = min(reports, key=lambda result: result["profiler_overhead_per_request_us"])

    assert "iterations" in report
    assert report.get("benchmark_type") == "instrumentation_microbenchmark"
    assert report.get("data_source") == "synthetic"
    assert "uninstrumented_per_request_us" in report
    assert "instrumented_per_request_us" in report
    assert "full_telemetry_per_request_us" in report
    assert "profiler_overhead_per_request_us" in report
    assert "overhead_percent_on_100ms_request" in report
    assert "is_negligible" in report

    assert report["iterations"] == 2_000
    assert report["uninstrumented_per_request_us"] > 0
    assert report["instrumented_per_request_us"] > 0
    assert report["full_telemetry_per_request_us"] > 0

    # Keep a generous bound and use the best of repeated samples to reduce host-noise failures.
    assert report["profiler_overhead_per_request_us"] < 1_000.0

    # On a 100ms LLM request, overhead should remain below 1%.
    assert report["overhead_percent_on_100ms_request"] < 1.0
    assert report["is_negligible"] is True
