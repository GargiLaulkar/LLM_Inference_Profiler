"""Unit tests for Member 4 Inference Cost Model (Task 1)."""

import csv
import json
import os
import sys
import tempfile
import urllib.request
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer.cost_model import (
    CostModel,
    StandaloneCostExporter,
    calculate_csv_cost,
    calculate_request_cost,
    compute_interval_union_seconds,
    load_gpu_price_from_config,
)


def test_calculate_request_cost_single_formula():
    """Verify single request formula: ((gpu_price * (latency / 3600)) / tokens_produced) * 1e6."""
    # 3.6s at $1.00/hr = 0.001 hours = $0.001
    # 100 output tokens -> $0.001 / 100 = $0.00001 per token -> $10.00 per 1M tokens
    cost = calculate_request_cost(
        total_latency_s=3.6,
        output_tokens=100,
        gpu_price_per_hour_usd=1.0,
    )
    assert cost == pytest.approx(10.0, rel=1e-6)


def test_overlapping_requests_not_double_counted():
    """Test 1: Overlapping concurrent requests are NOT double counted in GPU hours."""
    # Two requests overlapping between t=10 and t=20 (wall clock window = 10s)
    # Req 1: t=10 to t=20 (10s duration), 100 output tokens
    # Req 2: t=12 to t=20 (8s duration), 100 output tokens
    records = [
        {"send_ts": 10.0, "end_ts": 20.0, "total_latency_s": 10.0, "output_tokens": 100, "prompt_tokens": 50, "status": "ok"},
        {"send_ts": 12.0, "end_ts": 20.0, "total_latency_s": 8.0, "output_tokens": 100, "prompt_tokens": 50, "status": "ok"},
    ]
    model = CostModel(gpu_price_per_hour_usd=1.0)
    summary = model.evaluate_dataset(records)

    # Window = 20 - 10 = 10s
    # Interval union = [10, 20] = 10s
    # Active seconds = 10s (NOT 10 + 8 = 18s!)
    assert summary["wall_clock_window_seconds"] == pytest.approx(10.0)
    assert summary["active_seconds"] == pytest.approx(10.0)
    assert summary["time_share"] == pytest.approx(1.0)
    assert summary["gpu_hours_used"] == pytest.approx(10.0 / 3600.0)
    assert summary["tokens_produced"] == 200

    # Cost = (1.0 * (10 / 3600)) / 200 * 1e6 = 13.888889 USD
    expected_cost = (1.0 * (10.0 / 3600.0) / 200) * 1_000_000.0
    assert summary["cost_per_million_tokens_usd"] == pytest.approx(expected_cost, rel=1e-5)


def test_sequential_requests_with_gaps_and_flag():
    """Test 2: Sequential requests with gaps compute active time_share, with flag for 1.0."""
    # Req 1: t=0 to t=10 (10s), 100 output tokens
    # Gap: t=10 to t=20 (10s idle)
    # Req 2: t=20 to t=30 (10s), 100 output tokens
    records = [
        {"send_ts": 0.0, "end_ts": 10.0, "total_latency_s": 10.0, "output_tokens": 100, "prompt_tokens": 50, "status": "ok"},
        {"send_ts": 20.0, "end_ts": 30.0, "total_latency_s": 10.0, "output_tokens": 100, "prompt_tokens": 50, "status": "ok"},
    ]
    model = CostModel(gpu_price_per_hour_usd=1.0)

    # A) Dynamic time_share (active hours only)
    summary_active = model.evaluate_dataset(records, force_full_time_share=False)
    assert summary_active["wall_clock_window_seconds"] == pytest.approx(30.0)
    assert summary_active["active_seconds"] == pytest.approx(20.0)
    assert summary_active["time_share"] == pytest.approx(20.0 / 30.0)
    assert summary_active["gpu_hours_used"] == pytest.approx(20.0 / 3600.0)
    expected_active_cost = (1.0 * (20.0 / 3600.0) / 200) * 1_000_000.0
    assert summary_active["cost_per_million_tokens_usd"] == pytest.approx(expected_active_cost, rel=1e-5)

    # B) Dedicated GPU reservation flag (time_share=1.0)
    summary_full = model.evaluate_dataset(records, force_full_time_share=True)
    assert summary_full["time_share"] == pytest.approx(1.0)
    assert summary_full["gpu_hours_used"] == pytest.approx(30.0 / 3600.0)
    expected_full_cost = (1.0 * (30.0 / 3600.0) / 200) * 1_000_000.0
    assert summary_full["cost_per_million_tokens_usd"] == pytest.approx(expected_full_cost, rel=1e-5)


def test_zero_tokens_returns_zero():
    """Test 3: Zero tokens handled gracefully without division by zero."""
    records = [
        {"send_ts": 0.0, "end_ts": 5.0, "total_latency_s": 5.0, "output_tokens": 0, "prompt_tokens": 0, "status": "ok"},
    ]
    model = CostModel(gpu_price_per_hour_usd=1.0)
    summary = model.evaluate_dataset(records)
    assert summary["tokens_produced"] == 0
    assert summary["cost_per_million_tokens_usd"] == 0.0
    assert summary["cost_per_million_total_tokens_usd"] == 0.0


def test_failed_requests_excluded():
    """Test 4: Failed requests are completely excluded from cost and window calculations."""
    records = [
        {"send_ts": 0.0, "end_ts": 10.0, "total_latency_s": 10.0, "output_tokens": 100, "prompt_tokens": 50, "status": "ok"},
        # Failed request with huge delay should NOT inflate the window or token count
        {"send_ts": 10.0, "end_ts": 100.0, "total_latency_s": 90.0, "output_tokens": 500, "prompt_tokens": 50, "status": "error: 504_gateway_timeout"},
    ]
    model = CostModel(gpu_price_per_hour_usd=1.0)
    summary = model.evaluate_dataset(records)

    assert summary["request_count"] == 1
    assert summary["tokens_produced"] == 100
    assert summary["wall_clock_window_seconds"] == pytest.approx(10.0)
    assert summary["active_seconds"] == pytest.approx(10.0)
    expected_cost = (1.0 * (10.0 / 3600.0) / 100) * 1_000_000.0
    assert summary["cost_per_million_tokens_usd"] == pytest.approx(expected_cost, rel=1e-5)


def test_cost_per_million_total_tokens_optional_value():
    """Verify cost per 1M total tokens is available as a distinct metric."""
    records = [
        {"send_ts": 0.0, "end_ts": 3.6, "total_latency_s": 3.6, "output_tokens": 100, "prompt_tokens": 100, "status": "ok"},
    ]
    model = CostModel(gpu_price_per_hour_usd=1.0)
    summary = model.evaluate_dataset(records)

    # 3.6s at $1/hr = $0.001
    # output_tokens = 100 -> cost / 1M output tokens = $10.0
    # total_tokens = 200 -> cost / 1M total tokens = $5.0
    assert summary["cost_per_million_tokens_usd"] == pytest.approx(10.0, rel=1e-5)
    assert summary["cost_per_million_total_tokens_usd"] == pytest.approx(5.0, rel=1e-5)


def test_interval_union_empty_and_disjoint():
    """Verify compute_interval_union_seconds edge cases."""
    assert compute_interval_union_seconds([]) == 0.0
    # Disjoint intervals: [0, 5] and [10, 15] -> 10s
    assert compute_interval_union_seconds([(0, 5), (10, 15)]) == pytest.approx(10.0)
    # Contiguous intervals: [0, 5] and [5, 10] -> 10s
    assert compute_interval_union_seconds([(0, 5), (5, 10)]) == pytest.approx(10.0)
    # Nested intervals: [0, 10] and [2, 6] -> 10s
    assert compute_interval_union_seconds([(0, 10), (2, 6)]) == pytest.approx(10.0)


def test_load_gpu_price_from_real_config():
    """Verify loading from actual config.yaml in repo."""
    price = load_gpu_price_from_config("config.yaml")
    assert price == 1.0


def test_prometheus_exposition_format():
    """Verify Prometheus text format complies strictly with contract."""
    model = CostModel(gpu_price_per_hour_usd=1.0)
    rendered = model.render_prometheus_metrics(12.3456)

    assert "# HELP llmprof_cost_per_million_tokens_usd Amortized inference dollar cost per 1M tokens." in rendered
    assert "# TYPE llmprof_cost_per_million_tokens_usd gauge" in rendered
    assert "llmprof_cost_per_million_tokens_usd 12.345600" in rendered


def test_standalone_cost_exporter_server():
    """Verify StandaloneCostExporter serves /metrics correctly over HTTP."""
    model = CostModel(gpu_price_per_hour_usd=1.0)

    with StandaloneCostExporter(model, host="127.0.0.1", port=0) as exporter:
        port = exporter.port
        url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert "llmprof_cost_per_million_tokens_usd" in body
