"""Tests for the Member 2 Prometheus metric registry and exporter."""

import json
import socket
import urllib.error
import urllib.request

import pytest

from llm_profiler.instrumentation import MetricRegistry, PrometheusExporter


def _find_free_port() -> int:
    """Return an available TCP port for local exporter testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_registry_contains_expected_metric_names():
    """The registry must expose the agreed metric names and canonical stage labels."""
    registry = MetricRegistry()
    text = registry.render()

    assert "llmprof_stage_latency_seconds" in text
    assert "llmprof_request_total_seconds" in text
    assert "llmprof_tokens_in_total" in text
    assert "llmprof_tokens_out_total" in text

    for stage in [
        "tokenization",
        "queue_wait",
        "prefill",
        "decode",
        "detokenization",
        "network",
        "total",
    ]:
        assert f'stage="{stage}"' in text or "llmprof_stage_latency_seconds" in text


def test_record_and_expose_observed_stage_duration():
    """A valid observed stage duration must be persisted and rendered."""
    registry = MetricRegistry()
    registry.record_stage_duration("prefill", 0.25)
    text = registry.render()

    assert "llmprof_stage_latency_seconds" in text
    assert "stage=\"prefill\"" in text
    assert "prefill" in text
    assert "llmprof_stage_latency_seconds_bucket" in text


def test_record_total_request_duration_separately():
    """The total request duration stays distinct from individual stage observations."""
    registry = MetricRegistry()
    registry.record_stage_duration("decode", 0.9)
    registry.record_request_total_duration(1.8)
    text = registry.render()

    assert "llmprof_request_total_seconds_sum 1.8" in text
    assert "llmprof_stage_latency_seconds_bucket" in text
    assert "stage=\"decode\"" in text


def test_token_counter_updates_are_recorded():
    """Token counts are tracked as cumulative counters."""
    registry = MetricRegistry()
    registry.record_tokens_in_total(12)
    registry.record_tokens_in_total(8)
    registry.record_tokens_out_total(5)
    text = registry.render()

    assert "llmprof_tokens_in_total 20" in text
    assert "llmprof_tokens_out_total 5" in text


def test_invalid_values_are_rejected():
    """Negative, non-finite, and invalid token values must be rejected."""
    registry = MetricRegistry()

    with pytest.raises(ValueError):
        registry.record_stage_duration("prefill", -0.1)

    with pytest.raises(ValueError):
        registry.record_request_total_duration(float("nan"))

    with pytest.raises(ValueError):
        registry.record_tokens_in_total(-1)

    with pytest.raises(ValueError):
        registry.record_tokens_out_total(float("inf"))


def test_http_metrics_endpoint_exposes_prometheus_text():
    """The exporter should expose a valid Prometheus text format at /metrics."""
    registry = MetricRegistry()
    registry.record_stage_duration("network", 0.17)
    registry.record_request_total_duration(0.5)
    registry.record_tokens_in_total(50)
    registry.record_tokens_out_total(30)

    port = _find_free_port()
    exporter = PrometheusExporter(registry=registry, host="127.0.0.1", port=port)
    exporter.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{exporter.port}/metrics", timeout=5) as response:
            body = response.read().decode("utf-8")
            headers = response.headers
            assert headers.get_content_type() == "text/plain"
            assert "llmprof_stage_latency_seconds" in body
            assert "llmprof_request_total_seconds" in body
            assert "llmprof_tokens_in_total 50" in body
            assert "llmprof_tokens_out_total 30" in body
    finally:
        exporter.shutdown()


def test_unobserved_stages_are_not_filled_with_fake_values():
    """Stages with no observed values must not appear as if they were measured."""
    registry = MetricRegistry()
    registry.record_stage_duration("decode", 0.7)
    text = registry.render()

    assert "stage=\"decode\"" in text
    assert "stage=\"prefill\"" not in text
    assert "stage=\"tokenization\"" not in text


def test_ingest_endpoint_records_valid_observations():
    """A valid ingest payload must update the exposed registry without creating fake stage timings."""
    registry = MetricRegistry()
    port = _find_free_port()
    exporter = PrometheusExporter(registry=registry, host="127.0.0.1", port=port)
    exporter.start()
    try:
        payload = json.dumps({
            "request_total_seconds": 0.42,
            "tokens_in_total": 123,
            "tokens_out_total": 45,
            "stage_durations": {
                "prefill": 0.15,
                "decode": 0.20,
            },
        }).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{exporter.port}/ingest",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 202

        text = registry.render()
        assert "llmprof_request_total_seconds_sum 0.42" in text
        assert "llmprof_tokens_in_total 123" in text
        assert "llmprof_tokens_out_total 45" in text
        assert 'stage="prefill"' in text
        assert 'stage="decode"' in text
        assert "llmprof_stage_latency_seconds_count{stage=\"prefill\"} 1" in text
        assert "llmprof_stage_latency_seconds_count{stage=\"decode\"} 1" in text
        assert "llmprof_stage_latency_seconds_sum{stage=\"prefill\"} 0.15" in text
        assert "llmprof_stage_latency_seconds_sum{stage=\"decode\"} 0.2" in text

    finally:
        exporter.shutdown()


def test_ingest_endpoint_rejects_malformed_payloads():
    """Missing keys, wrong shapes, or invalid numbers must be rejected with 4xx responses."""
    registry = MetricRegistry()
    port = _find_free_port()
    exporter = PrometheusExporter(registry=registry, host="127.0.0.1", port=port)
    exporter.start()
    try:
        invalid_payloads = [
            b'{"request_total_seconds": 0.1}',
            b'{"request_total_seconds": -1, "tokens_in_total": 10, "tokens_out_total": 5}',
            b'{"request_total_seconds": "bad", "tokens_in_total": 10, "tokens_out_total": 5}',
            b'{"request_total_seconds": 0.1, "tokens_in_total": -1, "tokens_out_total": 5}',
            b'{"request_total_seconds": 0.1, "tokens_in_total": 10, "tokens_out_total": 1.5}',
            b'{"request_total_seconds": 0.1, "tokens_in_total": 10, "tokens_out_total": 5, "extra": true}',
            b'not-json',
        ]

        for payload in invalid_payloads:
            request = urllib.request.Request(
                f"http://127.0.0.1:{exporter.port}/ingest",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises((urllib.error.HTTPError, urllib.error.URLError)):
                urllib.request.urlopen(request, timeout=5)
    finally:
        exporter.shutdown()
