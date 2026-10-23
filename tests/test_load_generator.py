"""Unit and Integration tests for Load Generator and Shared Schema Contracts."""

import os
import subprocess
import sys
import pytest
import json
from unittest.mock import patch
import asyncio

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm_profiler.experiments.load_generator import (
    CSV_HEADERS,
    calculate_percentiles,
    emit_request_metrics_to_exporter,
    load_config,
    PromptCalibrator,
)

# Contract required column order
EXPECTED_COLUMNS = [
    "request_id",
    "send_ts",
    "end_ts",
    "total_latency_s",
    "prompt_tokens",
    "output_tokens",
    "status",
    "target_rps",
    "scenario_tag",
]


def test_csv_column_order():
    """Verify CSV header column names and order match the shared schema contract exactly."""
    assert CSV_HEADERS == EXPECTED_COLUMNS, (
        f"CSV headers {CSV_HEADERS} do not match contract {EXPECTED_COLUMNS}"
    )


def test_percentile_calculation_known_list():
    """Verify nearest-rank percentile calculations on a deterministic sample dataset."""
    latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pcts = calculate_percentiles(latencies)

    assert pytest.approx(pcts["mean"], rel=1e-4) == 0.55
    # Nearest rank:
    # 50th percentile (rank 5) -> 0.6
    assert pcts["p50"] == 0.6
    # 95th percentile (rank 9) -> 1.0
    assert pcts["p95"] == 1.0
    # 99th percentile (rank 9) -> 1.0
    assert pcts["p99"] == 1.0

    # Empty list edge case
    empty_pcts = calculate_percentiles([])
    assert empty_pcts["mean"] == 0.0
    assert empty_pcts["p50"] == 0.0


def test_config_loading():
    """Verify loading config.yaml and that all mandatory contract keys exist."""
    config_path = os.path.join(PROJECT_ROOT, "llm_profiler", "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")

    cfg = load_config(config_path)
    assert isinstance(cfg, dict), "Configuration must parse to a dictionary"

    mandatory_keys = [
        "model_name",
        "server_host",
        "server_port",
        "server_url",
        "gpu_price_per_hour_usd",
        "prometheus_scrape_interval_s",
        "results_dir",
        "scenarios",
    ]
    for key in mandatory_keys:
        assert key in cfg, f"Missing required configuration key: {key}"

    assert cfg["gpu_price_per_hour_usd"] == 1.0
    assert cfg["prometheus_scrape_interval_s"] == 5
    assert len(cfg["scenarios"]) >= 3


def test_dry_run_calibration():
    """Verify that PromptCalibrator generates expected token counts in dry-run mode."""
    calibrator = PromptCalibrator("Qwen/Qwen2.5-0.5B-Instruct")
    target_tokens = 128
    prompt = calibrator.build_prompt(target_tokens)
    token_count = calibrator.count_tokens(prompt)

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert abs(token_count - target_tokens) < 25


def test_dry_run_cli():
    """Verify that --dry-run CLI command executes with exit code 0 and outputs expected text."""
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "experiments", "load_generator.py"), "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert res.returncode == 0
    assert "LOAD GENERATOR DRY RUN" in res.stdout
    assert "Calibrated Tokens" in res.stdout


def test_emit_request_metrics_to_exporter_uses_actual_observations():
    """The load generator must forward the real measured latency and token counts, not fabricated data."""
    calls = []

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_urlopen(request, timeout=1.0):
        calls.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    with patch("llm_profiler.experiments.load_generator.urllib.request.urlopen", side_effect=fake_urlopen):
        ok = emit_request_metrics_to_exporter(
            request_total_seconds=0.75,
            tokens_in_total=111,
            tokens_out_total=42,
            exporter_url="http://127.0.0.1:9101",
            timeout=1.0,
        )

    assert ok is True
    assert calls == [{"request_total_seconds": 0.75, "tokens_in_total": 111, "tokens_out_total": 42}]


def test_emit_request_metrics_to_exporter_is_best_effort_when_unavailable():
    """Exporter errors must not fail a successful load-generator request or CSV write path."""
    def raise_url_error(*args, **kwargs):
        raise OSError("exporter unavailable")

    with patch("llm_profiler.experiments.load_generator.urllib.request.urlopen", side_effect=raise_url_error):
        ok = emit_request_metrics_to_exporter(
            request_total_seconds=1.25,
            tokens_in_total=99,
            tokens_out_total=17,
            exporter_url="http://127.0.0.1:9101",
            timeout=1.0,
        )

    assert ok is False


def test_send_single_request_only_counts_successful_usage_tokens():
    """Successful 200 responses should contribute actual prompt/completion tokens to the exporter."""

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def json(self):
            return {"usage": {"prompt_tokens": 123, "completion_tokens": 45}}

        async def text(self):
            return ""

    class FakeSession:
        def post(self, url, json, headers):
            return FakeResponse()

    async def _run():
        calls = []

        def fake_emit(**kwargs):
            calls.append(kwargs)
            return True

        with patch("llm_profiler.experiments.load_generator.emit_request_metrics_to_exporter", side_effect=fake_emit) as mock_emit:
            result = await __import__("llm_profiler.experiments.load_generator", fromlist=["send_single_request"]).send_single_request(
                session=FakeSession(),
                url="http://example.test/v1/completions",
                payload={"max_tokens": 12},
                request_id="req-1",
                target_rps=1.0,
                scenario_tag="bench",
                estimated_prompt_tokens=777,
            )

        assert result["status"] == "ok"
        assert result["prompt_tokens"] == 123
        assert result["output_tokens"] == 45
        assert mock_emit.call_count == 1
        assert calls[0]["tokens_in_total"] == 123
        assert calls[0]["tokens_out_total"] == 45
        assert calls[0]["request_total_seconds"] >= 0.0

    asyncio.run(_run())


def test_send_single_request_does_not_emit_tokens_for_failed_responses():
    """HTTP failures and timeouts must still record total latency but leave token counters unchanged."""

    class FakeResponse:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def text(self):
            return "server error"

    class FakeSession:
        def post(self, url, json, headers):
            return FakeResponse()

    async def _run():
        with patch("llm_profiler.experiments.load_generator.emit_request_metrics_to_exporter") as mock_emit:
            result = await __import__("llm_profiler.experiments.load_generator", fromlist=["send_single_request"]).send_single_request(
                session=FakeSession(),
                url="http://example.test/v1/completions",
                payload={"max_tokens": 12},
                request_id="req-2",
                target_rps=2.0,
                scenario_tag="bench",
                estimated_prompt_tokens=777,
            )

        assert result["status"] == "error:http_500"
        assert result["prompt_tokens"] == 777
        assert result["output_tokens"] == 12
        assert mock_emit.call_count == 1
        assert mock_emit.call_args.kwargs["tokens_in_total"] == 0
        assert mock_emit.call_args.kwargs["tokens_out_total"] == 0
        assert mock_emit.call_args.kwargs["request_total_seconds"] >= 0.0

    asyncio.run(_run())


def test_send_single_request_ignores_exporter_unavailability():
    """Exporter failures must not fail the load test request or CSV write path."""

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def json(self):
            return {"usage": {"prompt_tokens": 88, "completion_tokens": 17}}

        async def text(self):
            return ""

    class FakeSession:
        def post(self, url, json, headers):
            return FakeResponse()

    async def _run():
        with patch("llm_profiler.experiments.load_generator.emit_request_metrics_to_exporter", return_value=False):
            result = await __import__("llm_profiler.experiments.load_generator", fromlist=["send_single_request"]).send_single_request(
                session=FakeSession(),
                url="http://example.test/v1/completions",
                payload={"max_tokens": 17},
                request_id="req-3",
                target_rps=3.0,
                scenario_tag="bench",
                estimated_prompt_tokens=100,
            )

        assert result["status"] == "ok"
        assert result["prompt_tokens"] == 88
        assert result["output_tokens"] == 17

    asyncio.run(_run())
