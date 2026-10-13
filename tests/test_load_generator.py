"""Unit and Integration tests for Load Generator and Shared Schema Contracts."""

import os
import subprocess
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm_profiler.experiments.load_generator import (
    CSV_HEADERS,
    calculate_percentiles,
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
