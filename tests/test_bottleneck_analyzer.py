"""Unit and integration tests for Member 4 Bottleneck Analyzer."""

import csv
import json
import os
import sys
import tempfile
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer.bottleneck_analyzer import BottleneckAnalyzer


@pytest.fixture
def sample_data():
    """Create a temporary CSV and JSONL pair for deterministic testing."""
    tmp_dir = tempfile.TemporaryDirectory()
    csv_path = os.path.join(tmp_dir.name, "requests.csv")
    traces_path = os.path.join(tmp_dir.name, "traces.jsonl")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "request_id", "send_ts", "end_ts", "total_latency_s",
            "prompt_tokens", "output_tokens", "status", "target_rps", "scenario_tag"
        ])
        # Req 1: Total = 1.0s, prompt=100, output=100
        writer.writerow(["req-1", "100.0", "101.0", "1.0", "100", "100", "ok", "1.0", "test"])
        # Req 2: Total = 2.0s, prompt=100, output=100
        writer.writerow(["req-2", "101.0", "103.0", "2.0", "100", "100", "ok", "1.0", "test"])

    # Write Traces
    # Req 1: prefill=0.6s (60%), decode=0.3s (30%), queue_wait=0.08s (8%), detokenization=0.02s (2%) -> sum=1.0s
    # Req 2: prefill=1.4s (70%), decode=0.4s (20%), queue_wait=0.15s (7.5%), detokenization=0.05s (2.5%) -> sum=2.0s
    traces = [
        {"request_id": "req-1", "stage": "prefill", "start_ts": 100.1, "end_ts": 100.7, "duration_s": 0.6},
        {"request_id": "req-1", "stage": "decode", "start_ts": 100.7, "end_ts": 101.0, "duration_s": 0.3},
        {"request_id": "req-1", "stage": "queue_wait", "start_ts": 100.0, "end_ts": 100.08, "duration_s": 0.08},
        {"request_id": "req-1", "stage": "detokenization", "start_ts": 101.0, "end_ts": 101.02, "duration_s": 0.02},
        {"request_id": "req-2", "stage": "prefill", "start_ts": 101.1, "end_ts": 102.5, "duration_s": 1.4},
        {"request_id": "req-2", "stage": "decode", "start_ts": 102.5, "end_ts": 102.9, "duration_s": 0.4},
        {"request_id": "req-2", "stage": "queue_wait", "start_ts": 101.0, "end_ts": 101.15, "duration_s": 0.15},
        {"request_id": "req-2", "stage": "detokenization", "start_ts": 102.9, "end_ts": 102.95, "duration_s": 0.05},
    ]
    with open(traces_path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    yield csv_path, traces_path

    tmp_dir.cleanup()


def test_analyzer_contract_schema(sample_data):
    """Verify analyzer output JSON strictly complies with shared contract schema."""
    csv_path, traces_path = sample_data
    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
    result = analyzer.analyze(csv_path, traces_path)

    # Top level keys
    assert "top_bottlenecks" in result
    assert "cost_per_million_tokens_usd" in result
    assert "explained_share" in result

    assert isinstance(result["top_bottlenecks"], list)
    assert isinstance(result["cost_per_million_tokens_usd"], float)
    assert isinstance(result["explained_share"], float)

    # Check top_bottlenecks structure
    for b in result["top_bottlenecks"]:
        assert "rank" in b
        assert "stage" in b
        assert "share_of_latency" in b
        assert "evidence" in b
        assert "suggested_action" in b
        assert isinstance(b["rank"], int)
        assert isinstance(b["stage"], str)
        assert isinstance(b["share_of_latency"], float)
        assert isinstance(b["evidence"], str)
        assert isinstance(b["suggested_action"], str)


def test_analyzer_accurate_ranking(sample_data):
    """Verify top 3 bottlenecks are ranked in descending order of duration share."""
    csv_path, traces_path = sample_data
    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
    result = analyzer.analyze(csv_path, traces_path, max_bottlenecks=3)

    bottlenecks = result["top_bottlenecks"]
    assert len(bottlenecks) == 3

    # Total latency = 3.0s
    # prefill: 2.0s -> 66.67%
    # decode: 0.7s -> 23.33%
    # queue_wait: 0.23s -> 7.67%
    assert bottlenecks[0]["rank"] == 1
    assert bottlenecks[0]["stage"] == "prefill"
    assert bottlenecks[0]["share_of_latency"] == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert "chunked prefill" in bottlenecks[0]["suggested_action"].lower()

    assert bottlenecks[1]["rank"] == 2
    assert bottlenecks[1]["stage"] == "decode"
    assert bottlenecks[1]["share_of_latency"] == pytest.approx(0.7 / 3.0, rel=1e-3)

    assert bottlenecks[2]["rank"] == 3
    assert bottlenecks[2]["stage"] == "queue_wait"
    assert bottlenecks[2]["share_of_latency"] == pytest.approx(0.23 / 3.0, rel=1e-3)


def test_analyzer_explained_share(sample_data):
    """Verify explained_share computes the exact ratio of stage times to total latency."""
    csv_path, traces_path = sample_data
    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
    result = analyzer.analyze(csv_path, traces_path)

    # Stage total: 2.0 (prefill) + 0.7 (decode) + 0.23 (queue) + 0.07 (detok) = 3.0s
    # Total CSV latency: 3.0s -> explained_share == 1.0
    assert result["explained_share"] == pytest.approx(1.0, rel=1e-3)


def test_analyzer_discrepancy_and_unexplained_latency():
    """Verify that uninstrumented or missing latency is reflected in explained_share < 1.0."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "requests.csv")
        traces_path = os.path.join(tmp_dir, "traces.jsonl")

        # Total latency is 2.0s
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id", "send_ts", "end_ts", "total_latency_s",
                "prompt_tokens", "output_tokens", "status", "target_rps", "scenario_tag"
            ])
            writer.writerow(["req-1", "100.0", "102.0", "2.0", "50", "50", "ok", "1.0", "test"])

        # But stages only sum to 1.0s (50% unexplained latency, e.g. OS/socket delay)
        with open(traces_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "request_id": "req-1", "stage": "prefill", "start_ts": 100.0, "end_ts": 101.0, "duration_s": 1.0
            }) + "\n")

        analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
        result = analyzer.analyze(csv_path, traces_path)

        assert result["explained_share"] == pytest.approx(0.5, rel=1e-3)
        assert result["top_bottlenecks"][0]["stage"] == "prefill"
        assert result["top_bottlenecks"][0]["share_of_latency"] == pytest.approx(0.5, rel=1e-3)


def test_analyzer_graceful_missing_traces(sample_data):
    """Rule 6: Say clearly when data is missing instead of inventing it."""
    csv_path, _ = sample_data
    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
    # Pass non-existent traces path
    result = analyzer.analyze(csv_path, traces_path=None)

    assert result["explained_share"] == 0.0
    assert len(result["top_bottlenecks"]) == 1
    assert result["top_bottlenecks"][0]["stage"] == "missing_stage_traces"
    assert "was not provided or missing" in result["top_bottlenecks"][0]["evidence"]
    # Total cost is still accurately calculated from CSV request logs
    assert result["cost_per_million_tokens_usd"] > 0
