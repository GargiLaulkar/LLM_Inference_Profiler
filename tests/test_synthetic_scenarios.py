"""Tests validating Bottleneck Analyzer against Seeded Ground-Truth Scenarios (Task 4 & 7)."""

import json
import os
import sys
import tempfile
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer.bottleneck_analyzer import BottleneckAnalyzer
from tests.synthetic.make_synthetic_data import generate_all_synthetic_scenarios


@pytest.fixture(scope="module")
def synthetic_manifest():
    """Ensure seeded synthetic datasets exist and return manifest."""
    data_dir = os.path.join(os.path.dirname(__file__), "synthetic", "data")
    manifest = generate_all_synthetic_scenarios(base_dir=data_dir, num_requests=25)
    return manifest


@pytest.mark.parametrize("scenario_name", [
    "normal",
    "slow_tokenizer",
    "no_batching_queue_heavy",
    "cpu_preprocessing",
    "network_heavy",
])
def test_analyzer_matches_ground_truth_top_stage(synthetic_manifest, scenario_name):
    """Assert analyzer's top-1 stage strictly equals ground_truth.json for each scenario."""
    item = synthetic_manifest[scenario_name]
    csv_path = item["csv_path"]
    traces_path = item["traces_path"]
    ground_truth = item["ground_truth"]

    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
    result = analyzer.analyze(csv_path, traces_path)

    top_stage = result["top_bottlenecks"][0]["stage"]
    expected_top = ground_truth["expected_top_stage"]

    assert top_stage == expected_top, (
        f"Scenario '{scenario_name}': expected top-1 stage '{expected_top}', got '{top_stage}'"
    )
    assert result["explained_share"] >= 0.90 and result["explained_share"] <= 1.10
    assert "warning" not in result


def test_broken_sum_triggers_explained_share_warning(synthetic_manifest):
    """Assert broken-sum scenario triggers the explained_share invariant warning."""
    item = synthetic_manifest["broken_sum"]
    csv_path = item["csv_path"]
    traces_path = item["traces_path"]
    ground_truth = item["ground_truth"]

    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
    result = analyzer.analyze(csv_path, traces_path)

    # In broken_sum, total latency was inflated by 2s per request without trace time
    assert result["explained_share"] < 0.90
    assert "warning" in result
    assert "outside nominal bounds" in result["warning"]
    assert "Stage timings do not sum to total request latency" in result["warning"]
    # Evidence in Rank #1 also mentions the warning
    assert "outside nominal bounds" in result["top_bottlenecks"][0]["evidence"]


def test_memory_breakdown_unavailable_when_absent(synthetic_manifest):
    """Task 7: When memory_breakdown.json is absent, evidence says 'memory evidence unavailable'."""
    item = synthetic_manifest["normal"]
    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
    result = analyzer.analyze(item["csv_path"], item["traces_path"], memory_breakdown_path=None)

    for b in result["top_bottlenecks"]:
        assert "memory evidence unavailable" in b["evidence"]


def test_memory_breakdown_parsed_when_valid(synthetic_manifest):
    """Task 7: When valid memory_breakdown.json is provided, components are integrated into evidence."""
    item = synthetic_manifest["normal"]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
        json.dump({
            "phases": {
                "prefill": {
                    "weights": 980 * 1024 * 1024,
                    "kv_cache": 15 * 1024 * 1024,
                    "activations": 45 * 1024 * 1024,
                    "other": 300 * 1024 * 1024,
                },
                "decode": {
                    "weights": 980 * 1024 * 1024,
                    "kv_cache": 35 * 1024 * 1024,
                    "activations": 10 * 1024 * 1024,
                    "other": 310 * 1024 * 1024,
                },
            }
        }, f)
        mem_file = f.name

    try:
        analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
        result = analyzer.analyze(item["csv_path"], item["traces_path"], memory_breakdown_path=mem_file)

        # Evidence must NOT say "memory evidence unavailable"
        top_b = result["top_bottlenecks"][0]
        assert "memory evidence unavailable" not in top_b["evidence"]
        assert "memory:" in top_b["evidence"]
    finally:
        if os.path.exists(mem_file):
            os.remove(mem_file)


def test_memory_breakdown_schema_mismatch_reported(synthetic_manifest):
    """Task 7: When malformed memory_breakdown.json is provided, mismatch is reported clearly."""
    item = synthetic_manifest["normal"]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
        # Missing 'other' component in prefill
        json.dump({
            "phases": {
                "prefill": {"weights": 100, "kv_cache": 10, "activations": 20},
                "decode": {"weights": 100, "kv_cache": 20, "activations": 10, "other": 40},
            }
        }, f)
        bad_mem_file = f.name

    try:
        analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
        result = analyzer.analyze(item["csv_path"], item["traces_path"], memory_breakdown_path=bad_mem_file)
        top_b = result["top_bottlenecks"][0]
        assert "memory_breakdown schema mismatch" in top_b["evidence"]
    finally:
        if os.path.exists(bad_mem_file):
            os.remove(bad_mem_file)
