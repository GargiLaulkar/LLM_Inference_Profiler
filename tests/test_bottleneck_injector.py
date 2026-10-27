"""Unit tests for Member 4 Bottleneck Injector."""

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
from llm_profiler.analyzer.bottleneck_injector import BottleneckInjector
from llm_profiler.experiments.check_contracts import validate_csv, EXPECTED_HEADERS


def test_synthetic_workload_matches_csv_contract():
    """Verify generated synthetic CSV complies strictly with schema contract."""
    injector = BottleneckInjector(seed=42)
    csv_rows, traces = injector.generate_synthetic_workload(
        num_requests=25,
        scenario_tag="contract_test",
        target_rps=5.0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path, _ = injector.save_workload(csv_rows, traces, output_dir=tmp_dir)
        # Run M1 contract validator
        assert validate_csv(csv_path) is True


def test_synthetic_workload_trace_consistency():
    """Verify stage durations in synthetic traces add up to total request latency."""
    injector = BottleneckInjector(seed=42)
    csv_rows, traces = injector.generate_synthetic_workload(
        num_requests=20,
        scenario_tag="consistency_test",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path, traces_path = injector.save_workload(csv_rows, traces, output_dir=tmp_dir)
        analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
        result = analyzer.analyze(csv_path, traces_path)

        assert result["explained_share"] == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("target_stage", ["prefill", "decode", "queue_wait", "network", "tokenization"])
def test_injected_bottleneck_ranked_number_one(target_stage):
    """Verify that injecting a large delay into target_stage causes Analyzer to rank it #1."""
    injector = BottleneckInjector(seed=123)
    # Baseline total latency is ~0.11s. Injecting 0.8s will make target_stage >70% of total latency.
    csv_rows, traces = injector.generate_synthetic_workload(
        num_requests=30,
        scenario_tag=f"inject_{target_stage}",
        injected_stage=target_stage,
        injection_delay_s=0.8,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path, traces_path = injector.save_workload(csv_rows, traces, output_dir=tmp_dir)
        analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
        result = analyzer.analyze(csv_path, traces_path)

        top_bottleneck = result["top_bottlenecks"][0]
        assert top_bottleneck["rank"] == 1
        assert top_bottleneck["stage"] == target_stage
        assert top_bottleneck["share_of_latency"] > 0.50
        assert target_stage in top_bottleneck["evidence"]
        assert result["explained_share"] == pytest.approx(1.0, rel=1e-3)


def test_inject_into_existing_dataset():
    """Verify injecting delay into an existing dataset updates both traces and CSV."""
    injector = BottleneckInjector(seed=99)
    csv_rows, traces = injector.generate_synthetic_workload(
        num_requests=10,
        scenario_tag="base_run",
    )

    base_lat = sum(float(r["total_latency_s"]) for r in csv_rows)

    # Inject 0.5s into prefill on 100% of requests (10 * 0.5s = 5.0s added)
    mod_csv, mod_traces = injector.inject_into_existing_dataset(
        csv_rows=csv_rows,
        trace_records=traces,
        target_stage="prefill",
        delay_seconds=0.5,
        fraction_of_requests=1.0,
    )

    new_lat = sum(float(r["total_latency_s"]) for r in mod_csv)
    assert new_lat == pytest.approx(base_lat + 5.0, rel=1e-4)

    # Verify stage sums match new total latency
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path, traces_path = injector.save_workload(mod_csv, mod_traces, output_dir=tmp_dir)
        analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
        result = analyzer.analyze(csv_path, traces_path)

        assert result["explained_share"] == pytest.approx(1.0, rel=1e-3)
        assert result["top_bottlenecks"][0]["stage"] == "prefill"
