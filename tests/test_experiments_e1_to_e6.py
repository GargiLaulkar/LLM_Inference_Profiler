"""Integration tests for Experiment Suite E1 through E6 (Handbook IDs)."""

import os
import sys
import tempfile
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.experiments.run_experiments_e1_to_e6 import ExperimentSuiteRunner


@pytest.fixture
def suite_runner():
    tmp_dir = tempfile.TemporaryDirectory()
    runner = ExperimentSuiteRunner(output_dir=tmp_dir.name, gpu_price_per_hour_usd=1.0)
    yield runner
    tmp_dir.cleanup()


def test_e1_injected_bottleneck_detection(suite_runner):
    """Verify E1 (Injected-bottleneck detection) passes synthetic sanity check."""
    e1 = suite_runner.run_e1_injected_bottleneck_detection(requests_per_scenario=15)
    assert e1["experiment_id"] == "E1"
    assert e1["data_source"] == "synthetic"
    assert e1["all_passed"] is True
    assert "synthetic sanity check" in e1["validation_note"]


def test_e2_load_level_rps(suite_runner):
    """Verify E2 (Load level scaling) executes across 1, 5, 10 RPS."""
    e2 = suite_runner.run_e2_load_level_rps(requests_per_scenario=10)
    assert e2["experiment_id"] == "E2"
    assert e2["data_source"] == "synthetic"
    assert "rps_1" in e2["scenarios"]
    assert "rps_5" in e2["scenarios"]
    assert "rps_10" in e2["scenarios"]


def test_e3_prompt_output_length_mix(suite_runner):
    """Verify E3 (Prompt/output length mix) tests both prompt and output shifts."""
    e3 = suite_runner.run_e3_prompt_output_length_mix(requests_per_scenario=10)
    assert e3["experiment_id"] == "E3"
    assert e3["data_source"] == "synthetic"
    scenarios = e3["scenarios"]
    assert "short_prompt_32_64" in scenarios
    assert "long_prompt_1024_64" in scenarios
    assert "long_output_128_256" in scenarios


def test_e4_batching_marked_pending_live(suite_runner):
    """Verify E4 (Batching on/off) is marked pending_live with synthetic tag."""
    e4 = suite_runner.run_e4_batching_on_off()
    assert e4["experiment_id"] == "E4"
    assert e4["status"] == "pending_live"
    assert e4["data_source"] == "synthetic"


def test_e5_model_size_marked_pending_live(suite_runner):
    """Verify E5 (Model size) is marked pending_live with synthetic tag."""
    e5 = suite_runner.run_e5_model_size()
    assert e5["experiment_id"] == "E5"
    assert e5["status"] == "pending_live"
    assert e5["data_source"] == "synthetic"


def test_e6_profiler_overhead(suite_runner):
    """Verify E6 (Profiler overhead on vs off) runs instrumentation microbenchmark."""
    e6 = suite_runner.run_e6_profiler_overhead_on_vs_off(iterations=1_000)
    assert e6["experiment_id"] == "E6"
    assert e6["data_source"] == "synthetic"
    assert "overhead_metrics" in e6
    assert e6["overhead_metrics"]["is_negligible"] is True


def test_run_all_experiments_e1_to_e6(suite_runner):
    """Verify run_all completes, contains all 6 handbook IDs, and saves summary."""
    summary = suite_runner.run_all()
    assert summary["data_source"] == "synthetic"
    exps = summary["experiments"]
    for i in range(1, 7):
        assert f"e{i}" in exps
        assert exps[f"e{i}"]["experiment_id"] == f"E{i}"
        assert exps[f"e{i}"]["data_source"] == "synthetic"

    assert exps["e4"]["status"] == "pending_live"
    assert exps["e5"]["status"] == "pending_live"

    summary_file = os.path.join(suite_runner.output_dir, "experiments_e1_to_e6_summary.json")
    assert os.path.exists(summary_file)
