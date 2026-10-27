"""Tests for evidence-based analyzer actions and E1 validation orchestration."""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_profiler.analyzer.actions import build_evidence_notes, build_suggested_action
from llm_profiler.analyzer.bottleneck_analyzer import BottleneckAnalyzer
from llm_profiler.analyzer.bottleneck_injector import BottleneckInjector
from llm_profiler.analyzer.readers import CONTRACT_CSV_COLUMNS
from llm_profiler.experiments.run_injection_validation import (
    INJECTIONS,
    _mocked_subprocess_run,
    run_validation,
)


def _write_synthetic_pair(directory: Path, *, stage: str, delay_s: float):
    rows, traces = BottleneckInjector(seed=13).generate_synthetic_workload(
        num_requests=2,
        scenario_tag="evidence-test",
        injected_stage=stage,
        injection_delay_s=delay_s,
    )
    csv_path = directory / "requests.csv"
    trace_path = directory / "traces.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CONTRACT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    trace_path.write_text(
        "".join(json.dumps(trace) + "\n" for trace in traces),
        encoding="utf-8",
    )
    return csv_path, trace_path


def test_suggested_actions_are_tied_to_measured_stage_share():
    action = build_suggested_action("tokenization", 0.41)
    assert "tokenization share 41.0%" in action
    assert "fast Rust-based tokenizer" in action


def test_queue_wait_evidence_requires_low_gpu_utilization():
    facts = build_evidence_notes(
        "queue_wait",
        0.72,
        gpu_metrics={"gpu_utilization_percent": 23.0},
    )
    assert facts == ["high queue_wait share (72.0%) with low GPU utilization (23.0%)"]
    assert build_evidence_notes(
        "queue_wait",
        0.72,
        gpu_metrics={"gpu_utilization_percent": 86.0},
    ) == []


def test_decode_evidence_uses_power_limit_and_batching_memory():
    memory = {
        "phases": {
            "prefill": {"weights": 70, "kv_cache": 5, "activations": 20, "other": 5},
            "decode": {"weights": 10, "kv_cache": 80, "activations": 5, "other": 5},
        }
    }
    facts = build_evidence_notes(
        "decode",
        0.82,
        gpu_metrics={"gpu_power_watts": 90, "gpu_power_limit_watts": 100},
        memory_data=memory,
        batching_enabled=False,
    )
    assert any("high decode share" in fact for fact in facts)
    assert any("batching is off" in fact and "largest measured" in fact for fact in facts)

    assert build_evidence_notes(
        "decode",
        0.82,
        gpu_metrics={"gpu_power_watts": 90},
    ) == []


def test_analyzer_surfaces_evidence_rules_and_action_text(tmp_path):
    csv_path, trace_path = _write_synthetic_pair(
        tmp_path, stage="decode", delay_s=0.5
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({
        "phases": {
            phase: {"weights": 10, "kv_cache": 80, "activations": 5, "other": 5}
            for phase in ("prefill", "decode")
        }
    }), encoding="utf-8")

    result = BottleneckAnalyzer(gpu_price_per_hour_usd=1).analyze(
        csv_path,
        trace_path,
        memory_breakdown_path=memory_path,
        gpu_metrics={"gpu_power_watts": 90, "gpu_power_limit_watts": 100},
        batching_enabled=False,
    )
    top = result["top_bottlenecks"][0]
    assert top["stage"] == "decode"
    assert "high decode share" in top["evidence"]
    assert "batching is off" in top["evidence"]
    assert "decode share" in top["suggested_action"]


def test_dry_run_is_synthetic_resumable_and_uses_mocked_cli_calls(tmp_path):
    calls = []

    def mocked_runner(command, **kwargs):
        calls.append(command)
        return _mocked_subprocess_run(command, **kwargs)

    hooks = {name: False for name in INJECTIONS}
    out = tmp_path / "results"
    summary = run_validation(
        output_dir=out,
        repeat=3,
        num_requests=3,
        dry_run=True,
        available_hooks=hooks,
        command_runner=mocked_runner,
    )
    assert summary["data_source"] == "synthetic"
    assert summary["status"] == "pending_live"
    assert summary["detection_accuracy"] == 1.0
    assert summary["top3_hit_rate"] == 1.0
    assert summary["false_positive_check"]["status"] == "pass"
    assert len(calls) == 10
    assert all(
        flag in calls[0]
        for flag in (
            "--rps", "--num-requests", "--prompt-tokens",
            "--output-tokens", "--scenario-tag",
        )
    )

    resumed = run_validation(
        output_dir=out,
        repeat=3,
        num_requests=3,
        dry_run=True,
        available_hooks=hooks,
        command_runner=mocked_runner,
    )
    assert resumed["evaluated_runs"] == 9
    assert len(calls) == 10
    saved = json.loads((out / "e1_summary.json").read_text(encoding="utf-8"))
    assert saved["data_source"] == "synthetic"


def test_unimplemented_live_hooks_are_reported_pending_without_subprocess(tmp_path):
    hooks = {name: False for name in INJECTIONS}

    def unexpected_runner(*args, **kwargs):
        pytest.fail("No live subprocess should run when every injection is pending")

    summary = run_validation(
        output_dir=tmp_path / "live-results",
        dry_run=False,
        available_hooks=hooks,
        command_runner=unexpected_runner,
    )
    assert summary["status"] == "pending_live"
    assert summary["data_source"] == "live"
    assert summary["pending_live_injections"] == list(INJECTIONS)
    assert summary["evaluated_runs"] == 0
