from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_profiler.analyzer.readers import CONTRACT_CSV_COLUMNS
from llm_profiler.experiments.analyze_run_dir import analyze_run_dir


def _write_run(
    run_dir: Path,
    *,
    injection: str = "slow_tokenizer",
    request_count: int = 2,
    missing_trace_for: set[str] | None = None,
    unknown_stage: bool = False,
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"injection": injection, "repeat": 1, "model": "tiny-test-model"}),
        encoding="utf-8",
    )
    requests = []
    for index in range(request_count):
        request_id = f"req-{index}"
        requests.append(
            {
                "request_id": request_id,
                "send_ts": index * 2,
                "end_ts": index * 2 + 1,
                "total_latency_s": 1,
                "prompt_tokens": 2,
                "output_tokens": 1,
                "status": "ok",
                "target_rps": 1,
                "scenario_tag": injection,
            }
        )
    with (run_dir / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTRACT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(requests)

    missing_trace_for = missing_trace_for or set()
    trace_records = []
    for request in requests:
        request_id = request["request_id"]
        if request_id in missing_trace_for:
            continue
        dominant = "tokenization" if injection == "slow_tokenizer" else "decode"
        trace_records.extend(
            [
                {
                    "request_id": request_id,
                    "stage": dominant,
                    "start_ts": request["send_ts"],
                    "end_ts": request["send_ts"] + 0.8,
                    "duration_s": 0.8,
                },
                {
                    "request_id": request_id,
                    "stage": "prefill",
                    "start_ts": request["send_ts"] + 0.8,
                    "end_ts": request["end_ts"],
                    "duration_s": 0.2,
                },
            ]
        )
    if unknown_stage:
        trace_records.append(
            {
                "request_id": "req-0",
                "stage": "made_up",
                "start_ts": 0,
                "end_ts": 0.1,
                "duration_s": 0.1,
            }
        )
    (run_dir / "traces.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in trace_records),
        encoding="utf-8",
    )


def test_external_run_analysis_writes_live_detection_and_summary(tmp_path):
    run_root = tmp_path / "runs"
    _write_run(run_root / "slow_01")
    _write_run(run_root / "control_01", injection="control")
    output_dir = tmp_path / "out"

    summary = analyze_run_dir(run_root, output_dir=output_dir)

    assert summary["data_source"] == "live"
    assert summary["detection_accuracy"] == 1.0
    assert summary["top3_hit_rate"] == 1.0
    assert summary["control_false_positive_check"]["status"] == "pass"
    rows = list(csv.DictReader((output_dir / "e1_detection.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert all(row["validity"] == "valid" for row in rows)
    assert json.loads((output_dir / "e1_summary.json").read_text(encoding="utf-8"))["data_source"] == "live"


def test_more_than_five_percent_missing_traces_invalidates_run(tmp_path):
    run_root = tmp_path / "runs"
    _write_run(
        run_root / "slow_01",
        request_count=10,
        missing_trace_for={"req-0"},
    )

    summary = analyze_run_dir(run_root, output_dir=tmp_path / "out")

    assert summary["run_validity"][0]["validity"] == "invalid"
    assert "10.00%" in summary["run_validity"][0]["message"]
    assert summary["detection_accuracy"] is None


def test_unknown_stages_are_counted_and_run_is_invalid(tmp_path):
    run_root = tmp_path / "runs"
    _write_run(run_root / "slow_01", unknown_stage=True)

    summary = analyze_run_dir(run_root, output_dir=tmp_path / "out")

    assert summary["unknown_stage_count"] == 1
    assert summary["run_validity"][0]["validity"] == "invalid"
    assert "unknown stages" in summary["run_validity"][0]["message"]


def test_unexpected_trace_request_id_is_invalid(tmp_path):
    run_root = tmp_path / "runs"
    run_dir = run_root / "slow_01"
    _write_run(run_dir)
    trace_path = run_dir / "traces.jsonl"
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    records[0]["request_id"] = "not-in-csv"
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    summary = analyze_run_dir(run_root, output_dir=tmp_path / "out")

    assert summary["run_validity"][0]["validity"] == "invalid"
    assert "absent from requests.csv" in summary["run_validity"][0]["message"]


def test_bad_run_metadata_fails_input_validation(tmp_path):
    run_root = tmp_path / "runs"
    _write_run(run_root / "slow_01")
    (run_root / "slow_01" / "run.json").write_text(
        json.dumps({"injection": "unknown", "repeat": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="injection must be one of"):
        analyze_run_dir(run_root, output_dir=tmp_path / "out")
