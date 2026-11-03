"""Analyze completed E1 run folders without starting a server or load generator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..analyzer.bottleneck_analyzer import BottleneckAnalyzer
from ..analyzer.readers import (
    CONTRACT_CSV_COLUMNS,
    join_by_request_id,
    read_request_csv,
    read_trace_jsonl,
)
from ..instrumentation.stage_timing import STAGE_NAME_SET

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "results"
EXPECTED_STAGES = {
    "slow_tokenizer": "tokenization",
    "no_batching": "queue_wait",
    "cpu_preprocess": "tokenization",
    "control": "decode",
}
INJECTED_STAGES = {
    EXPECTED_STAGES["slow_tokenizer"],
    EXPECTED_STAGES["no_batching"],
    EXPECTED_STAGES["cpu_preprocess"],
}
CSV_FIELDS = [
    "run",
    "injection",
    "repeat",
    "model",
    "expected_stage",
    "top_stage",
    "top3_stages",
    "explained_share",
    "explained_share_warning",
    "request_count",
    "requests_with_trace",
    "requests_without_trace",
    "missing_trace_percent",
    "unknown_stage_count",
    "unexpected_trace_request_ids",
    "validity",
    "detection_status",
    "control_false_positive",
    "message",
]


def _read_metadata(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: run.json must contain a JSON object")
    required = {"injection", "repeat"}
    missing = required - set(metadata)
    extra = set(metadata) - (required | {"model"})
    if missing or extra:
        raise ValueError(
            f"{path}: invalid fields; missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    injection = metadata["injection"]
    if not isinstance(injection, str) or injection not in EXPECTED_STAGES:
        raise ValueError(
            f"{path}: injection must be one of {sorted(EXPECTED_STAGES)}, got {injection!r}"
        )
    repeat = metadata["repeat"]
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError(f"{path}: repeat must be a positive integer")
    model = metadata.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError(f"{path}: model, when provided, must be a non-empty string")
    return metadata


def _read_strict_requests(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: request CSV is missing a header")
        missing_columns = [column for column in CONTRACT_CSV_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"{path}: missing request CSV columns: {missing_columns}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"{path}: request CSV contains no request rows")

    requests = read_request_csv(path)
    if len(requests) != len(raw_rows):
        raise ValueError(f"{path}: one or more request rows are malformed")
    request_ids = [str(row["request_id"]) for row in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError(f"{path}: duplicate request_id values are not allowed")
    return requests


def _read_trace_ids(path: Path) -> Set[str]:
    trace_ids: Set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: trace record must be a JSON object")
            missing = {"request_id", "stage", "start_ts", "end_ts", "duration_s"} - set(record)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing trace fields: {sorted(missing)}")
            request_id = record["request_id"]
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError(f"{path}:{line_number}: request_id must be a non-empty string")
            trace_ids.add(request_id.strip())
    return trace_ids


def _analyze_run(run_dir: Path, analyzer: BottleneckAnalyzer) -> Dict[str, Any]:
    expected_files = {
        "requests.csv": run_dir / "requests.csv",
        "traces.jsonl": run_dir / "traces.jsonl",
        "run.json": run_dir / "run.json",
    }
    absent = [name for name, path in expected_files.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{run_dir}: missing required files: {', '.join(absent)}")

    metadata = _read_metadata(expected_files["run.json"])
    requests = _read_strict_requests(expected_files["requests.csv"])
    trace_request_ids = _read_trace_ids(expected_files["traces.jsonl"])
    traces, trace_stats = read_trace_jsonl(expected_files["traces.jsonl"])
    if trace_stats["skipped_malformed"]:
        raise ValueError(
            f"{expected_files['traces.jsonl']}: {trace_stats['skipped_malformed']} malformed trace records"
        )

    request_ids = {row["request_id"] for row in requests}
    unexpected_trace_ids = trace_request_ids - request_ids
    requests_with_trace_ids = trace_request_ids & request_ids
    missing_request_ids = request_ids - trace_request_ids
    missing_percent = 100.0 * len(missing_request_ids) / len(request_ids)
    unknown_stages = int(trace_stats["skipped_unknown_stages"])

    # Validate that the accepted records really join to the CSV request set.
    joined = join_by_request_id(requests, traces)
    requests_with_valid_trace = {
        request_id for request_id, item in joined.items() if item["traces"]
    }
    missing_valid_trace = request_ids - requests_with_valid_trace
    valid_trace_percent = 100.0 * len(missing_valid_trace) / len(request_ids)
    if valid_trace_percent > missing_percent:
        missing_percent = valid_trace_percent

    analysis = analyzer.analyze(
        expected_files["requests.csv"],
        expected_files["traces.jsonl"],
        max_bottlenecks=3,
    )
    top_bottlenecks = analysis.get("top_bottlenecks", [])
    top_stages = [str(item.get("stage", "")) for item in top_bottlenecks]
    top_stage = top_stages[0] if top_stages else ""
    top_share = float(top_bottlenecks[0].get("share_of_latency", 0.0)) if top_bottlenecks else 0.0
    explained_share = float(analysis.get("explained_share", 0.0))
    warning = (
        not math.isfinite(explained_share)
        or explained_share < 0.90
        or explained_share > 1.10
    )
    validity_reasons: List[str] = []
    if missing_percent > 5.0:
        validity_reasons.append(f"{missing_percent:.2f}% of requests lack a valid trace (limit: 5%)")
    if unexpected_trace_ids:
        validity_reasons.append(f"{len(unexpected_trace_ids)} trace request_id(s) are absent from requests.csv")
    if unknown_stages:
        validity_reasons.append(f"{unknown_stages} trace record(s) use unknown stages")
    validity = "valid" if not validity_reasons else "invalid"

    injection = str(metadata["injection"])
    expected_stage = EXPECTED_STAGES[injection]
    is_control_false_positive = (
        injection == "control"
        and top_stage in INJECTED_STAGES
        and top_share >= 0.50
    )
    detection_status = (
        "invalid" if validity != "valid"
        else "not_applicable" if injection == "control"
        else "pass" if top_stage == expected_stage
        else "fail"
    )

    return {
        "run": run_dir.name,
        "injection": injection,
        "repeat": metadata["repeat"],
        "model": metadata.get("model", ""),
        "expected_stage": expected_stage,
        "top_stage": top_stage,
        "top3_stages": top_stages,
        "explained_share": explained_share,
        "explained_share_warning": warning,
        "request_count": len(request_ids),
        "requests_with_trace": len(request_ids) - len(missing_valid_trace),
        "requests_without_trace": len(missing_valid_trace),
        "missing_trace_percent": missing_percent,
        "unknown_stage_count": unknown_stages,
        "unexpected_trace_request_ids": len(unexpected_trace_ids),
        "validity": validity,
        "detection_status": detection_status,
        "control_false_positive": is_control_false_positive if injection == "control" else "",
        "message": "; ".join(validity_reasons),
        "analysis": analysis,
    }


def _summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    materialized = list(rows)
    valid_injections = [
        row for row in materialized
        if row["validity"] == "valid" and row["injection"] != "control"
    ]
    evaluated = [row for row in valid_injections if row["detection_status"] in {"pass", "fail"}]
    passed = sum(row["detection_status"] == "pass" for row in evaluated)
    top3_hits = sum(row["expected_stage"] in row["top3_stages"] for row in evaluated)
    controls = [
        row for row in materialized
        if row["injection"] == "control" and row["validity"] == "valid"
    ]
    control_false_positives = sum(bool(row["control_false_positive"]) for row in controls)
    invalid_runs = sum(row["validity"] != "valid" for row in materialized)
    return {
        "experiment_id": "E1",
        "data_source": "live",
        "status": "invalid_runs" if invalid_runs else "completed",
        "run_count": len(materialized),
        "evaluated_injection_runs": len(evaluated),
        "detection_accuracy": passed / len(evaluated) if evaluated else None,
        "top3_hit_rate": top3_hits / len(evaluated) if evaluated else None,
        "control_false_positive_check": {
            "status": (
                "not_run" if not controls
                else "fail" if control_false_positives
                else "pass"
            ),
            "control_runs": len(controls),
            "false_positives": control_false_positives,
        },
        "explained_share_by_run": [
            {
                "run": row["run"],
                "explained_share": row["explained_share"],
                "warning": row["explained_share_warning"],
            }
            for row in materialized
        ],
        "run_validity": [
            {"run": row["run"], "validity": row["validity"], "message": row["message"]}
            for row in materialized
        ],
        "unknown_stage_count": sum(row["unknown_stage_count"] for row in materialized),
    }


def analyze_run_dir(
    run_root: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    root = Path(run_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Run folder does not exist or is not a directory: {root}")
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No run subfolders found under {root}")

    analyzer = BottleneckAnalyzer(config_path=config_path)
    rows = [_analyze_run(path, analyzer) for path in run_dirs]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "e1_detection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in CSV_FIELDS},
                    "top3_stages": json.dumps(row["top3_stages"]),
                }
            )

    summary = _summary(rows)
    with (output / "e1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    for row in rows:
        if row["explained_share_warning"]:
            print(
                f"[WARN] {row['run']}: explained_share {row['explained_share']:.4f} "
                "is outside [0.90, 1.10]",
                file=sys.stderr,
            )
        if row["validity"] != "valid":
            print(f"[INVALID] {row['run']}: {row['message']}", file=sys.stderr)
    print(f"[INFO] Wrote {output / 'e1_detection.csv'} and {output / 'e1_summary.json'}")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze completed E1 result folders without launching the server."
    )
    parser.add_argument("--run-dir", required=True, help="Parent folder containing one subfolder per run.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for e1_detection.csv and e1_summary.json.",
    )
    parser.add_argument("--config", default=None, help="Optional analyzer config path.")
    args = parser.parse_args(argv)
    try:
        analyze_run_dir(args.run_dir, output_dir=args.out_dir, config_path=args.config)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] External run analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
