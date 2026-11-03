"""M4 experiment runner for the E2-E5 scenario matrix.

This module follows the handbook-oriented experiment IDs and keeps the runner
fully synthetic-safe in dry-run mode. In live mode it exits gracefully with a
clear pending-live message instead of failing when M2's hooks or second model
are not yet available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from ..analyzer.bottleneck_analyzer import BottleneckAnalyzer
from ..analyzer.bottleneck_injector import BottleneckInjector

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_FILE = ROOT / "experiments" / "m4_scenarios.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "results"


def _default_experiment_summary(experiment_id: str, status: str, data_source: str = "synthetic") -> Dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": status,
        "data_source": data_source,
        "notes": [],
    }


def load_scenarios(path: Optional[str | Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    scenario_path = Path(path) if path else DEFAULT_SCENARIO_FILE
    with scenario_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {str(k): list(v) for k, v in data.items()}


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_jsonl(path: Path, traces: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")


def _scenario_summary_for_rows(rows: List[Dict[str, Any]], scenario_tag: str) -> Dict[str, Any]:
    if not rows:
        return {"scenario_tag": scenario_tag, "status": "empty", "data_source": "synthetic"}
    total_latency = sum(float(r.get("total_latency_s", 0.0)) for r in rows)
    total_tokens = sum(int(r.get("prompt_tokens", 0)) + int(r.get("output_tokens", 0)) for r in rows)
    return {
        "scenario_tag": scenario_tag,
        "status": "completed_synthetic",
        "data_source": "synthetic",
        "request_count": len(rows),
        "mean_latency_s": round(total_latency / len(rows), 6) if rows else 0.0,
        "total_tokens": total_tokens,
    }


def run_experiment_synthetic(experiment_id: str, scenarios: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    injector = BottleneckInjector(seed=42)
    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_scenario: List[Dict[str, Any]] = []

    for scenario in scenarios:
        tag = str(scenario.get("scenario_tag", experiment_id))
        rows, traces = injector.generate_synthetic_workload(
            num_requests=int(scenario.get("num_requests", 20)),
            scenario_tag=tag,
            target_rps=float(scenario.get("rps", 1.0)),
            prompt_tokens=int(scenario.get("prompt_tokens", 128)),
            output_tokens=int(scenario.get("output_tokens", 64)),
            injected_stage="prefill" if experiment_id == "E3" else None,
            injection_delay_s=0.2 if experiment_id == "E3" else 0.0,
        )
        csv_path = output_dir / f"{tag}.csv"
        traces_path = output_dir / f"{tag}.jsonl"
        analysis_path = output_dir / f"{tag}_analysis.json"
        _write_csv(csv_path, rows)
        _write_jsonl(traces_path, traces)
        analysis = analyzer.analyze(csv_path, traces_path)
        with analysis_path.open("w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, sort_keys=True)
        per_scenario.append(
            {
                "scenario_tag": tag,
                "data_source": "synthetic",
                "status": "completed_synthetic",
                "request_count": len(rows),
                "top_bottleneck": analysis["top_bottlenecks"][0]["stage"],
                "explained_share": analysis["explained_share"],
                "csv": str(csv_path),
                "trace": str(traces_path),
                "analysis": str(analysis_path),
            }
        )

    summary = _default_experiment_summary(experiment_id, "completed_synthetic", "synthetic")
    summary["scenarios"] = per_scenario
    summary["result_dir"] = str(output_dir)
    summary["notes"] = [
        "Synthetic-only execution used for dry-run validation; no live server or M2 hooks were invoked."
    ]
    with (output_dir / f"{experiment_id.lower()}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def run_experiment_live(experiment_id: str, scenarios: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    pending_reasons: List[str] = []
    if not os.getenv("LLMPROF_TRACE_PATH"):
        pending_reasons.append("LLMPROF_TRACE_PATH is unset; M2 per-stage trace output is not merged yet.")
    if not any(key in os.environ for key in ("LLMPROF_SLOW_TOKENIZER_MS", "LLMPROF_NO_BATCHING", "LLMPROF_CPU_PREPROCESS_MS")):
        pending_reasons.append("No live M2 injection hooks are present in the environment yet.")
    if not (Path("llm_profiler") / "experiments" / "load_generator.py").exists():
        pending_reasons.append("M1 load-generator CLI is missing from the repository tree.")

    summary = _default_experiment_summary(experiment_id, "pending_live", "live")
    summary["pending_live_reasons"] = pending_reasons or ["Live execution is not yet available in this branch."]
    summary["notes"] = ["Live attempts are intentionally skipped until the required M2 hooks and trace output are merged."]
    with (output_dir / f"{experiment_id.lower()}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[PENDING LIVE] {experiment_id}: {'; '.join(summary['pending_live_reasons'])}")
    return summary


def run_suite(*, dry_run: bool = True, output_dir: Optional[str | Path] = None, scenario_file: Optional[str | Path] = None) -> Dict[str, Any]:
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    scenarios_by_id = load_scenarios(scenario_file)
    run_results: Dict[str, Any] = {}
    for experiment_id in ["E2", "E3", "E4", "E5"]:
        scenarios = scenarios_by_id.get(experiment_id, [])
        if not scenarios:
            continue
        if dry_run:
            result = run_experiment_synthetic(experiment_id, scenarios, out_dir / experiment_id.lower())
        else:
            result = run_experiment_live(experiment_id, scenarios, out_dir / experiment_id.lower())
        run_results[experiment_id] = result

    final = {
        "experiment_ids": list(run_results.keys()),
        "data_source": "synthetic" if dry_run else "live",
        "status": "completed" if dry_run else "pending_live",
        "results": run_results,
    }
    with (out_dir / "suite_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, sort_keys=True)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E2-E5 scenario matrix with synthetic dry-run by default.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data and mocked outputs; default is synthetic-safe.")
    parser.add_argument("--live", action="store_true", help="Attempt live execution; prints pending-live messages when M2 hooks are unavailable.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for raw CSV, traces, analyzer JSON, and summary files.")
    parser.add_argument("--scenario-file", type=str, default=str(DEFAULT_SCENARIO_FILE), help="YAML scenario file to load.")
    args = parser.parse_args()

    dry_run = args.dry_run or not args.live
    run_suite(dry_run=dry_run, output_dir=args.out_dir, scenario_file=args.scenario_file)
    print(f"[INFO] Completed { 'synthetic dry-run' if dry_run else 'live' } experiment suite; output at {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
