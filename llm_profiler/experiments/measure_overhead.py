"""Instrumentation Microbenchmark (Observer Timing and Metrics Overhead).

Measures the microsecond-level CPU overhead introduced by monotonic StageTimer context
managers, RequestTrace allocations, and Prometheus MetricRegistry ingestion.

NOTE: This is an isolated instrumentation microbenchmark measuring Python timing
primitives; it is NOT the complete live Experiment E6 result (which requires end-to-end
client-server load generation comparing server throughput with profiling ON vs OFF).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from ..instrumentation.stage_timing import STAGE_NAMES, StageTimer
from ..instrumentation.request_trace import RequestTrace
from ..instrumentation.prometheus_exporter import MetricRegistry


def measure_uninstrumented_workload(iterations: int = 10_000) -> float:
    """Measure total execution time of an uninstrumented synthetic loop."""
    t0 = time.perf_counter()
    for i in range(iterations):
        _ = f"req-{i}"
        _ = 100 + 64
        _ = [j * 2 for j in range(5)]
    t1 = time.perf_counter()
    return t1 - t0


def measure_instrumented_workload(iterations: int = 10_000) -> float:
    """Measure total execution time with StageTimer and RequestTrace recorded."""
    t0 = time.perf_counter()
    for i in range(iterations):
        req_id = f"req-{i}"
        trace = RequestTrace(request_id=req_id, total_duration_s=0.1)

        for stage in ("tokenization", "queue_wait", "prefill", "decode", "detokenization", "network"):
            with StageTimer(stage) as timer:
                _ = 100 + 64
                _ = [j * 2 for j in range(5)]
            trace.record_stage_duration(stage, timer.elapsed_seconds)

    t1 = time.perf_counter()
    return t1 - t0


def measure_full_telemetry_workload(iterations: int = 10_000) -> float:
    """Measure execution time with StageTimer, RequestTrace, and MetricRegistry ingest."""
    registry = MetricRegistry()
    t0 = time.perf_counter()
    for i in range(iterations):
        req_id = f"req-{i}"
        trace = RequestTrace(request_id=req_id, total_duration_s=0.1)

        for stage in ("tokenization", "queue_wait", "prefill", "decode", "detokenization", "network"):
            with StageTimer(stage) as timer:
                _ = 100 + 64
                _ = [j * 2 for j in range(5)]
            trace.record_stage_duration(stage, timer.elapsed_seconds)
            registry.record_stage_duration(stage, timer.elapsed_seconds)

        registry.record_request_total_duration(0.1)
        registry.record_tokens_in_total(100)
        registry.record_tokens_out_total(64)

    t1 = time.perf_counter()
    return t1 - t0


def evaluate_instrumentation_microbenchmark(iterations: int = 10_000) -> Dict[str, Any]:
    """Run instrumentation microbenchmark and compute timing deltas.

    This measures the microsecond overhead of timers and trace objects.
    It is labeled as synthetic microbenchmark data, NOT live E6.
    """
    uninstrumented_s = measure_uninstrumented_workload(iterations)
    instrumented_s = measure_instrumented_workload(iterations)
    full_telemetry_s = measure_full_telemetry_workload(iterations)

    uninstrumented_per_req_us = (uninstrumented_s / iterations) * 1_000_000.0
    instrumented_per_req_us = (instrumented_s / iterations) * 1_000_000.0
    full_telemetry_per_req_us = (full_telemetry_s / iterations) * 1_000_000.0

    delta_full_us = full_telemetry_per_req_us - uninstrumented_per_req_us
    assumed_request_lat_us = 100_000.0  # 100ms typical request
    overhead_pct_vs_serving_latency = (delta_full_us / assumed_request_lat_us) * 100.0

    return {
        "benchmark_type": "instrumentation_microbenchmark",
        "data_source": "synthetic",
        "note": "Microbenchmark only. NOT live E6 result. Live E6 requires server load generator profiling ON vs OFF.",
        "iterations": iterations,
        "uninstrumented_total_s": round(uninstrumented_s, 6),
        "instrumented_total_s": round(instrumented_s, 6),
        "full_telemetry_total_s": round(full_telemetry_s, 6),
        "uninstrumented_per_request_us": round(uninstrumented_per_req_us, 2),
        "instrumented_per_request_us": round(instrumented_per_req_us, 2),
        "full_telemetry_per_request_us": round(full_telemetry_per_req_us, 2),
        "profiler_overhead_per_request_us": round(delta_full_us, 2),
        "overhead_percent_on_100ms_request": round(overhead_pct_vs_serving_latency, 4),
        "is_negligible": overhead_pct_vs_serving_latency < 1.0,
    }


# Backwards compatibility alias
evaluate_observer_overhead = evaluate_instrumentation_microbenchmark


def main() -> None:
    """CLI runner for instrumentation microbenchmark."""
    parser = argparse.ArgumentParser(description="Run instrumentation microbenchmark (timer & telemetry overhead).")
    parser.add_argument("--iterations", type=int, default=10_000, help="Benchmark iteration count")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    print(f"[INFO] Running instrumentation microbenchmark ({args.iterations:,} iterations)...")
    print("  [NOTE] This is an instrumentation microbenchmark; NOT the live E6 result.")
    report = evaluate_instrumentation_microbenchmark(iterations=args.iterations)

    print("\n=== Instrumentation Microbenchmark Results ===")
    print(f"  Uninstrumented per request:   {report['uninstrumented_per_request_us']:.2f} us")
    print(f"  Stage Timers + Traces:        {report['instrumented_per_request_us']:.2f} us")
    print(f"  Full Telemetry + Prometheus:  {report['full_telemetry_per_request_us']:.2f} us")
    print(f"  Net Overhead per Request:     {report['profiler_overhead_per_request_us']:.2f} us")
    print(f"  Overhead % on 100ms Request:  {report['overhead_percent_on_100ms_request']:.4f}%")
    print(f"  Status:                       {'Negligible (< 1%)' if report['is_negligible'] else 'High'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] Microbenchmark report saved to {args.out}")


if __name__ == "__main__":
    main()
