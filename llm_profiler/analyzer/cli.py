"""Command Line Interface for LLM Inference Profiler Bottleneck Analyzer (Member 4 - Sprint 2).

Provides unified CLI with support for:
- CSV request logs (--csv)
- Per-stage trace JSON lines (--trace)
- Optional memory breakdown (--memory-breakdown)
- Prometheus live queries and fallback (--prometheus-url, --source trace|prometheus)
- Contract JSON output formatting (--out)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .bottleneck_analyzer import BottleneckAnalyzer
from .readers import PrometheusClient, PrometheusUnreachableError, resolve_prometheus_url


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="llmprof-analyzer",
        description="Analyze LLM serving latency bottlenecks, cost, and memory telemetry.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to request log CSV file (required).",
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="Path to per-stage trace JSON lines file (optional).",
    )
    parser.add_argument(
        "--memory-breakdown",
        default=None,
        help="Path to optional M3 memory_breakdown.json (optional).",
    )
    parser.add_argument(
        "--prometheus-url",
        default=None,
        help="Base URL for Prometheus HTTP API (read from config.yaml or PROMETHEUS_URL if omitted).",
    )
    parser.add_argument(
        "--source",
        choices=["trace", "prometheus"],
        default="trace",
        help="Primary telemetry source for stage latency decomposition (default: trace).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to output JSON file. If omitted, results are written to stdout.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to central config.yaml (optional).",
    )
    parser.add_argument(
        "--gpu-price",
        type=float,
        default=None,
        help="Override GPU price per hour USD (loaded from config.yaml by default).",
    )
    parser.add_argument(
        "--full-time-share",
        action="store_true",
        help="Force time_share=1.0 for dedicated GPU instance cost modeling.",
    )
    parser.add_argument(
        "--max-bottlenecks",
        type=int,
        default=3,
        help="Maximum number of top bottlenecks to report (default: 3).",
    )
    return parser


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute analysis with fallback handling across data sources."""
    csv_path = args.csv
    trace_path = args.trace
    memory_path = args.memory_breakdown
    prom_url = resolve_prometheus_url(args.prometheus_url, args.config)

    analyzer = BottleneckAnalyzer(
        gpu_price_per_hour_usd=args.gpu_price,
        config_path=args.config,
        force_full_time_share=args.full_time_share,
    )

    prom_client = PrometheusClient(base_url=prom_url, config_path=args.config, timeout=3.0)
    gpu_metrics: Optional[Dict[str, Any]] = None
    stage_metrics: Optional[Dict[str, Dict[str, float]]] = None
    memory_metrics: Optional[Dict[str, Dict[str, float]]] = None

    if args.source == "prometheus":
        try:
            stage_summary = prom_client.get_stage_latency_summary()
            if any(
                stage != "total"
                and "mean" in metrics
                and float(metrics["mean"]) > 0
                for stage, metrics in stage_summary.items()
            ):
                stage_metrics = stage_summary
            else:
                print("per-stage metrics not available yet", file=sys.stderr)
                if trace_path and os.path.exists(trace_path):
                    print(f"[INFO] Falling back to trace file: {trace_path}", file=sys.stderr)
            gpu_metrics = prom_client.get_gpu_gauges()
            memory_metrics = prom_client.get_memory_breakdown_gauges()
        except PrometheusUnreachableError as exc:
            print(f"[WARN] {exc}", file=sys.stderr)
            if trace_path and os.path.exists(trace_path):
                print(f"[INFO] Falling back to trace file: {trace_path}", file=sys.stderr)
                gpu_metrics = None
                memory_metrics = None
            else:
                raise PrometheusUnreachableError(
                    f"{exc}; no valid --trace fallback file was provided"
                ) from exc
    else:
        try:
            gpu_metrics = prom_client.get_gpu_gauges()
            memory_metrics = prom_client.get_memory_breakdown_gauges()
        except PrometheusUnreachableError as exc:
            print(f"[WARN] {exc}; continuing with trace and file data.", file=sys.stderr)
            gpu_metrics = None
            memory_metrics = None

    result = analyzer.analyze(
        csv_path=csv_path,
        traces_path=trace_path,
        memory_breakdown_path=memory_path,
        gpu_metrics=gpu_metrics,
        stage_metrics=stage_metrics,
        memory_metrics=memory_metrics,
        max_bottlenecks=args.max_bottlenecks,
        force_full_time_share=args.full_time_share,
    )

    return result


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for analyzer CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_analysis(args)
    except FileNotFoundError as err:
        print(f"[ERROR] Input file not found: {err}", file=sys.stderr)
        sys.exit(1)
    except (PrometheusUnreachableError, OSError, ValueError) as err:
        print(f"[ERROR] Analysis failed: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"[ERROR] Analysis failed: {err}", file=sys.stderr)
        sys.exit(1)

    formatted_json = json.dumps(result, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(formatted_json + "\n")
        print(f"[INFO] Analysis saved to {args.out}", file=sys.stderr)

    # Always output JSON to stdout according to contract
    print(formatted_json)


if __name__ == "__main__":
    main()
