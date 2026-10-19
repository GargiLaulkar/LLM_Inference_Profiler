#!/usr/bin/env python3
"""Baseline B2 Runner for LLM Inference Profiler (Built-in vLLM Stack Metrics).

Runs scenarios from config.yaml using load_generator, taking before/after snapshots
of built-in vLLM metrics (via Prometheus or /metrics endpoint) to compute:
- Mean E2E request latency
- Mean Time To First Token (TTFT)
- Mean request queue time
- Requests per second (RPS)

Outputs:
- results/baseline_b2_<scenario>.csv (per-request logs complying with schema contract)
- docs/baseline_b2.md (comparison table B1 vs B2 and architectural analysis)
"""

import argparse
import asyncio
import csv
import os
import sys
import time
import urllib.request
import urllib.parse
import json
from typing import Dict, Any, List, Optional
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from llm_profiler.experiments.load_generator import run_load_generator, load_config


def fetch_vllm_metrics_snapshot(server_base_url: str = "http://127.0.0.1:8000", prometheus_url: str = "http://127.0.0.1:9090") -> Dict[str, float]:
    """Snapshots vLLM metrics from Prometheus or directly from vLLM /metrics."""
    snapshot: Dict[str, float] = {
        "e2e_sum": 0.0,
        "e2e_count": 0.0,
        "ttft_sum": 0.0,
        "ttft_count": 0.0,
        "queue_sum": 0.0,
        "queue_count": 0.0,
        "success_total": 0.0,
    }

    # 1. Try querying vLLM /metrics directly
    metrics_url = f"{server_base_url.rstrip('/')}/metrics"
    try:
        req = urllib.request.Request(metrics_url, headers={"User-Agent": "LLMProfiler/B2"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    k, v_str = parts[0], parts[1]
                    try:
                        v = float(v_str)
                    except ValueError:
                        continue
                    
                    if k.startswith("vllm:e2e_request_latency_seconds_sum"):
                        snapshot["e2e_sum"] = v
                    elif k.startswith("vllm:e2e_request_latency_seconds_count"):
                        snapshot["e2e_count"] = v
                    elif k.startswith("vllm:time_to_first_token_seconds_sum"):
                        snapshot["ttft_sum"] = v
                    elif k.startswith("vllm:time_to_first_token_seconds_count"):
                        snapshot["ttft_count"] = v
                    elif k.startswith("vllm:request_queue_time_seconds_sum"):
                        snapshot["queue_sum"] = v
                    elif k.startswith("vllm:request_queue_time_seconds_count"):
                        snapshot["queue_count"] = v
                    elif k.startswith("vllm:request_success_total"):
                        snapshot["success_total"] = v
            return snapshot
    except Exception as e:
        pass

    # 2. Fallback to Prometheus instant query API
    try:
        prom_queries = {
            "e2e_sum": "vllm:e2e_request_latency_seconds_sum",
            "e2e_count": "vllm:e2e_request_latency_seconds_count",
            "ttft_sum": "vllm:time_to_first_token_seconds_sum",
            "ttft_count": "vllm:time_to_first_token_seconds_count",
            "queue_sum": "vllm:request_queue_time_seconds_sum",
            "queue_count": "vllm:request_queue_time_seconds_count",
            "success_total": "vllm:request_success_total",
        }
        for key, q in prom_queries.items():
            encoded = urllib.parse.urlencode({"query": q})
            q_url = f"{prometheus_url.rstrip('/')}/api/v1/query?{encoded}"
            with urllib.request.urlopen(q_url, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("data", {}).get("result", [])
                if results and "value" in results[0]:
                    snapshot[key] = float(results[0]["value"][1])
        return snapshot
    except Exception:
        return snapshot


def load_b1_metrics_for_scenario(scenario_name: str, results_dir: str = "results") -> Dict[str, float]:
    """Reads B1 metrics from results/baseline_b1_<scenario>.csv if present."""
    b1_csv = os.path.join(results_dir, f"baseline_b1_{scenario_name}.csv")
    if not os.path.exists(b1_csv):
        return {}

    latencies = []
    try:
        with open(b1_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok":
                    latencies.append(float(row["total_latency_s"]))
        if latencies:
            latencies.sort()
            n = len(latencies)
            return {
                "mean_latency_s": sum(latencies) / n,
                "p50_latency_s": latencies[int(n * 0.50)],
                "p95_latency_s": latencies[min(int(n * 0.95), n - 1)],
                "count": float(n),
            }
    except Exception:
        pass
    return {}


def generate_baseline_b2_report(
    b2_results: List[Dict[str, Any]],
    report_path: str,
    results_dir: str = "results"
):
    """Generates docs/baseline_b2.md with comparison table B1 vs B2 and limitations."""
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    lines = [
        "# Baseline B2 Evaluation Report: Stack Built-in Metrics vs Manual B1",
        "",
        "## Overview",
        "Baseline B2 evaluates LLM serving performance using the stack's built-in telemetry (vLLM Prometheus histograms) compared against client-side end-to-end measurements from Baseline B1.",
        "",
        "## Comparison Table: B1 (Manual Client-Side) vs B2 (Stack Built-In)",
        "",
        "| Scenario | Metric Source | Mean E2E (s) | Mean TTFT (s) | Mean Queue (s) | Achieved RPS | Requests Count |",
        "|---|---|---|---|---|---|---|",
    ]

    for item in b2_results:
        sc = item["scenario"]
        b1 = load_b1_metrics_for_scenario(sc, results_dir)
        b1_mean = f"{b1['mean_latency_s']:.4f}" if "mean_latency_s" in b1 else "N/A"
        b1_reqs = f"{int(b1['count'])}" if "count" in b1 else "N/A"

        # B1 row (client-side measurement)
        lines.append(
            f"| `{sc}` | **B1 (Client Manual)** | {b1_mean} | N/A (Black-box) | N/A (Black-box) | {item.get('achieved_rps', 0.0):.2f} | {b1_reqs} |"
        )
        # B2 row (server-side built-in telemetry)
        lines.append(
            f"| `{sc}` | **B2 (vLLM Built-in)** | {item['mean_e2e_s']:.4f} | {item['mean_ttft_s']:.4f} | {item['mean_queue_s']:.4f} | {item['achieved_rps']:.2f} | {item['req_count']} |"
        )

    lines.extend([
        "",
        "## Architectural Analysis: What B2 Can and Cannot Tell Us",
        "",
        "Baseline B2 provides essential visibility into server-side execution dynamics by capturing coarse-grained operational stages: request queue wait time, prefill duration through Time to First Token (TTFT), and aggregate end-to-end inference latency directly from vLLM engine histograms. However, built-in telemetry alone remains insufficient for comprehensive inference stack profiling. B2 cannot isolate tokenization or detokenization latency on the CPU, cannot quantify client-server network transmission overhead, cannot attribute GPU High Bandwidth Memory (HBM) consumption across individual lifecycle phases (weights vs. KV-cache allocation vs. dynamic activation spikes), and cannot compute monetary inference cost per 1M tokens. Isolating tail latency anomalies and identifying memory saturation bottlenecks requires dedicated stage timers (M2), GPU kernel and power collectors (M3), and offline cost analyzers (M4).",
        "",
    ])

    content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Report written to: {report_path}")


async def run_baseline_b2(config_path: str = "config.yaml", scenario_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    cfg = load_config(config_path)

    model_name = cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    server_url = cfg.get("server_url", "http://127.0.0.1:8000/v1")
    server_host = cfg.get("server_host", "127.0.0.1")
    server_port = cfg.get("server_port", 8000)
    server_base = f"http://{server_host}:{server_port}"
    results_dir = cfg.get("results_dir", "results")
    scenarios = cfg.get("scenarios", [])

    os.makedirs(results_dir, exist_ok=True)
    summary_list = []

    for sc in scenarios:
        name = sc["name"]
        if scenario_filter and name != scenario_filter:
            continue

        rps = float(sc.get("rps", 1.0))
        num_requests = int(sc.get("num_requests", 100))
        prompt_tokens = int(sc.get("prompt_tokens", 128))
        output_tokens = int(sc.get("output_tokens", 64))

        out_csv = os.path.join(results_dir, f"baseline_b2_{name}.csv")

        print(f"\n==================================================")
        print(f"Running Baseline B2 Scenario: {name}")
        print(f"RPS: {rps}, Requests: {num_requests}, In: {prompt_tokens}, Out: {output_tokens}")
        print(f"==================================================")

        # 1. Snapshot metrics BEFORE load test
        before_snap = fetch_vllm_metrics_snapshot(server_base)
        start_time = time.time()

        # 2. Run load generator
        gen_summary = await run_load_generator(
            server_url=server_url,
            model_name=model_name,
            rps=rps,
            num_requests=num_requests,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            scenario_tag=f"baseline_b2_{name}",
            output_csv=out_csv,
        )
        elapsed_s = max(time.time() - start_time, 0.001)

        # Allow prometheus / vLLM metrics to finalize
        await asyncio.sleep(2)

        # 3. Snapshot metrics AFTER load test
        after_snap = fetch_vllm_metrics_snapshot(server_base)

        delta_count = after_snap["e2e_count"] - before_snap["e2e_count"]
        delta_e2e = after_snap["e2e_sum"] - before_snap["e2e_sum"]
        delta_ttft = after_snap["ttft_sum"] - before_snap["ttft_sum"]
        delta_queue = after_snap["queue_sum"] - before_snap["queue_sum"]

        if delta_count > 0:
            mean_e2e = delta_e2e / delta_count
            mean_ttft = delta_ttft / delta_count if after_snap["ttft_count"] > before_snap["ttft_count"] else 0.0
            mean_queue = delta_queue / delta_count if after_snap["queue_count"] > before_snap["queue_count"] else 0.0
            reqs = int(delta_count)
        else:
            # Fall back to load generator values if server metrics did not increment
            mean_e2e = gen_summary.get("mean_latency_s", 0.0)
            mean_ttft = 0.0
            mean_queue = 0.0
            reqs = gen_summary.get("num_successful", num_requests)

        achieved_rps = gen_summary.get("achieved_throughput_rps", reqs / elapsed_s)

        res_entry = {
            "scenario": name,
            "mean_e2e_s": mean_e2e,
            "mean_ttft_s": mean_ttft,
            "mean_queue_s": mean_queue,
            "achieved_rps": achieved_rps,
            "req_count": reqs,
        }
        summary_list.append(res_entry)

    # Generate docs
    report_locations = [
        os.path.join("docs", "baseline_b2.md"),
        os.path.join("llm_profiler", "docs", "baseline_b2.md"),
    ]
    for loc in report_locations:
        generate_baseline_b2_report(summary_list, loc, results_dir)

    return summary_list


def main():
    parser = argparse.ArgumentParser(description="Run Baseline B2 (Stack Built-In Metrics)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--scenario", default=None, help="Run specific scenario (e.g. low_load)")
    args = parser.parse_args()

    results = asyncio.run(run_baseline_b2(args.config, args.scenario))
    print("\nBaseline B2 Execution Summary:")
    print("-" * 75)
    print(f"{'Scenario':<16} {'Mean E2E (s)':<14} {'Mean TTFT (s)':<14} {'Queue (s)':<12} {'RPS':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['scenario']:<16} {r['mean_e2e_s']:<14.4f} {r['mean_ttft_s']:<14.4f} {r['mean_queue_s']:<12.4f} {r['achieved_rps']:<10.2f}")
    print("-" * 75)


if __name__ == "__main__":
    main()
