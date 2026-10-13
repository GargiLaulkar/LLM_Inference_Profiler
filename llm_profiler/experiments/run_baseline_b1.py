#!/usr/bin/env python3
"""Baseline B1 Runner for LLM Inference Profiler.

Reads scenarios from config.yaml, runs each through load generator logic,
saves results/baseline_b1_<scenario>.csv, and writes docs/baseline_b1.md.
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Dict, Any, List
import yaml

# Import load generator logic directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from llm_profiler.experiments.load_generator import run_load_generator, load_config


def generate_baseline_markdown_report(results_table: List[Dict[str, Any]], report_path: str):
    """Writes the markdown summary table and honest limitation paragraph to docs/baseline_b1.md."""
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    lines = [
        "# Baseline B1 Evaluation Report",
        "",
        "## Overview",
        "Baseline B1 establishes black-box end-to-end performance profiles across varying load and prompt scenarios.",
        "",
        "## Summary Metrics",
        "",
        "| Scenario | Target RPS | Mean (s) | P50 (s) | P95 (s) | P99 (s) | Throughput (req/s) | Error Rate (%) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for row in results_table:
        error_rate = 100.0 - row["success_rate_pct"]
        lines.append(
            f"| `{row['name']}` | {row['rps']:.1f} | {row['mean_latency_s']:.4f} | {row['p50_latency_s']:.4f} | "
            f"{row['p95_latency_s']:.4f} | {row['p99_latency_s']:.4f} | {row['achieved_throughput_rps']:.2f} | {error_rate:.2f}% |"
        )

    lines.extend([
        "",
        "## Architectural Findings & Analysis",
        "",
        "B1 shows only total latency; it cannot say which stage is slow. Without fine-grained stage timers decomposing tokenization, queue wait, KV cache lookup, prefill compute, iterative decode steps, detokenization, and network transmission, black-box benchmarks cannot isolate server-side bottlenecks or identify whether high tail latency stems from memory bandwidth saturation, scheduling contention, or prompt prefix processing overhead.",
        "",
    ])

    content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Report written to: {report_path}")


async def run_baselines(config_path: str, scenario_filter: str = None) -> List[Dict[str, Any]]:
    cfg = load_config(config_path)

    model_name = cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    server_url = cfg.get("server_url", "http://127.0.0.1:8000/v1")
    results_dir = cfg.get("results_dir", "results")
    scenarios = cfg.get("scenarios", [])

    if not scenarios:
        raise ValueError("No scenarios found in configuration.")

    os.makedirs(results_dir, exist_ok=True)
    results_summary_table = []

    for sc in scenarios:
        name = sc["name"]
        if scenario_filter and name != scenario_filter:
            continue

        rps = float(sc.get("rps", 1.0))
        num_requests = int(sc.get("num_requests", 100))
        prompt_tokens = int(sc.get("prompt_tokens", 128))
        output_tokens = int(sc.get("output_tokens", 64))

        out_csv = os.path.join(results_dir, f"baseline_b1_{name}.csv")

        print(f"\n>>> Running Scenario: {name} (RPS: {rps}, Requests: {num_requests}, In: {prompt_tokens}, Out: {output_tokens})")

        summary = await run_load_generator(
            server_url=server_url,
            model_name=model_name,
            rps=rps,
            num_requests=num_requests,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            scenario_tag=f"baseline_b1_{name}",
            output_csv=out_csv,
        )

        row_data = {
            "name": name,
            "rps": rps,
            "mean_latency_s": summary["mean_latency_s"],
            "p50_latency_s": summary["p50_latency_s"],
            "p95_latency_s": summary["p95_latency_s"],
            "p99_latency_s": summary["p99_latency_s"],
            "achieved_throughput_rps": summary["achieved_throughput_rps"],
            "success_rate_pct": summary["success_rate_pct"],
        }
        results_summary_table.append(row_data)

    # Write report to docs
    report_locations = [
        os.path.join("llm_profiler", "docs", "baseline_b1.md"),
        os.path.join("docs", "baseline_b1.md"),
    ]
    for loc in report_locations:
        generate_baseline_markdown_report(results_summary_table, loc)

    return results_summary_table


def main():
    parser = argparse.ArgumentParser(description="Run Baseline B1 Profiling Scenarios")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--scenario", default=None, help="Run a specific scenario by name (e.g. low_load)")
    args = parser.parse_args()

    table = asyncio.run(run_baselines(args.config, args.scenario))
    
    # Exit with code 1 if any scenario has success rate < 95%
    for row in table:
        if row["success_rate_pct"] < 95.0:
            print(f"[FAILURE] Scenario {row['name']} success rate {row['success_rate_pct']:.2f}% is below 95% threshold!", file=sys.stderr)
            sys.exit(1)

    print("\nAll executed baseline scenarios met success criteria (>= 95%).")


if __name__ == "__main__":
    main()
