#!/usr/bin/env python3
"""Scenario Matrix Runner for LLM Inference Profiler.

Executes parameterized scenario sweeps (load levels, prompt/output length mixes)
from config.yaml with repeat support, tagging CSVs and maintaining results/index.csv.
"""

import argparse
import asyncio
import csv
import datetime
import os
import sys
import time
from typing import Dict, Any, List, Optional
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from llm_profiler.experiments.load_generator import run_load_generator, load_config

INDEX_HEADERS = [
    "run_id",
    "scenario",
    "repeat",
    "rps",
    "num_requests",
    "prompt_tokens",
    "output_tokens",
    "csv_file",
    "mean_latency_s",
    "p50_latency_s",
    "p95_latency_s",
    "throughput_rps",
    "success_rate_pct",
    "timestamp",
]


def update_index_csv(index_path: str, row_data: Dict[str, Any]):
    """Appends or creates results/index.csv with the run record."""
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    write_header = not os.path.exists(index_path) or os.path.getsize(index_path) == 0

    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row_data)


async def run_scenario_matrix(
    config_path: str = "config.yaml",
    scenario_filter: Optional[str] = None,
    repeat: int = 1,
) -> List[Dict[str, Any]]:
    cfg = load_config(config_path)

    model_name = cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    server_url = cfg.get("server_url", "http://127.0.0.1:8000/v1")
    results_dir = cfg.get("results_dir", "results")
    scenarios = cfg.get("scenarios", [])

    os.makedirs(results_dir, exist_ok=True)
    index_path = os.path.join(results_dir, "index.csv")
    executed_runs = []

    for sc in scenarios:
        name = sc["name"]
        if scenario_filter and name != scenario_filter:
            continue

        rps = float(sc.get("rps", 1.0))
        num_requests = int(sc.get("num_requests", 50))
        prompt_tokens = int(sc.get("prompt_tokens", 128))
        output_tokens = int(sc.get("output_tokens", 64))

        for rep_idx in range(1, repeat + 1):
            tag = f"matrix_{name}_rep{rep_idx}" if repeat > 1 else f"matrix_{name}"
            out_csv = os.path.join(results_dir, f"{tag}.csv")
            run_id = f"{name}_r{rep_idx}_{int(time.time())}"

            print(f"\n==================================================")
            print(f"Matrix Run: {name} (Repeat {rep_idx}/{repeat})")
            print(f"RPS: {rps}, Requests: {num_requests}, In: {prompt_tokens}, Out: {output_tokens}")
            print(f"Output CSV: {out_csv}")
            print(f"==================================================")

            summary = await run_load_generator(
                server_url=server_url,
                model_name=model_name,
                rps=rps,
                num_requests=num_requests,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                scenario_tag=tag,
                output_csv=out_csv,
            )

            record = {
                "run_id": run_id,
                "scenario": name,
                "repeat": rep_idx,
                "rps": rps,
                "num_requests": num_requests,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "csv_file": os.path.basename(out_csv),
                "mean_latency_s": round(summary.get("mean_latency_s", 0.0), 4),
                "p50_latency_s": round(summary.get("p50_latency_s", 0.0), 4),
                "p95_latency_s": round(summary.get("p95_latency_s", 0.0), 4),
                "throughput_rps": round(summary.get("achieved_throughput_rps", 0.0), 2),
                "success_rate_pct": round(summary.get("success_rate_pct", 0.0), 2),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }

            update_index_csv(index_path, record)
            executed_runs.append(record)

    return executed_runs


def main():
    parser = argparse.ArgumentParser(description="Run Scenario Matrix for LLM Profiler")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--scenario", default=None, help="Run specific scenario by name (e.g. rps_1)")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions per scenario")
    args = parser.parse_args()

    runs = asyncio.run(run_scenario_matrix(args.config, args.scenario, args.repeat))
    print(f"\nExecuted {len(runs)} matrix runs. Results logged to results/index.csv.")


if __name__ == "__main__":
    main()
