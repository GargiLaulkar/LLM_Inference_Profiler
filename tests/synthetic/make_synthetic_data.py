"""Seeded Synthetic Dataset Generator for Testing (Member 4 - Task 4).

Generates deterministic synthetic benchmark datasets and corresponding ground_truth.json:
- normal: Baseline conversation (decode dominant)
- slow_tokenizer: Heavy tokenization delay (tokenization dominant)
- no_batching_queue_heavy: FIFO queuing delay (queue_wait dominant)
- cpu_preprocessing: Heavy preprocessing delay (tokenization dominant)
- network_heavy: High transmission latency (network dominant)
- broken_sum: Total latency inconsistent with stage traces (triggers explained_share warning)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer.bottleneck_injector import BottleneckInjector


def generate_all_synthetic_scenarios(
    base_dir: str = os.path.join(os.path.dirname(__file__), "data"),
    num_requests: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """Generate all test scenarios with seeded reproducibility."""
    os.makedirs(base_dir, exist_ok=True)
    injector = BottleneckInjector(seed=42)

    scenarios = {
        "normal": {
            "injected_stage": None,
            "delay_s": 0.0,
            "expected_top_stage": "decode",
            "expected_warning": False,
        },
        "slow_tokenizer": {
            "injected_stage": "tokenization",
            "delay_s": 0.50,
            "expected_top_stage": "tokenization",
            "expected_warning": False,
        },
        "no_batching_queue_heavy": {
            "injected_stage": "queue_wait",
            "delay_s": 0.80,
            "expected_top_stage": "queue_wait",
            "expected_warning": False,
        },
        "cpu_preprocessing": {
            "injected_stage": "tokenization",
            "delay_s": 0.40,
            "expected_top_stage": "tokenization",
            "expected_warning": False,
        },
        "network_heavy": {
            "injected_stage": "network",
            "delay_s": 0.60,
            "expected_top_stage": "network",
            "expected_warning": False,
        },
    }

    manifest: Dict[str, Dict[str, Any]] = {}

    for name, config in scenarios.items():
        scenario_dir = os.path.join(base_dir, name)
        os.makedirs(scenario_dir, exist_ok=True)

        csv_rows, traces = injector.generate_synthetic_workload(
            num_requests=num_requests,
            scenario_tag=name,
            target_rps=5.0,
            prompt_tokens=128,
            output_tokens=64,
            injected_stage=config["injected_stage"],
            injection_delay_s=config["delay_s"],
        )

        csv_path = os.path.join(scenario_dir, "requests.csv")
        traces_path = os.path.join(scenario_dir, "traces.jsonl")
        ground_truth_path = os.path.join(scenario_dir, "ground_truth.json")

        # Save CSV
        headers = [
            "request_id", "send_ts", "end_ts", "total_latency_s",
            "prompt_tokens", "output_tokens", "status", "target_rps", "scenario_tag"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

        # Save Traces JSONL
        with open(traces_path, "w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")

        # Save Ground Truth
        ground_truth = {
            "scenario": name,
            "data_source": "synthetic",
            "injected_stage": config["injected_stage"],
            "delay_s": config["delay_s"],
            "expected_top_stage": config["expected_top_stage"],
            "expected_warning": config["expected_warning"],
            "requests_count": num_requests,
        }
        with open(ground_truth_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)

        manifest[name] = {
            "csv_path": csv_path,
            "traces_path": traces_path,
            "ground_truth_path": ground_truth_path,
            "ground_truth": ground_truth,
        }

    # ─── Generate Broken-Sum Scenario ─────────────────────────────────────────
    broken_dir = os.path.join(base_dir, "broken_sum")
    os.makedirs(broken_dir, exist_ok=True)

    # In broken_sum, total_latency_s is 2.50s, but stage traces only add up to 0.40s
    csv_rows, traces = injector.generate_synthetic_workload(
        num_requests=num_requests,
        scenario_tag="broken_sum",
        target_rps=5.0,
        prompt_tokens=128,
        output_tokens=64,
    )
    # Artificially inflate CSV latency to break invariant
    for r in csv_rows:
        cur_lat = float(r["total_latency_s"])
        broken_lat = cur_lat + 2.0  # Add 2.0s unexplained latency
        r["total_latency_s"] = f"{broken_lat:.6f}"
        r["end_ts"] = f"{float(r['send_ts']) + broken_lat:.6f}"

    broken_csv_path = os.path.join(broken_dir, "requests.csv")
    broken_traces_path = os.path.join(broken_dir, "traces.jsonl")
    broken_gt_path = os.path.join(broken_dir, "ground_truth.json")

    with open(broken_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    with open(broken_traces_path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    broken_gt = {
        "scenario": "broken_sum",
        "data_source": "synthetic",
        "expected_top_stage": "decode",
        "expected_warning": True,
        "warning_keyword": "outside nominal bounds",
        "requests_count": num_requests,
    }
    with open(broken_gt_path, "w", encoding="utf-8") as f:
        json.dump(broken_gt, f, indent=2)

    manifest["broken_sum"] = {
        "csv_path": broken_csv_path,
        "traces_path": broken_traces_path,
        "ground_truth_path": broken_gt_path,
        "ground_truth": broken_gt,
    }

    return manifest


def main() -> None:
    print("[INFO] Generating seeded synthetic test datasets...")
    manifest = generate_all_synthetic_scenarios()
    print(f"[SUCCESS] Generated {len(manifest)} scenarios under tests/synthetic/data/:")
    for name, item in manifest.items():
        print(f"  - {name}: expected top-1 = {item['ground_truth']['expected_top_stage']}, warning = {item['ground_truth']['expected_warning']}")


if __name__ == "__main__":
    main()
