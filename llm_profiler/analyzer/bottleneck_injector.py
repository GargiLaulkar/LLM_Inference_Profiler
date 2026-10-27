"""Bottleneck Injection and Synthetic Workload Generator for LLM Inference Profiler (Member 4).

Allows controlled injection of latency anomalies into specific serving stages
(queue_wait, prefill, decode, tokenization, detokenization, network) to validate
that the BottleneckAnalyzer accurately identifies and ranks the injected bottleneck.

Contracts strictly honored:
- Stage names: tokenization, queue_wait, prefill, decode, detokenization, network, plus total.
- Request log CSV columns: request_id, send_ts, end_ts, total_latency_s,
  prompt_tokens, output_tokens, status, target_rps, scenario_tag
- Per-stage trace JSON lines fields: request_id, stage, start_ts, end_ts, duration_s
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..instrumentation.stage_timing import STAGE_NAMES, validate_stage_name


class BottleneckInjector:
    """Injects synthetic latency spikes into designated stages to test analyzer ranking."""

    def __init__(self, seed: Optional[int] = 42) -> None:
        if seed is not None:
            random.seed(seed)

    def inject_into_existing_dataset(
        self,
        csv_rows: List[Dict[str, Any]],
        trace_records: List[Dict[str, Any]],
        target_stage: str,
        delay_seconds: float,
        fraction_of_requests: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Inject additional latency into a target stage across existing traces and CSV rows.

        Parameters:
            csv_rows: List of request CSV row dictionaries.
            trace_records: List of per-stage trace dictionaries.
            target_stage: Canonical stage name to inject delay into.
            delay_seconds: Latency added to the stage in seconds.
            fraction_of_requests: Fraction of requests to inject (0.0 to 1.0).

        Returns:
            Tuple of (modified_csv_rows, modified_trace_records).
        """
        target_stage = validate_stage_name(target_stage)
        if delay_seconds < 0:
            raise ValueError(f"delay_seconds cannot be negative: {delay_seconds}")
        if not (0.0 <= fraction_of_requests <= 1.0):
            raise ValueError("fraction_of_requests must be between 0.0 and 1.0")

        # Determine which request IDs will receive the injection
        all_req_ids = [str(r.get("request_id", "")).strip() for r in csv_rows if r.get("request_id")]
        sample_size = int(round(len(all_req_ids) * fraction_of_requests))
        injected_req_ids = set(random.sample(all_req_ids, sample_size)) if sample_size > 0 else set()

        # Update trace records
        modified_traces: List[Dict[str, Any]] = []
        delays_per_req: Dict[str, float] = {}

        for trace in trace_records:
            cloned = dict(trace)
            req_id = str(cloned.get("request_id", "")).strip()
            stage = str(cloned.get("stage", "")).strip()

            if req_id in injected_req_ids and stage == target_stage:
                cur_dur = float(cloned.get("duration_s", 0.0))
                new_dur = cur_dur + delay_seconds
                cloned["duration_s"] = round(new_dur, 6)
                if "end_ts" in cloned and "start_ts" in cloned:
                    cloned["end_ts"] = round(float(cloned["start_ts"]) + new_dur, 6)
                delays_per_req[req_id] = delays_per_req.get(req_id, 0.0) + delay_seconds

            modified_traces.append(cloned)

        # Update CSV rows to maintain mathematical consistency
        modified_csv: List[Dict[str, Any]] = []
        for row in csv_rows:
            cloned_row = dict(row)
            req_id = str(cloned_row.get("request_id", "")).strip()
            if req_id in delays_per_req:
                injected_amt = delays_per_req[req_id]
                cur_lat = float(cloned_row.get("total_latency_s", 0.0))
                new_lat = cur_lat + injected_amt
                cloned_row["total_latency_s"] = round(new_lat, 6)
                if "end_ts" in cloned_row and "send_ts" in cloned_row:
                    cloned_row["end_ts"] = round(float(cloned_row["send_ts"]) + new_lat, 6)
            modified_csv.append(cloned_row)

        return modified_csv, modified_traces

    def generate_synthetic_workload(
        self,
        num_requests: int = 50,
        scenario_tag: str = "synthetic_baseline",
        target_rps: float = 5.0,
        prompt_tokens: int = 128,
        output_tokens: int = 64,
        injected_stage: Optional[str] = None,
        injection_delay_s: float = 0.0,
        base_timestamp: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate synthetic request CSV rows and per-stage traces strictly matching contracts.

        Parameters:
            num_requests: Number of requests to generate.
            scenario_tag: Tag for the experiment scenario.
            target_rps: Request arrival rate.
            prompt_tokens: Number of prompt tokens per request.
            output_tokens: Number of generated tokens per request.
            injected_stage: Optional stage to inject an anomaly into.
            injection_delay_s: Extra delay to add to the injected stage.
            base_timestamp: Starting epoch timestamp.

        Returns:
            Tuple of (csv_rows, trace_records).
        """
        if injected_stage:
            injected_stage = validate_stage_name(injected_stage)

        start_time = base_timestamp if base_timestamp is not None else time.time()
        csv_rows: List[Dict[str, Any]] = []
        trace_records: List[Dict[str, Any]] = []

        cur_send_ts = start_time
        inter_arrival = 1.0 / target_rps if target_rps > 0 else 0.2

        for i in range(1, num_requests + 1):
            req_id = f"synth-{scenario_tag}-{i:04d}"

            # Realistic baseline timings (in seconds):
            # Tokenization: ~0.001 - 0.002s
            t_tok = max(0.0005, random.gauss(0.0012, 0.0002))
            # Queue wait: depends on target_rps
            queue_mean = 0.004 + (max(0.0, target_rps - 2.0) * 0.003)
            t_queue = max(0.001, random.gauss(queue_mean, 0.001))
            # Prefill: ~0.020s for 128 tokens (~0.00015s per prompt token)
            prefill_mean = prompt_tokens * 0.00018
            t_prefill = max(0.005, random.gauss(prefill_mean, prefill_mean * 0.08))
            # Decode: ~0.0012s per output token
            decode_mean = output_tokens * 0.00125
            t_decode = max(0.008, random.gauss(decode_mean, decode_mean * 0.05))
            # Detokenization: ~0.0008s
            t_detok = max(0.0003, random.gauss(0.0008, 0.0001))
            # Network round-trip: ~0.0025s
            t_net = max(0.001, random.gauss(0.0025, 0.0004))

            # Apply injected delay if specified
            if injected_stage == "tokenization":
                t_tok += injection_delay_s
            elif injected_stage == "queue_wait":
                t_queue += injection_delay_s
            elif injected_stage == "prefill":
                t_prefill += injection_delay_s
            elif injected_stage == "decode":
                t_decode += injection_delay_s
            elif injected_stage == "detokenization":
                t_detok += injection_delay_s
            elif injected_stage == "network":
                t_net += injection_delay_s

            stages = [
                ("network", t_net / 2.0),
                ("tokenization", t_tok),
                ("queue_wait", t_queue),
                ("prefill", t_prefill),
                ("decode", t_decode),
                ("detokenization", t_detok),
                ("network", t_net / 2.0),
            ]

            # Accumulate chronological stage traces
            stage_cursor = cur_send_ts
            stage_durations: Dict[str, float] = {}

            for s_name, dur in stages:
                s_start = stage_cursor
                s_end = stage_cursor + dur
                stage_cursor = s_end

                trace_records.append({
                    "request_id": req_id,
                    "stage": s_name,
                    "start_ts": round(s_start, 6),
                    "end_ts": round(s_end, 6),
                    "duration_s": round(dur, 6),
                })
                stage_durations[s_name] = stage_durations.get(s_name, 0.0) + dur

            total_lat = stage_cursor - cur_send_ts
            end_ts = cur_send_ts + total_lat

            # Write request CSV row matching exact contract columns
            csv_rows.append({
                "request_id": req_id,
                "send_ts": f"{cur_send_ts:.6f}",
                "end_ts": f"{end_ts:.6f}",
                "total_latency_s": f"{total_lat:.6f}",
                "prompt_tokens": str(prompt_tokens),
                "output_tokens": str(output_tokens),
                "status": "ok",
                "target_rps": f"{target_rps:.1f}",
                "scenario_tag": scenario_tag,
            })

            cur_send_ts += inter_arrival

        return csv_rows, trace_records

    def save_workload(
        self,
        csv_rows: List[Dict[str, Any]],
        trace_records: List[Dict[str, Any]],
        output_dir: str,
        prefix: str = "workload",
    ) -> Tuple[str, str]:
        """Save generated CSV rows and trace JSON lines to disk.

        Returns:
            Tuple of (csv_path, traces_path).
        """
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{prefix}_requests.csv")
        traces_path = os.path.join(output_dir, f"{prefix}_traces.jsonl")

        # Save CSV
        headers = [
            "request_id", "send_ts", "end_ts", "total_latency_s",
            "prompt_tokens", "output_tokens", "status", "target_rps", "scenario_tag"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

        # Save Traces JSON lines
        with open(traces_path, "w", encoding="utf-8") as f:
            for rec in trace_records:
                f.write(json.dumps(rec) + "\n")

        return csv_path, traces_path


def main() -> None:
    """CLI entry point for Bottleneck Injector."""
    parser = argparse.ArgumentParser(description="Synthetic workload generator and bottleneck injector.")
    parser.add_argument("--stage", choices=STAGE_NAMES, default=None, help="Stage to inject delay into")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds to inject")
    parser.add_argument("--num-requests", type=int, default=50, help="Number of requests")
    parser.add_argument("--rps", type=float, default=5.0, help="Target RPS")
    parser.add_argument("--prompt-tokens", type=int, default=128, help="Prompt tokens")
    parser.add_argument("--output-tokens", type=int, default=64, help="Output tokens")
    parser.add_argument("--out-dir", default="results/injected", help="Directory to save generated files")
    parser.add_argument("--prefix", default="injected_run", help="File prefix")
    args = parser.parse_args()

    injector = BottleneckInjector()
    csv_rows, traces = injector.generate_synthetic_workload(
        num_requests=args.num_requests,
        scenario_tag=f"inject_{args.stage or 'none'}",
        target_rps=args.rps,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        injected_stage=args.stage,
        injection_delay_s=args.delay,
    )

    csv_path, traces_path = injector.save_workload(
        csv_rows, traces, output_dir=args.out_dir, prefix=args.prefix
    )

    print(f"[SUCCESS] Synthetic workload saved:")
    print(f"  CSV:    {csv_path} ({len(csv_rows)} rows)")
    print(f"  Traces: {traces_path} ({len(traces)} stage entries)")


if __name__ == "__main__":
    main()
