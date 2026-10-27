"""Bottleneck Analyzer and Consistency Validator for LLM Serving Stacks (Member 4).

Ranks the top 3 bottlenecks, computes explained latency share, calculates
inference cost per 1M tokens, validates consistency between stage sums and total latency,
incorporates optional GPU and memory breakdown evidence, and provides domain-specific actions.

Contracts strictly honored:
- Stage names: tokenization, queue_wait, prefill, decode, detokenization, network, plus total.
- Request log CSV columns: request_id, send_ts, end_ts, total_latency_s,
  prompt_tokens, output_tokens, status, target_rps, scenario_tag
- Per-stage trace JSON lines fields: request_id, stage, start_ts, end_ts, duration_s
- Analyzer output JSON schema:
  {
    "top_bottlenecks": [
      {
        "rank": 1,
        "stage": "...",
        "share_of_latency": 0.0,
        "evidence": "...",
        "suggested_action": "..."
      }
    ],
    "cost_per_million_tokens_usd": 0.0,
    "explained_share": 0.0
  }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from .cost_model import CostModel, load_gpu_price_from_config
from ..instrumentation.stage_timing import STAGE_NAMES, STAGE_NAME_SET

# Dominant stage threshold: if a stage accounts for >= 50% of latency, it is dominant.
DOMINANT_STAGE_THRESHOLD = 0.50

# Invariant bounds for explained_share: flag warning if outside [0.9, 1.1]
EXPLAINED_SHARE_MIN = 0.90
EXPLAINED_SHARE_MAX = 1.10

SUGGESTED_ACTIONS: Dict[str, str] = {
    "prefill": (
        "Enable chunked prefill (--enable-chunked-prefill), utilize prompt prefix caching "
        "(--enable-prefix-caching), optimize prompt token length, or scale tensor parallelism."
    ),
    "decode": (
        "Enable speculative decoding, optimize KV cache block size, raise tensor parallelism, "
        "or restrict generation max_tokens to prevent autoregressive memory bandwidth saturation."
    ),
    "queue_wait": (
        "Increase serving concurrency (--max-num-seqs), scale out server replicas behind a "
        "load balancer, optimize KV cache memory allocation, or apply client rate limiting."
    ),
    "tokenization": (
        "Switch to a fast Rust-based HuggingFace tokenizer, cache tokenized prompts on client/worker, "
        "or offload tokenization off the serving critical path."
    ),
    "detokenization": (
        "Stream incremental tokens directly to client, batch UTF-8 detokenization passes, "
        "or offload detokenization to client."
    ),
    "network": (
        "Colocate client and inference server within the same VPC/subnet, enable persistent HTTP "
        "keep-alive connection pooling, or migrate to gRPC / HTTP/2 transport."
    ),
    "total": (
        "Investigate cumulative end-to-end stack latency across all compute and communication stages."
    ),
}

CONTRACT_COMPONENTS = ["weights", "kv_cache", "activations", "other"]
CONTRACT_PHASES = ["prefill", "decode"]


def parse_memory_breakdown(
    memory_breakdown_path: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse optional memory_breakdown.json according to M3 component/phase layout.

    Returns:
        Tuple of (parsed_data, status_or_evidence_string).
    """
    if not memory_breakdown_path or not os.path.exists(memory_breakdown_path):
        return None, "memory evidence unavailable"

    try:
        with open(memory_breakdown_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return None, f"memory breakdown parse error: {exc}"

    if not isinstance(data, dict):
        return None, "memory_breakdown schema mismatch: expected top-level JSON object"

    phases = data.get("phases")
    if not isinstance(phases, dict):
        return None, "memory_breakdown schema mismatch: missing 'phases' dictionary"

    # Validate phases
    missing_phases = [p for p in CONTRACT_PHASES if p not in phases]
    if missing_phases:
        return None, f"memory_breakdown schema mismatch: missing phases {missing_phases}"

    # Validate components in each phase
    for phase_name in CONTRACT_PHASES:
        phase_dict = phases[phase_name]
        if not isinstance(phase_dict, dict):
            return None, f"memory_breakdown schema mismatch: phase '{phase_name}' must be a dictionary"
        missing_comp = [c for c in CONTRACT_COMPONENTS if c not in phase_dict]
        if missing_comp:
            return None, f"memory_breakdown schema mismatch: phase '{phase_name}' missing components {missing_comp}"

    return data, "valid"


class BottleneckAnalyzer:
    """Analyzes request logs and per-stage traces to identify bottlenecks and calculate costs."""

    def __init__(
        self,
        gpu_price_per_hour_usd: Optional[float] = None,
        config_path: Optional[str] = None,
        force_full_time_share: bool = False,
    ) -> None:
        self.config_path = config_path
        self.gpu_price_per_hour_usd = (
            float(gpu_price_per_hour_usd)
            if gpu_price_per_hour_usd is not None
            else load_gpu_price_from_config(config_path)
        )
        self.force_full_time_share = force_full_time_share
        self.cost_model = CostModel(
            gpu_price_per_hour_usd=self.gpu_price_per_hour_usd,
            force_full_time_share=self.force_full_time_share,
        )

    def analyze(
        self,
        csv_path: str,
        traces_path: Optional[str] = None,
        memory_breakdown_path: Optional[str] = None,
        gpu_metrics: Optional[Dict[str, Any]] = None,
        max_bottlenecks: int = 3,
        force_full_time_share: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Analyze request CSV and per-stage traces, returning contract JSON.

        Parameters:
            csv_path: Path to request log CSV.
            traces_path: Optional path to per-stage trace JSON lines file.
            memory_breakdown_path: Optional path to M3 memory_breakdown.json.
            gpu_metrics: Optional dictionary containing live GPU telemetry snapshot.
            max_bottlenecks: Number of top bottlenecks to rank (default 3).
            force_full_time_share: Optional override for time_share=1.0.

        Returns:
            Dictionary strictly adhering to the contract JSON schema.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Request log CSV not found: {csv_path}")

        # 1. Parse CSV requests
        csv_requests: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = str(row.get("status", "ok")).strip()
                if status == "ok" or status.startswith("ok"):
                    csv_requests.append(row)

        # 2. Parse Traces if available
        traces_by_req: Dict[str, List[Dict[str, Any]]] = {}
        missing_traces_file = False

        if traces_path and os.path.exists(traces_path):
            with open(traces_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                        req_id = str(record.get("request_id", "")).strip()
                        if req_id:
                            traces_by_req.setdefault(req_id, []).append(record)
                    except json.JSONDecodeError as err:
                        print(f"[WARN] Invalid JSON line {line_idx} in {traces_path}: {err}", file=sys.stderr)
        else:
            missing_traces_file = True

        # 3. Parse optional memory breakdown
        memory_data, memory_status = parse_memory_breakdown(memory_breakdown_path)

        # 4. Calculate Cost using Task 1 interval-union model
        cost_summary = self.cost_model.evaluate_dataset(
            csv_requests,
            force_full_time_share=force_full_time_share,
        )
        cost_per_million = cost_summary["cost_per_million_tokens_usd"]

        # 5. Consistency Check: Validate stage times vs total latency
        total_csv_latency = sum(float(r.get("total_latency_s", 0.0)) for r in csv_requests)
        total_stage_time_sum = 0.0

        stage_totals: Dict[str, float] = {s: 0.0 for s in STAGE_NAMES if s != "total"}
        stage_counts: Dict[str, int] = {s: 0 for s in STAGE_NAMES if s != "total"}
        stage_values: Dict[str, List[float]] = {s: [] for s in STAGE_NAMES if s != "total"}

        requests_with_traces = 0
        requests_without_traces = 0

        for r in csv_requests:
            req_id = str(r.get("request_id", "")).strip()
            total_lat = float(r.get("total_latency_s", 0.0))

            if req_id in traces_by_req:
                requests_with_traces += 1
                req_stages = traces_by_req[req_id]
                req_stage_sum = 0.0

                for trace in req_stages:
                    stage = str(trace.get("stage", "")).strip()
                    dur = float(trace.get("duration_s", 0.0))
                    if dur < 0:
                        continue

                    # Omit "total" from stage breakdown sum to prevent double counting
                    if stage in stage_totals and stage != "total":
                        stage_totals[stage] += dur
                        stage_counts[stage] += 1
                        stage_values[stage].append(dur)
                        req_stage_sum += dur

                total_stage_time_sum += req_stage_sum
            else:
                requests_without_traces += 1

        # 6. Compute Explained Share & Invariant Warning
        if total_csv_latency > 0 and requests_with_traces > 0:
            explained_share = total_stage_time_sum / total_csv_latency
        else:
            explained_share = 0.0

        # Check invariant bounds [0.90, 1.10]
        warning_msg: Optional[str] = None
        if requests_with_traces > 0 and (explained_share < EXPLAINED_SHARE_MIN or explained_share > EXPLAINED_SHARE_MAX):
            warning_msg = (
                f"Warning: explained_share {explained_share:.4f} is outside nominal bounds "
                f"[{EXPLAINED_SHARE_MIN}, {EXPLAINED_SHARE_MAX}]. Stage timings do not sum to total request latency."
            )

        # 7. Format Memory & GPU Evidence Snippets
        def get_memory_evidence_for_stage(stage_name: str) -> str:
            if memory_status != "valid" or not memory_data:
                return memory_status

            phases = memory_data.get("phases", {})
            mib = 1024 * 1024
            if stage_name == "prefill":
                act_mib = phases.get("prefill", {}).get("activations", 0) / mib
                w_mib = phases.get("prefill", {}).get("weights", 0) / mib
                return f"memory: prefill activations={act_mib:.1f} MiB, weights={w_mib:.1f} MiB"
            elif stage_name == "decode":
                kv_mib = phases.get("decode", {}).get("kv_cache", 0) / mib
                act_mib = phases.get("decode", {}).get("activations", 0) / mib
                return f"memory: decode kv_cache={kv_mib:.1f} MiB, activations={act_mib:.1f} MiB"
            else:
                w_mib = phases.get("prefill", {}).get("weights", 0) / mib
                peak_kv = phases.get("decode", {}).get("kv_cache", 0) / mib
                return f"memory: static weights={w_mib:.1f} MiB, peak kv_cache={peak_kv:.1f} MiB"

        gpu_evidence_str = ""
        if gpu_metrics and isinstance(gpu_metrics, dict):
            util = gpu_metrics.get("gpu_utilization_percent")
            power = gpu_metrics.get("gpu_power_watts")
            if util is not None:
                gpu_evidence_str = f" GPU: {util:.1f}% util"
                if power is not None:
                    gpu_evidence_str += f", {power:.1f}W"
                gpu_evidence_str += "."

        # 8. Rank Bottlenecks
        top_bottlenecks: List[Dict[str, Any]] = []

        if missing_traces_file or requests_with_traces == 0:
            top_bottlenecks.append({
                "rank": 1,
                "stage": "missing_stage_traces",
                "share_of_latency": 0.0,
                "evidence": (
                    f"Per-stage trace JSON lines data was not provided or missing for {len(csv_requests)} "
                    f"requests (memory evidence unavailable)."
                ),
                "suggested_action": (
                    "Enable per-stage tracing via M2 profiler_exporter (port 9101) or provide "
                    "per-stage trace JSONL logs to decompose latency."
                ),
            })
        else:
            stage_shares: List[Tuple[str, float]] = []
            for stage, total_dur in stage_totals.items():
                if total_csv_latency > 0 and total_dur > 0:
                    share = total_dur / total_csv_latency
                    stage_shares.append((stage, share))

            # Rank by largest share of latency first
            stage_shares.sort(key=lambda x: x[1], reverse=True)

            rank = 1
            for stage, share in stage_shares[:max_bottlenecks]:
                vals = stage_values[stage]
                count = len(vals)
                mean_dur = sum(vals) / count if count > 0 else 0.0
                sorted_v = sorted(vals)
                p95_idx = max(0, min(int(math.ceil(0.95 * count)) - 1, count - 1)) if count > 0 else 0
                p95_dur = sorted_v[p95_idx] if count > 0 else 0.0

                stage_mem_ev = get_memory_evidence_for_stage(stage)
                evidence = (
                    f"'{stage}' accounts for {share:.1%} of total latency "
                    f"(mean: {mean_dur:.4f}s, p95: {p95_dur:.4f}s across {count} requests). "
                    f"[{stage_mem_ev}]{gpu_evidence_str}"
                )

                if share >= DOMINANT_STAGE_THRESHOLD:
                    evidence += " [DOMINANT BOTTLENECK >= 50%]"

                action = SUGGESTED_ACTIONS.get(
                    stage,
                    "Profile server kernel traces and adjust hardware resource allocation.",
                )

                top_bottlenecks.append({
                    "rank": rank,
                    "stage": stage,
                    "share_of_latency": round(share, 4),
                    "evidence": evidence,
                    "suggested_action": action,
                })
                rank += 1

            if warning_msg and top_bottlenecks:
                top_bottlenecks[0]["evidence"] += f" [{warning_msg}]"

        output: Dict[str, Any] = {
            "top_bottlenecks": top_bottlenecks,
            "cost_per_million_tokens_usd": round(cost_per_million, 6),
            "explained_share": round(explained_share, 4),
        }

        if warning_msg:
            output["warning"] = warning_msg

        return output


def main() -> None:
    """CLI entry point for Bottleneck Analyzer."""
    parser = argparse.ArgumentParser(description="Analyze LLM serving latency bottlenecks and cost.")
    parser.add_argument("--csv", required=True, help="Path to request log CSV file")
    parser.add_argument("--traces", default=None, help="Path to per-stage trace JSON lines file")
    parser.add_argument("--memory-breakdown", default=None, help="Path to optional memory_breakdown.json")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--gpu-price", type=float, default=None, help="Override GPU price per hour USD")
    parser.add_argument("--full-time-share", action="store_true", help="Force time_share=1.0")
    parser.add_argument("--out", default=None, help="Path to output JSON file (defaults to stdout)")
    args = parser.parse_args()

    analyzer = BottleneckAnalyzer(
        gpu_price_per_hour_usd=args.gpu_price,
        config_path=args.config,
        force_full_time_share=args.full_time_share,
    )

    try:
        result = analyzer.analyze(
            csv_path=args.csv,
            traces_path=args.traces,
            memory_breakdown_path=args.memory_breakdown,
        )
    except Exception as exc:
        print(f"[ERROR] Bottleneck analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)

    formatted_json = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(formatted_json + "\n")
        print(f"[INFO] Analysis saved to {args.out}")
    else:
        print(formatted_json)


if __name__ == "__main__":
    main()
