"""Bottleneck Analyzer and Consistency Validator for LLM Serving Stacks (Member 4 - Sprint 2).

Ranks the top 3 bottlenecks, computes explained latency share, calculates
inference cost per 1M tokens, validates consistency between stage sums and total latency,
incorporates optional GPU and memory breakdown evidence, and provides domain-specific actions.

Supports three telemetry analysis modes:
1. Stage data only
2. Stage data + GPU metrics
3. Stage data + GPU metrics + memory breakdown

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

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cost_model import CostModel, load_gpu_price_from_config
from .actions import (
    DOMINANT_STAGE_SHARE,
    SUGGESTED_ACTIONS,
    build_evidence_notes,
    build_suggested_action,
)
from .readers import (
    PrometheusClient,
    PrometheusUnreachableError,
    join_by_request_id,
    read_memory_breakdown_json,
    read_request_csv,
    read_trace_jsonl,
)
from ..instrumentation.stage_timing import STAGE_NAMES, STAGE_NAME_SET

# Dominant stage threshold: if a stage accounts for >= 50% of latency, it is dominant.
DOMINANT_STAGE_THRESHOLD = DOMINANT_STAGE_SHARE

# Invariant bounds for explained_share: flag warning if outside [0.9, 1.1]
EXPLAINED_SHARE_MIN = 0.90
EXPLAINED_SHARE_MAX = 1.10

def parse_memory_breakdown(
    memory_breakdown_path: Optional[str | Path],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse optional memory_breakdown.json according to M3 component/phase layout.

    For backwards compatibility, delegates to read_memory_breakdown_json.
    """
    return read_memory_breakdown_json(memory_breakdown_path)


class BottleneckAnalyzer:
    """Analyzes request logs and per-stage traces to identify bottlenecks and calculate costs."""

    def __init__(
        self,
        gpu_price_per_hour_usd: Optional[float] = None,
        config_path: Optional[str | Path] = None,
        force_full_time_share: bool = False,
        prom_client: Optional[PrometheusClient] = None,
    ) -> None:
        self.config_path = str(config_path) if config_path else None
        self.gpu_price_per_hour_usd = (
            float(gpu_price_per_hour_usd)
            if gpu_price_per_hour_usd is not None
            else load_gpu_price_from_config(self.config_path)
        )
        self.force_full_time_share = force_full_time_share
        self.cost_model = CostModel(
            gpu_price_per_hour_usd=self.gpu_price_per_hour_usd,
            force_full_time_share=self.force_full_time_share,
        )
        self.prom_client = prom_client

    def analyze(
        self,
        csv_path: str | Path,
        traces_path: Optional[str | Path] = None,
        memory_breakdown_path: Optional[str | Path] = None,
        gpu_metrics: Optional[Dict[str, Any]] = None,
        stage_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        memory_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        max_bottlenecks: int = 3,
        force_full_time_share: Optional[bool] = None,
        batching_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Analyze request CSV and per-stage traces, returning contract JSON.

        Supports 3 telemetry modes:
        1. Stage data only
        2. Stage data + GPU metrics
        3. Stage data + GPU metrics + memory breakdown

        Parameters:
            csv_path: Path to request log CSV.
            traces_path: Optional path to per-stage trace JSON lines file.
            memory_breakdown_path: Optional path to M3 memory_breakdown.json.
            gpu_metrics: Optional dictionary containing GPU telemetry gauges.
            stage_metrics: Optional Prometheus mean/quantile latency summary.
            memory_metrics: Optional Prometheus memory gauges grouped by phase.
            max_bottlenecks: Number of top bottlenecks to rank (default 3).
            force_full_time_share: Optional override for time_share=1.0.
            batching_enabled: Whether batching was enabled for this measured run.

        Returns:
            Dictionary strictly adhering to the contract JSON schema.
        """
        # 1. Parse CSV requests via robust reader
        all_csv_records = read_request_csv(csv_path)
        csv_requests = [
            r for r in all_csv_records
            if str(r.get("status", "ok")).strip() == "ok" or str(r.get("status", "")).startswith("ok")
        ]

        # Prefer measured Prometheus means when supplied; otherwise use trace records.
        has_stage_metrics = bool(stage_metrics) and any(
            stage in STAGE_NAME_SET
            and stage != "total"
            and math.isfinite(float(values.get("mean", 0.0)))
            and float(values.get("mean", 0.0)) > 0
            for stage, values in (stage_metrics or {}).items()
        )
        joined_requests: Dict[str, Dict[str, Any]] = {}
        missing_traces_file = not has_stage_metrics
        if traces_path and os.path.exists(traces_path):
            valid_traces, _ = read_trace_jsonl(traces_path)
            joined_requests = join_by_request_id(csv_requests, valid_traces)
            if not has_stage_metrics and valid_traces:
                missing_traces_file = False

        # 3. Parse optional memory breakdown via robust reader
        memory_data, memory_status = read_memory_breakdown_json(memory_breakdown_path)
        if memory_status != "valid" and memory_metrics:
            memory_data = {"phases": memory_metrics}
            memory_status = "valid"

        # 4. Resolve live GPU metrics if not explicitly passed
        active_gpu_metrics = gpu_metrics
        if active_gpu_metrics is None and self.prom_client is not None:
            try:
                active_gpu_metrics = self.prom_client.get_gpu_gauges()
            except PrometheusUnreachableError as exc:
                print(f"[WARN] {exc}; continuing without GPU metrics.", file=sys.stderr)
                active_gpu_metrics = None

        # 5. Calculate Cost using Sprint 1 interval-union model
        cost_summary = self.cost_model.evaluate_dataset(
            csv_requests,
            force_full_time_share=force_full_time_share,
        )
        cost_per_million = cost_summary["cost_per_million_tokens_usd"]

        # 6. Consistency Check: Validate stage times vs total latency
        total_csv_latency = sum(float(r.get("total_latency_s", 0.0)) for r in csv_requests)
        total_stage_time_sum = 0.0
        latency_denominator = total_csv_latency

        stage_totals: Dict[str, float] = {s: 0.0 for s in STAGE_NAMES if s != "total"}
        stage_values: Dict[str, List[float]] = {s: [] for s in STAGE_NAMES if s != "total"}

        requests_with_traces = 0
        if has_stage_metrics and stage_metrics is not None:
            for stage, values in stage_metrics.items():
                if stage not in stage_totals or stage == "total":
                    continue
                mean = float(values.get("mean", 0.0))
                if math.isfinite(mean) and mean > 0:
                    stage_totals[stage] = mean
                    stage_values[stage] = [mean]
                    total_stage_time_sum += mean
            total_mean = float(stage_metrics.get("total", {}).get("mean", 0.0))
            if math.isfinite(total_mean) and total_mean > 0:
                latency_denominator = total_mean
            elif csv_requests:
                latency_denominator = total_csv_latency / len(csv_requests)
        else:
            for r in csv_requests:
                req_id = str(r.get("request_id", "")).strip()
                joined = joined_requests.get(req_id)
                if not joined or not joined["traces"]:
                    continue
                requests_with_traces += 1
                for stage, duration in joined["stages"].items():
                    if stage not in stage_totals or stage == "total":
                        continue
                    dur = float(duration)
                    if not math.isfinite(dur) or dur < 0:
                        continue
                    stage_totals[stage] += dur
                    stage_values[stage].append(dur)
                    total_stage_time_sum += dur

        # 7. Compute Explained Share & Invariant Warning
        has_stage_data = has_stage_metrics or requests_with_traces > 0
        if latency_denominator > 0 and has_stage_data:
            explained_share = total_stage_time_sum / latency_denominator
        else:
            explained_share = 0.0

        # Check invariant bounds [0.90, 1.10]
        warning_msg: Optional[str] = None
        if has_stage_data and (explained_share < EXPLAINED_SHARE_MIN or explained_share > EXPLAINED_SHARE_MAX):
            warning_msg = (
                f"Warning: explained_share {explained_share:.4f} is outside nominal bounds "
                f"[{EXPLAINED_SHARE_MIN}, {EXPLAINED_SHARE_MAX}]. Stage timings do not sum to total request latency."
            )

        # 8. Format Evidence String across the 3 Telemetry Modes
        def build_evidence_string(
            stage_name: str,
            share: float,
            mean_dur: float,
            p95_dur: float,
            count: Optional[int],
        ) -> str:
            # Base stage duration evidence
            stage_metrics_for_evidence = (stage_metrics or {}).get(stage_name, {})
            if has_stage_metrics:
                p50 = stage_metrics_for_evidence.get("p50")
                quantile_evidence = f", p50: {p50:.4f}s" if p50 is not None else ""
                evidence = (
                    f"{stage_name} is {share:.1%} of mean latency "
                    f"(mean {mean_dur:.4f}s{quantile_evidence}, p95 {p95_dur:.4f}s; Prometheus histogram)"
                )
            else:
                evidence = (
                    f"{stage_name} is {share:.1%} of mean latency "
                    f"(mean {mean_dur:.4f}s, p95 {p95_dur:.4f}s across {count} requests)"
                )

            # GPU evidence snippet
            gpu_snippet = ""
            has_gpu = False
            if active_gpu_metrics and isinstance(active_gpu_metrics, dict):
                util = active_gpu_metrics.get("gpu_utilization_percent")
                power = active_gpu_metrics.get("gpu_power_watts")
                memory_used = active_gpu_metrics.get("gpu_memory_used_bytes")
                gpu_facts = []
                if util is not None:
                    gpu_facts.append(f"GPU utilization avg {float(util):.1f}%")
                if power is not None:
                    gpu_facts.append(f"GPU power avg {float(power):.1f}W")
                if memory_used is not None:
                    gpu_facts.append(
                        f"GPU memory avg {float(memory_used) / (1024 * 1024):.1f} MiB"
                    )
                if gpu_facts:
                    has_gpu = True
                    gpu_snippet = "; " + ", ".join(gpu_facts)

            # Missing or invalid memory is relevant only to measured memory phases.
            mem_snippet = ""
            if (
                stage_name in ("prefill", "decode")
                and memory_status != "valid"
                and memory_breakdown_path
                and os.path.exists(memory_breakdown_path)
            ):
                mem_snippet = f". [{memory_status}]"
            elif stage_name in ("prefill", "decode") and memory_status != "valid":
                mem_snippet = ". [memory evidence unavailable]"

            # Combine based on active mode
            if has_gpu and memory_status == "valid" and mem_snippet:
                # Mode 3: Stage + GPU + Memory
                evidence += f"{gpu_snippet}{mem_snippet}."
            elif has_gpu:
                # Mode 2: Stage + GPU
                evidence += f"{gpu_snippet}{mem_snippet}"
            else:
                # Mode 1: Stage only
                evidence += f"{mem_snippet}"

            if share >= DOMINANT_STAGE_THRESHOLD:
                evidence += " [DOMINANT BOTTLENECK >= 50%]"

            rule_evidence = build_evidence_notes(
                stage_name,
                share,
                gpu_metrics=active_gpu_metrics,
                memory_data=memory_data if memory_status == "valid" else None,
                batching_enabled=batching_enabled,
            )
            if rule_evidence:
                evidence += "; " + "; ".join(rule_evidence)

            return evidence

        # 9. Rank Bottlenecks
        top_bottlenecks: List[Dict[str, Any]] = []

        if missing_traces_file or not has_stage_data:
            top_bottlenecks.append({
                "rank": 1,
                "stage": "missing_stage_traces",
                "share_of_latency": 0.0,
                "evidence": (
                    f"Per-stage trace JSON lines data was not provided or missing for {len(csv_requests)} "
                    "requests."
                ),
                "suggested_action": (
                    "Enable per-stage tracing via M2 profiler_exporter (port 9101) or provide "
                    "per-stage trace JSONL logs to decompose latency."
                ),
            })
        else:
            stage_shares: List[Tuple[str, float]] = []
            for stage, total_dur in stage_totals.items():
                if latency_denominator > 0 and total_dur > 0:
                    share = total_dur / latency_denominator
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
                if has_stage_metrics:
                    metric_values = (stage_metrics or {}).get(stage, {})
                    p95_dur = float(metric_values.get("p95", p95_dur))
                    evidence_count: Optional[int] = None
                else:
                    evidence_count = count

                evidence = build_evidence_string(stage, share, mean_dur, p95_dur, evidence_count)

                action = build_suggested_action(
                    stage,
                    share,
                    gpu_metrics=active_gpu_metrics,
                    memory_data=memory_data if memory_status == "valid" else None,
                    batching_enabled=batching_enabled,
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


def main(argv: Optional[List[str]] = None) -> None:
    """Keep the original analyzer wrapper compatible with the new CLI."""
    from .cli import main as cli_main

    cli_main(argv)
