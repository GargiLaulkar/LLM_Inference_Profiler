"""Inference Cost Model for LLM Serving Stacks (Member 4).

Computes amortized dollar cost per 1M tokens based on GPU operating rate,
active serving duration (via interval union to prevent double-counting overlapping requests),
and total tokens produced.

Shared contract & formulas:
- cost_per_million_tokens_usd = (gpu_price_per_hour_usd * gpu_hours_used) / tokens_produced * 1_000_000
- gpu_hours_used = (wall_clock_window_seconds / 3600.0) * time_share
- wall_clock_window_seconds = max(end_ts) - min(send_ts) over successful requests
- time_share = active_in_flight_seconds / wall_clock_window_seconds (union of intervals)
- tokens_produced = sum(output_tokens) over successful requests
- cost_per_million_total_tokens_usd = optional metric dividing by (prompt_tokens + output_tokens)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml


def load_gpu_price_from_config(config_path: Optional[str] = None) -> float:
    """Load gpu_price_per_hour_usd from config.yaml without hardcoding.

    Parameters:
        config_path: Optional explicit path to configuration YAML.

    Returns:
        float: GPU price per hour in USD.
    """
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Cost configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "gpu_price_per_hour_usd" not in cfg:
        raise ValueError(f"{path} must define gpu_price_per_hour_usd")
    try:
        parsed = float(cfg["gpu_price_per_hour_usd"])
    except (TypeError, ValueError) as exc:
        raise ValueError("gpu_price_per_hour_usd must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("gpu_price_per_hour_usd must be a finite non-negative number")
    return parsed


def compute_interval_union_seconds(intervals: Sequence[Tuple[float, float]]) -> float:
    """Compute the total duration of the union of intervals.

    Prevents double-counting time when multiple requests are in flight concurrently.

    Parameters:
        intervals: Sequence of (start_ts, end_ts) tuples.

    Returns:
        float: Total active seconds with at least one request in flight.
    """
    valid = []
    for s, e in intervals:
        start_f = float(s)
        end_f = float(e)
        if end_f >= start_f and math.isfinite(start_f) and math.isfinite(end_f):
            valid.append((start_f, end_f))

    if not valid:
        return 0.0

    valid.sort(key=lambda x: (x[0], x[1]))
    merged: List[List[float]] = []

    for s, e in valid:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return sum(e - s for s, e in merged)


def calculate_request_cost(
    total_latency_s: float,
    output_tokens: int,
    gpu_price_per_hour_usd: Optional[float] = None,
    prompt_tokens: int = 0,
) -> float:
    """Calculate the cost in USD per 1M tokens for a single inference request.

    Formula:
        Cost per 1M Tokens = ((gpu_price_per_hour_usd * (total_latency_s / 3600))
                              / tokens_produced) * 1,000,000

    Parameters:
        total_latency_s: Request duration in seconds.
        output_tokens: Generated output tokens (tokens_produced).
        gpu_price_per_hour_usd: GPU operational cost per hour in USD.
        prompt_tokens: Optional prompt token count for backwards compatibility.

    Returns:
        float: Cost in USD per 1M output tokens. Returns 0.0 if output_tokens is zero.
    """
    if isinstance(total_latency_s, bool) or not isinstance(total_latency_s, (int, float)):
        raise TypeError("total_latency_s must be a real number")
    if not math.isfinite(total_latency_s):
        raise ValueError("total_latency_s must be finite")
    if total_latency_s < 0:
        raise ValueError(f"total_latency_s cannot be negative: {total_latency_s}")

    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
        raise TypeError("output_tokens must be an integer")
    if output_tokens < 0:
        raise ValueError(f"output_tokens cannot be negative: {output_tokens}")

    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise TypeError("prompt_tokens must be an integer")
    if prompt_tokens < 0:
        raise ValueError(f"prompt_tokens cannot be negative: {prompt_tokens}")

    price = (
        load_gpu_price_from_config()
        if gpu_price_per_hour_usd is None
        else float(gpu_price_per_hour_usd)
    )
    if not math.isfinite(price) or price < 0:
        raise ValueError("gpu_price_per_hour_usd must be a finite non-negative number")

    if output_tokens == 0:
        return 0.0

    gpu_hours = total_latency_s / 3600.0
    request_dollar_cost = price * gpu_hours
    cost_per_million = (request_dollar_cost / output_tokens) * 1_000_000.0
    return float(cost_per_million)


class CostModel:
    """Inference cost evaluator and Prometheus metric generator."""

    def __init__(
        self,
        gpu_price_per_hour_usd: Optional[float] = None,
        config_path: Optional[str] = None,
        force_full_time_share: bool = False,
    ) -> None:
        """Initialize the cost model.

        Parameters:
            gpu_price_per_hour_usd: Operational rate in USD/hour.
            config_path: Optional path to config.yaml.
            force_full_time_share: When True, assumes dedicated GPU reservation
                                   (time_share=1.0) regardless of idle gaps.
        """
        if gpu_price_per_hour_usd is not None:
            price = float(gpu_price_per_hour_usd)
            if not math.isfinite(price) or price < 0:
                raise ValueError("gpu_price_per_hour_usd must be a finite non-negative number")
            self.gpu_price_per_hour_usd = price
        else:
            self.gpu_price_per_hour_usd = load_gpu_price_from_config(config_path)

        self.force_full_time_share = force_full_time_share
        self._latest_cost_per_million: float = 0.0

    def evaluate_request(
        self,
        total_latency_s: float,
        output_tokens: int,
        prompt_tokens: int = 0,
    ) -> float:
        """Evaluate and record cost per 1M tokens for a single request."""
        cost = calculate_request_cost(
            total_latency_s=total_latency_s,
            output_tokens=output_tokens,
            gpu_price_per_hour_usd=self.gpu_price_per_hour_usd,
            prompt_tokens=prompt_tokens,
        )
        self._latest_cost_per_million = cost
        return cost

    def evaluate_dataset(
        self,
        records: Sequence[Dict[str, Any]],
        force_full_time_share: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate and statistical cost metrics across a batch of requests.

        Formula:
            cost_per_million_tokens_usd = (gpu_price_per_hour_usd * gpu_hours_used)
                                          / tokens_produced * 1_000_000
            gpu_hours_used = (wall_clock_window_seconds / 3600) * time_share
            wall_clock_window = max(end_ts) - min(send_ts) over successful requests.
            time_share = fraction of window with >=1 request in flight (union of intervals).

        Parameters:
            records: List of request dictionaries containing send_ts, end_ts,
                     total_latency_s, prompt_tokens, output_tokens, status.
            force_full_time_share: Override flag for dedicated GPU billing (time_share=1.0).

        Returns:
            Dictionary of aggregated metrics and distribution percentiles.
        """
        use_full_share = self.force_full_time_share if force_full_time_share is None else force_full_time_share

        valid_intervals: List[Tuple[float, float]] = []
        tokens_produced = 0
        total_prompt_tokens = 0
        per_request_costs: List[float] = []
        successful_requests = 0

        min_send = float("inf")
        max_end = float("-inf")

        for r in records:
            # Failed requests excluded
            status = str(r.get("status", "ok")).strip()
            if status != "ok" and not status.startswith("ok"):
                continue

            try:
                lat = float(r.get("total_latency_s", 0.0))
                p_tok = int(r.get("prompt_tokens", 0))
                o_tok = int(r.get("output_tokens", 0))

                # Extract send_ts and end_ts
                send_ts_val = r.get("send_ts")
                end_ts_val = r.get("end_ts")

                if send_ts_val is not None and end_ts_val is not None:
                    s_ts = float(send_ts_val)
                    e_ts = float(end_ts_val)
                else:
                    # Fallback if only latency is present
                    s_ts = 0.0
                    e_ts = lat

                if lat < 0 or p_tok < 0 or o_tok < 0 or e_ts < s_ts:
                    continue

            except (ValueError, TypeError):
                continue

            successful_requests += 1
            min_send = min(min_send, s_ts)
            max_end = max(max_end, e_ts)
            valid_intervals.append((s_ts, e_ts))
            tokens_produced += o_tok
            total_prompt_tokens += p_tok

            if o_tok > 0:
                cost_single = calculate_request_cost(
                    total_latency_s=lat,
                    output_tokens=o_tok,
                    gpu_price_per_hour_usd=self.gpu_price_per_hour_usd,
                    prompt_tokens=p_tok,
                )
                per_request_costs.append(cost_single)

        if successful_requests == 0:
            return {
                "cost_per_million_tokens_usd": 0.0,
                "cost_per_million_total_tokens_usd": 0.0,
                "tokens_produced": 0,
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "wall_clock_window_seconds": 0.0,
                "active_seconds": 0.0,
                "time_share": 0.0,
                "gpu_hours_used": 0.0,
                "request_count": 0,
                "gpu_price_per_hour_usd": self.gpu_price_per_hour_usd,
                "p50_cost_usd": 0.0,
                "p95_cost_usd": 0.0,
                "p99_cost_usd": 0.0,
                "mean_cost_usd": 0.0,
                "min_cost_usd": 0.0,
                "max_cost_usd": 0.0,
            }

        wall_clock_window_seconds = max(0.0, max_end - min_send)
        active_seconds = compute_interval_union_seconds(valid_intervals)

        if wall_clock_window_seconds > 0:
            computed_time_share = min(1.0, max(0.0, active_seconds / wall_clock_window_seconds))
        else:
            computed_time_share = 0.0

        if use_full_share:
            time_share = 1.0
        else:
            time_share = computed_time_share

        gpu_hours_used = (wall_clock_window_seconds / 3600.0) * time_share

        if tokens_produced > 0:
            aggregate_cost = (
                (self.gpu_price_per_hour_usd * gpu_hours_used) / tokens_produced
            ) * 1_000_000.0
        else:
            aggregate_cost = 0.0

        total_tokens = total_prompt_tokens + tokens_produced
        if total_tokens > 0:
            cost_per_million_total = (
                (self.gpu_price_per_hour_usd * gpu_hours_used) / total_tokens
            ) * 1_000_000.0
        else:
            cost_per_million_total = 0.0

        self._latest_cost_per_million = aggregate_cost

        sorted_costs = sorted(per_request_costs)
        n = len(sorted_costs)

        def percentile(p: float) -> float:
            if n == 0:
                return 0.0
            idx = int(math.ceil(p * n)) - 1
            idx = max(0, min(idx, n - 1))
            return sorted_costs[idx]

        summary = {
            "cost_per_million_tokens_usd": round(aggregate_cost, 6),
            "cost_per_million_total_tokens_usd": round(cost_per_million_total, 6),
            "tokens_produced": tokens_produced,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "wall_clock_window_seconds": round(wall_clock_window_seconds, 6),
            "active_seconds": round(active_seconds, 6),
            "time_share": round(time_share, 6),
            "gpu_hours_used": round(gpu_hours_used, 8),
            "request_count": successful_requests,
            "gpu_price_per_hour_usd": self.gpu_price_per_hour_usd,
            "p50_cost_usd": round(percentile(0.50), 6),
            "p95_cost_usd": round(percentile(0.95), 6),
            "p99_cost_usd": round(percentile(0.99), 6),
            "mean_cost_usd": round(sum(per_request_costs) / n, 6) if n > 0 else 0.0,
            "min_cost_usd": round(min(per_request_costs), 6) if n > 0 else 0.0,
            "max_cost_usd": round(max(per_request_costs), 6) if n > 0 else 0.0,
        }
        return summary

    def render_prometheus_metrics(
        self,
        cost_per_million_usd: Optional[float] = None,
    ) -> str:
        """Render cost metric in strict Prometheus text exposition format.

        Contract metric:
            llmprof_cost_per_million_tokens_usd (Gauge, no labels)
        """
        val = cost_per_million_usd if cost_per_million_usd is not None else self._latest_cost_per_million
        if not math.isfinite(float(val)) or float(val) < 0:
            raise ValueError("Cost gauge value must be finite and non-negative")
        lines = [
            "# HELP llmprof_cost_per_million_tokens_usd Amortized inference dollar cost per 1M tokens.",
            "# TYPE llmprof_cost_per_million_tokens_usd gauge",
            f"llmprof_cost_per_million_tokens_usd {float(val):.6f}",
        ]
        return "\n".join(lines) + "\n"


def calculate_csv_cost(
    csv_path: str,
    gpu_price_per_hour_usd: Optional[float] = None,
    config_path: Optional[str] = None,
    force_full_time_share: bool = False,
) -> Dict[str, Any]:
    """Load a request log CSV file and compute the aggregate inference cost.

    Parameters:
        csv_path: Path to request log CSV.
        gpu_price_per_hour_usd: Optional explicit GPU price override.
        config_path: Optional config.yaml path.
        force_full_time_share: Allow time_share=1.0 via flag.

    Returns:
        Dictionary of cost summary metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    records: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    model = CostModel(
        gpu_price_per_hour_usd=gpu_price_per_hour_usd,
        config_path=config_path,
        force_full_time_share=force_full_time_share,
    )
    return model.evaluate_dataset(records)


class StandaloneCostExporter:
    """Lightweight HTTP server exporting llmprof_cost_per_million_tokens_usd."""

    def __init__(
        self,
        cost_model: CostModel,
        host: str = "127.0.0.1",
        port: int = 9101,
    ) -> None:
        self.cost_model = cost_model
        self.host = host
        self.port = int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "StandaloneCostExporter":
        if self._server is not None:
            return self

        cost_model = self.cost_model

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = cost_model.render_prometheus_metrics().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None

    def __enter__(self) -> "StandaloneCostExporter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


def main() -> None:
    """CLI entry point for evaluating cost on request CSVs."""
    parser = argparse.ArgumentParser(description="Evaluate inference cost per 1M tokens from request logs.")
    parser.add_argument("--csv", required=True, help="Path to request log CSV file")
    parser.add_argument("--config", default=None, help="Path to config.yaml (reads gpu_price_per_hour_usd)")
    parser.add_argument("--gpu-price", type=float, default=None, help="Override GPU price per hour in USD")
    parser.add_argument("--full-time-share", action="store_true", help="Force time_share=1.0 (dedicated instance)")
    parser.add_argument("--export-metrics", action="store_true", help="Print Prometheus exposition format")
    args = parser.parse_args()

    try:
        summary = calculate_csv_cost(
            csv_path=args.csv,
            gpu_price_per_hour_usd=args.gpu_price,
            config_path=args.config,
            force_full_time_share=args.full_time_share,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to calculate cost: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary, indent=2))

    if args.export_metrics:
        model = CostModel(gpu_price_per_hour_usd=summary["gpu_price_per_hour_usd"])
        print("\n--- Prometheus Exposition ---")
        print(model.render_prometheus_metrics(summary["cost_per_million_tokens_usd"]))


if __name__ == "__main__":
    main()
