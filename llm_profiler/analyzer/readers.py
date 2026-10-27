"""Telemetry and Log Readers for LLM Inference Profiler (Member 4 - Sprint 2).

Provides robust readers and client wrappers:
1. read_request_csv: Parses request CSV logs adhering to the shared contract schema.
2. read_trace_jsonl: Parses per-stage trace JSON lines with validation of canonical
   stage names, counting and skipping unknown stages and malformed lines without crashing.
3. join_by_request_id: Correlates request logs and per-stage traces on request_id.
4. read_memory_breakdown_json: Loads and validates optional M3 memory_breakdown.json.
5. PrometheusClient: Queries live Prometheus HTTP API for stage latencies, token rates,
   GPU gauges, and memory breakdown gauges, with robust error handling and fallbacks.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from ..instrumentation.stage_timing import STAGE_NAMES, STAGE_NAME_SET

# Contract Constants
CONTRACT_CSV_COLUMNS: List[str] = [
    "request_id",
    "send_ts",
    "end_ts",
    "total_latency_s",
    "prompt_tokens",
    "output_tokens",
    "status",
    "target_rps",
    "scenario_tag",
]

CONTRACT_TRACE_FIELDS: List[str] = [
    "request_id",
    "stage",
    "start_ts",
    "end_ts",
    "duration_s",
]

CONTRACT_COMPONENTS: List[str] = ["weights", "kv_cache", "activations", "other"]
CONTRACT_PHASES: List[str] = ["prefill", "decode"]


class PrometheusUnreachableError(Exception):
    """Raised when Prometheus HTTP endpoint is unreachable or returns a fatal connection error."""


def read_request_csv(file_path: str | Path) -> List[Dict[str, Any]]:
    """Parse request log CSV adhering to the shared contract schema.

    Parameters:
        file_path: Path to request log CSV.

    Returns:
        List of parsed request records with numeric types cast.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required contract headers are missing.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Request CSV file not found: {path_obj}")

    records: List[Dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Request CSV {path_obj} is empty or has no header.")

        # Check required columns
        missing_cols = [c for c in CONTRACT_CSV_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Request CSV {path_obj} missing required contract columns: {missing_cols}")

        for row_idx, row in enumerate(reader, start=2):
            try:
                request_id = str(row["request_id"]).strip()
                send_ts = float(row["send_ts"])
                end_ts = float(row["end_ts"])
                total_latency_s = float(row["total_latency_s"])
                prompt_tokens_value = float(row["prompt_tokens"])
                output_tokens_value = float(row["output_tokens"])
                if (
                    not math.isfinite(prompt_tokens_value)
                    or not math.isfinite(output_tokens_value)
                    or not prompt_tokens_value.is_integer()
                    or not output_tokens_value.is_integer()
                ):
                    raise ValueError("token counts must be finite integers")
                prompt_tokens = int(prompt_tokens_value)
                output_tokens = int(output_tokens_value)
                target_rps = float(row.get("target_rps", 1.0))
                numeric_values = (send_ts, end_ts, total_latency_s, target_rps)
                if (
                    not request_id
                    or not all(math.isfinite(value) for value in numeric_values)
                    or end_ts < send_ts
                    or total_latency_s < 0
                    or prompt_tokens < 0
                    or output_tokens < 0
                    or target_rps < 0
                ):
                    raise ValueError("empty request_id or invalid numeric value")
                record = {
                    "request_id": request_id,
                    "send_ts": send_ts,
                    "end_ts": end_ts,
                    "total_latency_s": total_latency_s,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "status": str(row.get("status", "ok")).strip(),
                    "target_rps": target_rps,
                    "scenario_tag": str(row.get("scenario_tag", "")).strip(),
                }
                records.append(record)
            except (ValueError, TypeError, KeyError) as err:
                # Count and skip malformed row without crashing
                print(f"[WARN] Skipping malformed CSV row {row_idx} in {path_obj}: {err}", file=sys.stderr)
                continue

    return records


def read_trace_jsonl(file_path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Parse per-stage trace JSON lines file.

    Validates stage names against canonical stages. Counts and skips unknown stages
    and malformed lines without crashing.

    Parameters:
        file_path: Path to per-stage trace JSONL file.

    Returns:
        Tuple of (valid_traces, stats_dictionary).
        stats_dictionary contains:
          - "total_lines": Total non-empty lines read.
          - "valid_traces": Number of accepted contract trace records.
          - "skipped_malformed": Lines with invalid JSON or missing/unparseable fields.
          - "skipped_unknown_stages": Lines with unrecognized stage names.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Trace JSONL file not found: {path_obj}")

    valid_traces: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "total_lines": 0,
        "valid_traces": 0,
        "skipped_malformed": 0,
        "skipped_unknown_stages": 0,
    }

    with path_obj.open("r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            stats["total_lines"] += 1

            # 1. Parse JSON
            try:
                raw_record = json.loads(line_str)
            except json.JSONDecodeError:
                stats["skipped_malformed"] += 1
                continue

            if not isinstance(raw_record, dict):
                stats["skipped_malformed"] += 1
                continue

            # 2. Check required fields
            missing = [k for k in CONTRACT_TRACE_FIELDS if k not in raw_record]
            if missing:
                stats["skipped_malformed"] += 1
                continue

            # 3. Validate stage name
            stage_name = str(raw_record["stage"]).strip()
            if stage_name not in STAGE_NAME_SET:
                stats["skipped_unknown_stages"] += 1
                continue

            # 4. Parse and validate timestamps and duration
            try:
                start_ts = float(raw_record["start_ts"])
                end_ts = float(raw_record["end_ts"])
                duration_s = float(raw_record["duration_s"])
            except (ValueError, TypeError):
                stats["skipped_malformed"] += 1
                continue

            if (
                not all(math.isfinite(value) for value in (start_ts, end_ts, duration_s))
                or duration_s < 0.0
                or end_ts < start_ts
            ):
                stats["skipped_malformed"] += 1
                continue

            req_id = str(raw_record["request_id"]).strip()
            if not req_id:
                stats["skipped_malformed"] += 1
                continue

            valid_record = {
                "request_id": req_id,
                "stage": stage_name,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_s": duration_s,
            }
            valid_traces.append(valid_record)
            stats["valid_traces"] += 1

    return valid_traces, stats


def join_by_request_id(
    requests: Sequence[Dict[str, Any]],
    traces: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Join request records and per-stage traces on request_id.

    Parameters:
        requests: Parsed request records (from read_request_csv).
        traces: Parsed trace records (from read_trace_jsonl).

    Returns:
        Dictionary mapping request_id to combined record:
        {
          "request": request_dict,
          "stages": {stage_name: duration_s, ...},
          "traces": [trace_dict, ...]
        }
    """
    traces_by_req: Dict[str, List[Dict[str, Any]]] = {}
    for t in traces:
        req_id = t["request_id"]
        traces_by_req.setdefault(req_id, []).append(t)

    joined: Dict[str, Dict[str, Any]] = {}
    for r in requests:
        req_id = r["request_id"]
        req_traces = traces_by_req.get(req_id, [])
        stages_dict: Dict[str, float] = {}
        for tr in req_traces:
            stage = tr["stage"]
            stages_dict[stage] = stages_dict.get(stage, 0.0) + tr["duration_s"]

        joined[req_id] = {
            "request": r,
            "stages": stages_dict,
            "traces": req_traces,
        }

    return joined


def read_memory_breakdown_json(
    file_path: Optional[str | Path],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse and validate optional memory_breakdown.json according to M3 schema.

    Expected layout:
    {
      "phases": {
        "prefill": {"weights": bytes, "kv_cache": bytes, "activations": bytes, "other": bytes},
        "decode": {"weights": bytes, "kv_cache": bytes, "activations": bytes, "other": bytes}
      }
    }

    Returns:
        Tuple of (parsed_data, status_or_evidence_string).
        status_or_evidence_string is 'valid', 'memory evidence unavailable', or error description.
    """
    if not file_path:
        return None, "memory evidence unavailable"

    path_obj = Path(file_path)
    if not path_obj.exists():
        return None, "memory evidence unavailable"

    try:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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
        for component in CONTRACT_COMPONENTS:
            if isinstance(phase_dict[component], bool):
                return None, (
                    f"memory_breakdown schema mismatch: '{phase_name}.{component}' "
                    "must be a finite non-negative byte count"
                )
            try:
                value = float(phase_dict[component])
            except (TypeError, ValueError):
                return None, (
                    f"memory_breakdown schema mismatch: '{phase_name}.{component}' "
                    "must be a finite non-negative byte count"
                )
            if not math.isfinite(value) or value < 0:
                return None, (
                    f"memory_breakdown schema mismatch: '{phase_name}.{component}' "
                    "must be a finite non-negative byte count"
                )
            phase_dict[component] = value

    return data, "valid"


def resolve_prometheus_url(
    explicit_url: Optional[str] = None,
    config_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Resolve Prometheus URL from CLI, config.yaml, or PROMETHEUS_URL."""
    if explicit_url:
        return explicit_url.rstrip("/")

    # Try config.yaml
    cfg_candidates = []
    if config_path:
        cfg_candidates.append(Path(config_path))
    cfg_candidates.extend([
        Path("config.yaml"),
        Path(__file__).resolve().parent.parent.parent / "config.yaml",
    ])

    for cand in cfg_candidates:
        if cand.exists():
            with cand.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError(f"Prometheus configuration in {cand} must be a YAML mapping")
            prom_url = cfg.get("prometheus_url")
            if prom_url:
                return str(prom_url).rstrip("/")
            prom_port = cfg.get("prometheus_port")
            prom_host = cfg.get("prometheus_host")
            if prom_port is not None or prom_host is not None:
                if prom_port is None or prom_host is None:
                    raise ValueError(
                        f"Both prometheus_host and prometheus_port must be set in {cand}"
                    )
                return f"http://{prom_host}:{prom_port}"

    # Environment variable fallback
    env_url = os.environ.get("PROMETHEUS_URL")
    if env_url:
        return env_url.rstrip("/")

    return None


class PrometheusClient:
    """HTTP Client for Prometheus metrics querying with error resilience.

    Provides access to:
    - Stage latency quantiles (p50, p95) from llmprof_stage_latency_seconds_bucket
    - Token rates from llmprof_tokens_in_total and llmprof_tokens_out_total
    - Live GPU gauges (utilization, memory used, power)
    - Memory breakdown gauges from llmprof_memory_breakdown_bytes
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = resolve_prometheus_url(base_url, config_path)
        self.timeout = float(timeout)

    def query_instant(self, expr: str) -> List[Dict[str, Any]]:
        """Execute an instant query against Prometheus /api/v1/query.

        Parameters:
            expr: PromQL expression string.

        Returns:
            List of metric result dicts (from data.result).

        Raises:
            PrometheusUnreachableError: If connection fails or server is down.
        """
        if not self.base_url:
            raise PrometheusUnreachableError(
                "Prometheus URL is not configured; pass --prometheus-url or set prometheus_url in config.yaml"
            )
        encoded_query = urllib.parse.urlencode({"query": expr})
        url = f"{self.base_url}/api/v1/query?{encoded_query}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LLMProfiler-M4/1.0", "Accept": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status_code = resp.getcode()
                if status_code != 200:
                    raise PrometheusUnreachableError(
                        f"Prometheus query failed with HTTP {status_code} for query: {expr}"
                    )
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as exc:
            raise PrometheusUnreachableError(
                f"Prometheus at {self.base_url} is unreachable: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PrometheusUnreachableError(
                f"Prometheus returned non-JSON response from {url}: {exc}"
            ) from exc

        if not isinstance(payload, dict) or payload.get("status") != "success":
            err_msg = payload.get("error", "Unknown Prometheus query error") if isinstance(payload, dict) else "Invalid response"
            raise PrometheusUnreachableError(
                f"Prometheus query failed for '{expr}': {err_msg}"
            )

        data = payload.get("data", {})
        if not isinstance(data, dict) or not isinstance(data.get("result", []), list):
            raise PrometheusUnreachableError("Prometheus response does not contain a result list")
        return data.get("result", [])

    @staticmethod
    def _value(item: Dict[str, Any]) -> Optional[float]:
        """Return a finite scalar from an instant-query result item."""
        value_pair = item.get("value")
        if not isinstance(value_pair, list) or len(value_pair) != 2:
            return None
        try:
            value = float(value_pair[1])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def get_stage_latency_quantiles(
        self, window: str = "5m"
    ) -> Dict[str, Dict[str, float]]:
        """Query p50 and p95 per-stage latency quantiles from Prometheus histogram.

        Queries:
          histogram_quantile(0.50, sum by (le, stage) (rate(llmprof_stage_latency_seconds_bucket[window])))
          histogram_quantile(0.95, sum by (le, stage) (rate(llmprof_stage_latency_seconds_bucket[window])))

        Returns:
            Dict mapping stage -> {"p50": float, "p95": float}.
            Returns empty dict if no stage metrics are present in Prometheus.
        """
        results: Dict[str, Dict[str, float]] = {}

        # 1. Query p50
        q_p50 = f"histogram_quantile(0.50, sum by (le, stage) (rate(llmprof_stage_latency_seconds_bucket[{window}])))"
        try:
            p50_data = self.query_instant(q_p50)
            for item in p50_data:
                metric = item.get("metric", {})
                stage = metric.get("stage") if isinstance(metric, dict) else None
                val = self._value(item)
                if stage in STAGE_NAME_SET and val is not None and val >= 0:
                    results.setdefault(stage, {})["p50"] = val
        except PrometheusUnreachableError:
            raise

        # 2. Query p95
        q_p95 = f"histogram_quantile(0.95, sum by (le, stage) (rate(llmprof_stage_latency_seconds_bucket[{window}])))"
        try:
            p95_data = self.query_instant(q_p95)
            for item in p95_data:
                metric = item.get("metric", {})
                stage = metric.get("stage") if isinstance(metric, dict) else None
                val = self._value(item)
                if stage in STAGE_NAME_SET and val is not None and val >= 0:
                    results.setdefault(stage, {})["p95"] = val
        except PrometheusUnreachableError:
            raise

        return results

    def get_stage_latency_means(self, window: str = "5m") -> Dict[str, float]:
        """Query mean per-stage latency using histogram sum/count rates."""
        query = (
            "sum by (stage) (rate(llmprof_stage_latency_seconds_sum["
            f"{window}])) / sum by (stage) "
            f"(rate(llmprof_stage_latency_seconds_count[{window}]))"
        )
        results: Dict[str, float] = {}
        for item in self.query_instant(query):
            metric = item.get("metric", {})
            stage = metric.get("stage") if isinstance(metric, dict) else None
            value = self._value(item)
            if stage in STAGE_NAME_SET and value is not None and value >= 0:
                results[stage] = value
        return results

    def get_request_latency_mean(self, window: str = "5m") -> Optional[float]:
        """Query mean end-to-end request latency from the request histogram."""
        query = (
            "sum(rate(llmprof_request_total_seconds_sum["
            f"{window}])) / sum(rate(llmprof_request_total_seconds_count[{window}]))"
        )
        results = self.query_instant(query)
        return self._value(results[0]) if results else None

    def get_stage_latency_summary(self, window: str = "5m") -> Dict[str, Dict[str, float]]:
        """Collect histogram quantiles and mean stage durations for ranking."""
        summary = self.get_stage_latency_quantiles(window)
        for stage, mean in self.get_stage_latency_means(window).items():
            summary.setdefault(stage, {})["mean"] = mean
        total_mean = self.get_request_latency_mean(window)
        if total_mean is not None and total_mean >= 0:
            summary.setdefault("total", {})["mean"] = total_mean
        return summary

    def has_stage_latency_metrics(self, window: str = "5m") -> bool:
        """Check whether per-stage metrics from M2 exporter are active in Prometheus."""
        stages = self.get_stage_latency_summary(window)
        return any(stage != "total" and "mean" in values for stage, values in stages.items())

    def get_token_rates(self, window: str = "5m") -> Dict[str, Optional[float]]:
        """Query token processing rates (tokens/second) from Prometheus."""
        rates: Dict[str, Optional[float]] = {
            "tokens_in_rate": None,
            "tokens_out_rate": None,
        }

        q_in = f"sum(rate(llmprof_tokens_in_total[{window}]))"
        q_out = f"sum(rate(llmprof_tokens_out_total[{window}]))"

        res_in = self.query_instant(q_in)
        res_out = self.query_instant(q_out)
        if res_in:
            value = self._value(res_in[0])
            rates["tokens_in_rate"] = value if value is not None and value >= 0 else None
        if res_out:
            value = self._value(res_out[0])
            rates["tokens_out_rate"] = value if value is not None and value >= 0 else None

        return rates

    def get_gpu_gauges(self) -> Dict[str, Optional[float]]:
        """Query average live GPU telemetry gauges from Prometheus."""
        gauges: Dict[str, Optional[float]] = {
            "gpu_utilization_percent": None,
            "gpu_memory_used_bytes": None,
            "gpu_power_watts": None,
        }

        queries = {
            "gpu_utilization_percent": "avg(llmprof_gpu_utilization_percent)",
            "gpu_memory_used_bytes": "avg(llmprof_gpu_memory_used_bytes)",
            "gpu_power_watts": "avg(llmprof_gpu_power_watts)",
        }

        for key, q in queries.items():
            res = self.query_instant(q)
            if res:
                value = self._value(res[0])
                if value is not None and value >= 0:
                    gauges[key] = value

        return gauges

    def get_memory_breakdown_gauges(self) -> Dict[str, Dict[str, float]]:
        """Query memory breakdown gauges grouped by phase and component."""
        breakdown: Dict[str, Dict[str, float]] = {}

        q = "llmprof_memory_breakdown_bytes"
        res = self.query_instant(q)
        for item in res:
            metric = item.get("metric", {})
            phase = metric.get("phase") if isinstance(metric, dict) else None
            comp = metric.get("component") if isinstance(metric, dict) else None
            val = self._value(item)
            if phase in CONTRACT_PHASES and comp in CONTRACT_COMPONENTS and val is not None and val >= 0:
                breakdown.setdefault(phase, {})[comp] = val

        return breakdown
