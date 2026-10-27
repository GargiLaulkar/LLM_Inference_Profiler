# M4 Change Requests & Upstream Integration Notes

This document logs formal change requests from Member M4 (Analyzer & Validation Lead) to existing files maintained by M1 (Platform/Load Lead), M2 (Instrumentation Lead), and M3 (GPU Telemetry Lead).

In strict accordance with **Rule 1** ("DO NOT modify any existing file written by M1, M2 or M3... Only ADD new files. If I need a new config key or a change to someone else's file, write the request in docs/m4_change_requests.md instead of editing"), all existing configuration and source files remain unmodified on the `m4-analyzer-validation` branch.

---

## Change Request CR-01: Expose Cost Metric in `profiler_exporter` (Target: M2)
- **Target File**: `llm_profiler/instrumentation/prometheus_exporter.py`
- **Component Owner**: Member 2 (Stage Instrumentation Lead)
- **Description**: Add Prometheus gauge `llmprof_cost_per_million_tokens_usd` (Gauge, no labels) to `MetricRegistry` and `PrometheusExporter`.
- **Rationale**:
  The shared contract specifies `llmprof_cost_per_million_tokens_usd` as a top-level Gauge without labels, fed by the cost model based on `gpu_price_per_hour_usd`. Adding this metric to the central exporter allows Grafana (port 3000) and Prometheus (port 9090) to scrape stage latencies and cost simultaneously on port 9101.
- **Proposed Diff**:
  ```python
  # In MetricRegistry:
  self._cost_per_million_tokens_usd = 0.0

  def set_cost_per_million_tokens_usd(self, cost_usd: float) -> float:
      self._cost_per_million_tokens_usd = max(0.0, float(cost_usd))
      return self._cost_per_million_tokens_usd

  # In render():
  lines.append("# HELP llmprof_cost_per_million_tokens_usd Amortized inference dollar cost per 1M tokens.")
  lines.append("# TYPE llmprof_cost_per_million_tokens_usd gauge")
  lines.append(f"llmprof_cost_per_million_tokens_usd {self._cost_per_million_tokens_usd}")
  ```
- **M4 Workaround**: M4 provides a standalone cost exporter and offline analyzer export module in `llm_profiler/analyzer/cost_model.py` so validation proceeds with zero blockers.

---

## Change Request CR-02: Support Per-Stage Trace Logging in Load Generator (Target: M1 / M2)
- **Target File**: `llm_profiler/experiments/load_generator.py`
- **Component Owner**: Member 1 & Member 2
- **Description**: Add an optional CLI flag `--traces-output <path.jsonl>` to `experiments/load_generator.py` to stream per-stage duration traces for each request alongside the request log CSV.
- **Rationale**:
  The shared contract requires per-stage trace JSON lines with fields:
  `request_id, stage, start_ts, end_ts, duration_s`.
  Currently, `load_generator.py` generates the request log CSV with `request_id, send_ts, end_ts, total_latency_s, prompt_tokens, output_tokens, status, target_rps, scenario_tag`. Correlating stage breakdowns with CSV request logs enables the offline analyzer to compute exact per-request `explained_share`.
- **M4 Workaround**: M4 provides an instrumented request trace generator and bottleneck injector in `llm_profiler/analyzer/` and `experiments/` that produces paired CSV and JSONL traces strictly matching both schemas.

---

## Change Request CR-03: Graceful Handling for Missing `memory_breakdown.json` (Target: M3)
- **Target File**: `llm_profiler/collectors/gpu_collector.py`
- **Component Owner**: Member 3 (GPU Profiling Lead)
- **Description**: Ensure `gpu_collector.py` logs an informative warning once when `results/deep_dive/memory_breakdown.json` is not present, without raising an unhandled exception or reporting fabricated memory numbers.
- **Rationale**:
  `memory_breakdown.json` is generated only during single-request offline deep-dive profiling. During standard inference benchmarks, the file may be absent.
- **Status**: Verified in `test_memory_breakdown.py`; M4 analyzer gracefully treats missing deep-dive snapshots as optional context.

---

## Change Request CR-04: Grafana Main Dashboard Row 5 Update (Target: M1)
- **Target File**: `dashboards/llm_profiler_main.json`
- **Component Owner**: Member 1 (Platform Lead)
- **Description**: Expand Row 5 ("Cost per 1M Tokens (M4 — awaiting profiler_exporter)") to include the panel bindings for `llmprof_cost_per_million_tokens_usd` and Top Bottlenecks table.
- **Rationale**:
  To avoid touching M1's provisioned dashboard file, M4 ships a complementary standalone dashboard: `dashboards/llm_profiler_m4_analyzer.json`. When ready, M1 can merge the panel JSON into `dashboards/llm_profiler_main.json`.
