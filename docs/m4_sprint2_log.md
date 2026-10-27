# M4 Sprint 2 Implementation Log

## Built

- Added validated request CSV, trace JSONL, memory JSON, and Prometheus HTTP API readers under `llm_profiler/analyzer/`.
- Added `llmprof-analyzer` CLI entry points. Prometheus stage summaries use histogram p50/p95 and histogram sum/count means; missing stage summaries fall back to trace files.
- Updated the analyzer to rank trace or Prometheus stage data, retain `explained_share` warnings, and add measured GPU and memory evidence only when values are available.
- Added a standalone live cost exporter. It observes `llmprof_tokens_out_total`, measures elapsed time from its initial sample, and applies the configured `time_share` and GPU hourly rate.
- Added an M4 exporter Dockerfile and updated the separate M4 Grafana dashboard to query the cost contract metric and identify panels that require M2 histograms.
- Added reader, mocked-Prometheus, analyzer-mode, fallback, synthetic CLI, and cost exposition tests.

## Validation

- Targeted M4/analyzer/cost/synthetic tests: **37 passed**.
- Final full `pytest -q`: **195 passed, 5 skipped**. The timing-sensitive observer-overhead test exceeded its existing threshold in an earlier run, then passed on the final run. No out-of-scope instrumentation or timing test was modified.
- Root analyzer and cost exporter `--help` commands worked, and the M4 dashboard JSON parsed successfully.

## Not Tested Live

- No injection runs, serving experiments, live exporter deployment, or live inference reports were run.
- Prometheus API behavior was tested with mocked HTTP responses, not against the live Prometheus instance. The M2 stage histograms and token counters still need to be confirmed in Prometheus.
- The cost exporter was not built or deployed in Docker, and its scrape target has not been activated.
- Grafana import and rendering were not tested.
- No local `memory_breakdown.json` was provided. The documented `phases` / `component` JSON structure was tested with synthetic fixtures only. The actual M3 snapshot must be checked before interpreting memory evidence.

## Pending Integration

- **M1**: Apply the exact `config.yaml`, Prometheus scrape-target, and Docker Compose snippets in `m4_change_requests.md`. Set `M4_TIME_SHARE` to an approved or measured value rather than relying on a default.
- **M2**: Confirm `llmprof_stage_latency_seconds` histogram labels plus `_sum`/`_count`, `llmprof_request_total_seconds`, and `llmprof_tokens_in_total` / `llmprof_tokens_out_total` are exposed and scraped.
- **M3**: Provide the actual memory snapshot and confirm it contains `phases.prefill` and `phases.decode`, each with `weights`, `kv_cache`, `activations`, and `other` byte counts.
- **M4/manual**: Import the standalone dashboard, verify cost and stage panels populate, and confirm the new exporter is an UP Prometheus target.
