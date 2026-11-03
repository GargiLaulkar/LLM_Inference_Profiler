# M4 Integration Notes for M1

The standalone dashboard is `dashboards/llm_profiler_m4_analyzer.json`; M1 may merge its panels into the main dashboard later.

Add this scrape target to `docker/prometheus/prometheus.yml`:

```yaml
- job_name: "m4_cost_exporter"
  static_configs:
    - targets: ["m4_cost_exporter:9103"]
  metrics_path: /metrics
  scrape_interval: 5s
```

Add this optional, explicitly profiled service to `docker/docker-compose.yml`. It uses the existing `llmprof` network, Prometheus service name, and M4 Dockerfile path. An unset `M4_TIME_SHARE` expands to an empty value and does not prevent the default Compose stack from starting; set it when enabling the `m4` profile.

```yaml
m4_cost_exporter:
  profiles: ["m4"]
  build:
    context: ..
    dockerfile: docker/Dockerfile.m4_cost_exporter
  container_name: llmprof_m4_cost_exporter
  environment:
    M4_TIME_SHARE: ${M4_TIME_SHARE:-}
    PROMETHEUS_URL: http://prometheus:9090
  command:
    - --host
    - 0.0.0.0
    - --port
    - ${M4_COST_EXPORTER_PORT:-9103}
  ports:
    - ${M4_COST_EXPORTER_PORT:-9103}:9103
  depends_on:
    - prometheus
  networks:
    - llmprof
```

The current `config.yaml` has no Prometheus URL key. Add the keys the analyzer reads:

```yaml
prometheus_url: "http://127.0.0.1:9090"

cost_exporter:
  host: "127.0.0.1"
  port: 9103
```

`gpu_price_per_hour_usd` is already configured at the root of `config.yaml`. `time_share` is supplied with `M4_TIME_SHARE` or `--time-share`; no default billing share is assumed.

## Sprint 3 E1 integration requests

- **M2 — injection hooks:** Implement and document the proposed `LLMPROF_SLOW_TOKENIZER_MS`, `LLMPROF_NO_BATCHING`, and `LLMPROF_CPU_PREPROCESS_MS` server behavior. Until source implementations are present, M4 reports these hooks as pending and does not run live injections.
- **M2 — per-request trace output:** The E1 runner starts the server with `LLMPROF_TRACE_PATH` set to a trial-specific JSONL destination. Write one contract trace record per measured stage using `request_id`, `stage`, `start_ts`, `end_ts`, and `duration_s`, correlating requests with the `X-Request-ID` sent by M1. The trace must be complete before the server is stopped.
- **M2 — profiler on/off switch:** Add `LLMPROF_PROFILER_ENABLED=0|1` to the gateway/server. `0` must keep the same gateway/server in pass-through mode with stage timing, trace writing, and metric ingest disabled; `1` enables all three. E6 will run the same workload through that same gateway in both modes, so the difference isolates profiler cost. Until the switch exists, E6 remains `pending_live`.
- **M2 — optional gateway-overhead comparison:** Separately measure the same workload direct-to-vLLM and through the gateway. Report this only as gateway overhead, not as profiler overhead.
- **M3 — power classification (optional):** The M4 evidence rule can call decode power high only when a measured power limit is available. If desired, expose a power-limit gauge so M4 can compare `llmprof_gpu_power_watts` with the device's measured limit; otherwise M4 reports no high-power classification.
