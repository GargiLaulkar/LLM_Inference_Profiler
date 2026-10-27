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
