# M1 Platform Architecture & Telemetry Pipeline

This document describes the container architecture, network topology, storage volumes, and telemetry data flow for the LLM Inference Profiler platform stack.

---

## 1. Container & Service Architecture Diagram

```text
+----------------------------------------------------------------------------------------------------+
|                                      HOST SYSTEM (Windows / WSL2)                                  |
|                                                                                                    |
|   +--------------------------------------------------------------------------------------------+   |
|   |                              Docker Network: docker_llmprof                                |   |
|   |                                                                                            |   |
|   |   +------------------------------------+          +------------------------------------+   |   |
|   |   |           vllm (Server)            |          |        prometheus (Metrics)        |   |   |
|   |   |  Image: vllm/vllm-openai:v0.31.0   |          |  Image: prom/prometheus:v3.13.4    |   |   |
|   |   |  Port: 8000                        |          |  Port: 9090                        |   |   |
|   |   |  Runtime: nvidia (GPU enabled)     |          |  Config: ./prometheus.yml:ro       |   |   |
|   |   |  Vol: hf_cache                     |          |  Vol: prometheus_data              |   |   |
|   |   |                                    |          |                                    |   |   |
|   |   |  Endpoints:                        |          |  Scrape Jobs (every 5s):           |   |   |
|   |   |    POST /v1/completions            |          |    - vllm:8000/metrics             |   |   |
|   |   |    GET  /health                    |<---------|    - profiler_exporter:9101        |   |   |
|   |   |    GET  /metrics                   |  Scrapes |    - gpu_collector:9102            |   |   |
|   |   +------------------------------------+          +-----------------+------------------+   |   |
|   |                     ^                                               |                      |   |
|   |                     |                                               |                      |   |
|   |        HTTP Requests|                                    PromQL     |                      |   |
|   |        (Open-loop)  |                                    Queries    v                      |   |
|   |   +-----------------+------------------+          +------------------------------------+   |   |
|   |   |      experiments/load_generator    |          |         grafana (Dashboard)        |   |   |
|   |   |      - Async aiohttp client        |          |  Image: grafana/grafana:11.5.0     |   |   |
|   |   |      - Open-loop Poisson arrivals  |          |  Port: 3000                        |   |   |
|   |   |      - Schema-compliant CSV logs   |          |  Datasource: Prometheus (proxy)    |   |   |
|   |   +------------------------------------+          |  Dashboards: llmprof-main          |   |   |
|   |                                                   +------------------------------------+   |   |
|   |                                                                     ^                      |   |
|   |   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - | - - - - - - - - -    |   |
|   |   Teammate Exporter Blocks (Uncomment to enable):                   |                      |   |
|   |                                                                     | Web UI               |   |
|   |   +------------------------------------+                            | (http://localhost)   |   |
|   |   |      profiler_exporter (M2/M4)     |                            v                      |   |
|   |   |      Port: 9101                    |--------------------+  +-----------------------+   |   |
|   |   |      - llmprof_stage_latency_*     |                    |  | User / Browser        |   |   |
|   |   |      - llmprof_cost_per_million_*  |                    +->| Grafana UI (:3000)    |   |   |
|   |   +------------------------------------+                       +-----------------------+   |   |
|   |                                                                                            |   |
|   |   +------------------------------------+                                                   |   |
|   |   |         gpu_collector (M3)         |                                                   |   |
|   |   |         Port: 9102                 |                                                   |   |
|   |   |         - llmprof_gpu_utilization  |                                                   |   |
|   |   |         - llmprof_memory_breakdown |                                                   |   |
|   |   +------------------------------------+                                                   |   |
|   +--------------------------------------------------------------------------------------------+   |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Telemetry and Data Flow Pipeline

1. **Traffic Generation**:
   The async load generator (`experiments/load_generator.py`) generates controlled synthetic request traffic (parameterized by RPS, prompt tokens, and output tokens from `config.yaml`) directed at `http://127.0.0.1:8000/v1/completions`.
   
2. **Server Execution & Telemetry Generation**:
   - `vllm` receives requests, schedules them via continuous batching, and serves completions using GPU compute.
   - Built-in metrics are continuously updated and exposed at `/metrics`:
     - `vllm:e2e_request_latency_seconds_bucket`
     - `vllm:time_to_first_token_seconds_bucket`
     - `vllm:request_queue_time_seconds_bucket`
     - `vllm:num_requests_running` / `vllm:num_requests_waiting`
     - `vllm:gpu_cache_usage_factor`
     - `vllm:prompt_tokens_total` / `vllm:generation_tokens_total`

3. **Metric Aggregation & Storage**:
   - `prometheus` scrapes `vllm:8000/metrics` every 5 seconds.
   - When enabled by teammates, it simultaneously scrapes `profiler_exporter:9101` and `gpu_collector:9102`.

4. **Visualization**:
   - `grafana` automatically provisions the Prometheus datasource (`uid: prometheus`) and loads `dashboards/llm_profiler_main.json`.
   - Real-time PromQL expressions visualize quantiles (p50, p95, p99), active concurrency, KV cache saturation, and token throughput.

---

## 3. Persistent Volumes

| Volume Name | Target Mount Point | Purpose |
|---|---|---|
| `hf_cache` | `/root/.cache/huggingface` | Shared model weight cache across container restarts |
| `prometheus_data` | `/prometheus` | Prometheus time-series database (7-day retention) |
| `grafana_data` | `/var/lib/grafana` | Grafana dashboard settings and user configurations |
