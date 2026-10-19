# Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks

## Overview
This project profiles per-stage latency (`tokenization`, `queue_wait`, `prefill`, `decode`, `detokenization`, `network`, `total`), GPU metrics, and cost per 1M tokens in an LLM serving stack (vLLM OpenAI-compatible API), visualized in Grafana.

---

## Repository Structure

```text
llm_profiler/
├── instrumentation/   # Stage timers and request tracing hooks
├── collectors/        # GPU metric collectors (nvidia-smi, DCGM, Prometheus)
├── analyzer/          # Offline log processing and cost/latency modeling
├── dashboards/        # Grafana dashboard JSON models
├── docker/            # Docker compose and container definitions
├── experiments/       # Load generator, baseline runner, server scripts
│   ├── check_env.py       # Environment and GPU verification script
│   ├── reference_server.py # OpenAI & Prometheus compatible reference server
│   ├── load_generator.py  # Async open-loop load generator
│   ├── run_baseline_b1.py # Baseline B1 runner
│   ├── check_contracts.py # Request CSV contract validator
│   ├── start_server.sh    # vLLM launch script (Linux / WSL)
│   └── start_server.ps1   # Server launch script (Windows)
├── tests/             # Unit and integration test suite
│   └── test_load_generator.py
├── docs/              # Progress logs, baseline reports, team sync templates
│   ├── m1_progress.md
│   ├── baseline_b1.md
│   └── team_sync_template.md
├── results/           # Benchmark CSV outputs (request logs)
└── config.yaml        # Centralized configuration file
```

---

## Shared Contracts

### 1. Stage Names
`tokenization`, `queue_wait`, `prefill`, `decode`, `detokenization`, `network`, `total`

### 2. Request Log CSV Schema
Exact column order:
```text
request_id,send_ts,end_ts,total_latency_s,prompt_tokens,output_tokens,status,target_rps,scenario_tag
```

---

## Clean-Start Guide

### 1. Prerequisites
- **OS**: Windows 11 with WSL2 (Ubuntu) or Linux.
- **Hardware**: NVIDIA GPU (e.g., RTX 3060 6GB) with NVIDIA Driver 535+ / CUDA 12.4+.
- **Software**: Docker Desktop (WSL2 backend enabled), Python 3.10+, GNU Make.

### 2. Environment Setup
Copy the example environment configuration:
```bash
cp .env.example .env
```
Ensure virtual environment dependencies are installed:
```bash
pip install -r requirements.txt
```

### 3. Start the Platform Stack
Start the serving stack (vLLM, Prometheus, Grafana) in the background:
```bash
make up
```
To check container status:
```bash
docker compose -f docker/docker-compose.yml ps
```
To tail logs:
```bash
make logs
```

### 4. Service Endpoints & Dashboards
- **Grafana Observability Dashboard**: [http://localhost:3000](http://localhost:3000)
  - Default User: `admin`
  - Default Password: `changeme_strong_password_here` (configured in `.env`)
  - Pre-provisioned Dashboard: `LLM Inference Profiler` (`uid: llmprof-main`)
- **Prometheus Metrics & Target Status**: [http://localhost:9090](http://localhost:9090)
- **vLLM API & Metrics**:
  - Completions: `http://localhost:8000/v1/completions`
  - Built-in Metrics: `http://localhost:8000/metrics`
  - Health: `http://localhost:8000/health`

### 5. Running Profiler Benchmarks
```bash
# Quick dry-run & smoke test:
make smoke

# Run Baseline B1 (client-side latency benchmarks):
make baseline-b1

# Run Baseline B2 (built-in vLLM Prometheus metrics):
make baseline-b2

# Run Scenario Matrix sweep:
make matrix

# Run test suite & contract validation:
make test

# Tear down the stack:
make down
```

---

## Teammate Integration Guide

### How Teammates Enable Custom Exporters
In `docker/docker-compose.yml`, uncomment the respective service blocks:
1. **M2 / M4 Stage Timers & Cost Exporter** (`port 9101`):
   - Uncomment the `profiler_exporter` block.
   - Set the image name to your built exporter container.
2. **M3 GPU Telemetry Collector** (`port 9102`):
   - Uncomment the `gpu_collector` block.
   - Set the image name (e.g. `dcgm-exporter` or custom collector).

Prometheus is already configured in `docker/prometheus/prometheus.yml` to scrape both endpoints every 5 seconds.

### Shared Contract Metric Names
When implementing exporters or querying dashboards, use these exact metric names:

| Metric Name | Labels | Description | Owner |
|---|---|---|---|
| `llmprof_stage_latency_seconds` | `stage="tokenization\|queue_wait\|prefill\|decode\|detokenization\|network\|total"` | Per-stage latency histogram / summary | M2 |
| `llmprof_request_total_seconds` | | End-to-end request duration | M2 |
| `llmprof_tokens_in_total` | | Cumulative input prompt tokens | M2 |
| `llmprof_tokens_out_total` | | Cumulative generated output tokens | M2 |
| `llmprof_gpu_utilization_percent` | `gpu="0"` | GPU core utilization percentage | M3 |
| `llmprof_gpu_memory_used_bytes` | `gpu="0"` | Total GPU memory currently allocated | M3 |
| `llmprof_gpu_power_watts` | `gpu="0"` | Instantaneous GPU power draw | M3 |
| `llmprof_memory_breakdown_bytes` | `component="weights\|kv_cache\|activations", phase="idle\|prefill\|decode"` | Fine-grained VRAM breakdown | M3 |
| `llmprof_cost_per_million_tokens_usd` | | Amortized inference dollar cost per 1M tokens | M4 |

### Adding Panels to Grafana
The dashboard `dashboards/llm_profiler_main.json` contains pre-arranged, labeled rows with placeholders for M2, M3, and M4. Once your exporter is running and scraped, open the panel in Grafana, bind it to datasource `prometheus`, and enter your PromQL query using the contract metric names above.

