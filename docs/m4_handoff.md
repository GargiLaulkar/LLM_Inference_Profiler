# Member M4 Handoff Guide: Analyzer, Cost Model & Validation

**Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**  
**Role:** M1 (Platform & Load Lead) &rarr; M4 (Analyzer, Cost Model & Validation Lead)  
**Branch:** `integrate-m3`  
**Host Target:** Windows 11 / WSL2 / NVIDIA RTX 3060 (6GB)

---

## 1. Quick Start: Launching the Full Stack
Start the serving, metric scraping, and dashboard pipeline with these 3 commands:

```powershell
# 1. Environment configuration
Copy-Item .env.example .env

# 2. Install Python profiler dependencies
pip install -r requirements.txt

# 3. Build and launch Docker Compose stack (vLLM, Prometheus, Grafana, GPU Collector)
docker compose -f docker/docker-compose.yml up -d --build
```
Verify all containers are healthy: `docker compose -f docker/docker-compose.yml ps`.

---

## 2. Live Prometheus Metrics & Labels (Verified in Session)
Prometheus runs at `http://localhost:9090` (scraping every 5s). Currently active metrics queried live:

| Metric Name | Labels | Source | Status |
|---|---|---|---|
| `llmprof_gpu_utilization_percent` | `gpu="0"`, `instance="gpu_collector:9102"`, `job="gpu_collector"` | `gpu_collector` (NVML) | **LIVE** (idle 0%, load up to 88%) |
| `llmprof_gpu_memory_used_bytes` | `gpu="0"`, `instance="gpu_collector:9102"`, `job="gpu_collector"` | `gpu_collector` (NVML) | **LIVE** (~4.48 GB allocated) |
| `llmprof_gpu_power_watts` | `gpu="0"`, `instance="gpu_collector:9102"`, `job="gpu_collector"` | `gpu_collector` (NVML) | **LIVE** (~10.2 W idle) |
| `vllm:*` (100 built-in metrics) | `model_name="Qwen/Qwen2.5-0.5B-Instruct"`, `instance="vllm:8000"` | `vllm` native | **LIVE** (e.g. `vllm:num_requests_running`) |

*Note: `profiler_exporter:9101` (`llmprof_stage_latency_seconds`) is DOWN until M2/M4 exporter is launched.*

---

## 3. Results Files, Baselines & Scenario Matrix
- **Results Directory**: All per-request CSV logs are saved to `results/`.
- **Run Index**: `results/index.csv` logs metadata for every benchmark run.
- **Baseline B1 (Client Latency)**: `results/baseline_b1_low_load.csv`, documentation at `docs/baseline_b1.md`.
- **Baseline B2 (vLLM Engine Telemetry)**: `results/baseline_b2_low_load.csv`, documentation at `docs/baseline_b2.md`.
- **Scenario Matrix Runner**:
  ```powershell
  python experiments/run_matrix.py --repeat 3
  ```
  Sweeps scenarios defined in `config.yaml` (`low_load`, `medium_load`, `rps_1`, `rps_5`, `rps_10`, `short_prompt`, `long_prompt`, `long_output`).

---

## 4. Cost Model & Configuration
- **Configuration Key**: `gpu_price_per_hour_usd: 1.0` in `config.yaml`.
- **Inference Cost Formula (USD per 1M Tokens)**:
  $$\text{Cost per 1M Tokens} = \left(\frac{\text{gpu\_price\_per\_hour\_usd} \times (\text{total\_latency\_s} / 3600)}{\text{prompt\_tokens} + \text{output\_tokens}}\right) \times 1{,}000{,}000$$
- Export metric name: `llmprof_cost_per_million_tokens_usd` on port `9101`.

---

## 5. Known Gaps & Current Status
- **NOT YET VERIFIED — Memory Breakdown Panels**: Panels 41 & 42 in Grafana display no data because `results/deep_dive/memory_breakdown.json` is missing. M3 measured this on a Kaggle T4, which does not reflect this RTX 3060.
- **NOT YET VERIFIED — Nsight Deep Dive**: Nsight profiling (`deep_dive_nsys.ps1`) was not run in this session. When running locally, stop vLLM first (`docker compose stop vllm`) to prevent 6GB VRAM exhaustion.
- **Exporter Target**: `profiler_exporter:9101` is marked down in Prometheus until the stage instrumentation server is started.

---

## 6. Shared Contracts (Do Not Change)
1. **Stage Names**: `tokenization`, `queue_wait`, `prefill`, `decode`, `detokenization`, `network`, `total`.
2. **CSV Schema**: `request_id,send_ts,end_ts,total_latency_s,prompt_tokens,output_tokens,status,target_rps,scenario_tag`.
3. **Dashboard UID**: `llmprof-main` at `http://localhost:3000`.
