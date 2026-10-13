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

## Step-by-Step Instructions

### 1. Create and Activate Virtual Environment
```bash
# On Linux / WSL:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Check Environment & GPU
```bash
python experiments/check_env.py
```

### 4. Start Server
```bash
# In WSL / Linux:
bash experiments/start_server.sh

# On Windows PowerShell:
.\experiments\start_server.ps1
```
Verify endpoints:
```bash
# Check completion endpoint
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"prompt\": \"The capital of France is\", \"max_tokens\": 16}"

# Check metrics endpoint
curl http://127.0.0.1:8000/metrics
```

### 5. Run Smoke Test
```bash
python experiments/load_generator.py --dry-run
python experiments/load_generator.py --num-requests 5 --rps 1 --scenario-tag smoke_test
```

### 6. Run Baseline B1
```bash
# Run low_load scenario (100 requests):
python experiments/run_baseline_b1.py --scenario low_load

# Run all scenarios defined in config.yaml:
python experiments/run_baseline_b1.py
```

### 7. Run Test Suite
```bash
pytest -v
```

### 8. Validate Contracts on Generated CSVs
```bash
python experiments/check_contracts.py results/baseline_b1_low_load.csv
```
