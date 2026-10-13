# Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks

## Overview
This project profiles per-stage latency (tokenization, queue_wait, prefill, decode, detokenization, network, total), GPU metrics, and cost per 1M tokens in an LLM serving stack (vLLM OpenAI-compatible API), visualized in Grafana.

This component covers:
- Repository architecture & shared schema contracts
- Environment health verification
- Reference server management (vLLM)
- Open-loop async load generator with exact token calibration
- Baseline B1 profiling and scenario analysis
- Contract validation and testing suite

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
│   ├── load_generator.py  # Async open-loop load generator
│   ├── run_baseline_b1.py # Baseline B1 runner
│   ├── check_contracts.py # Request CSV contract validator
│   ├── start_server.sh    # vLLM launch script (Linux/WSL)
│   └── start_server.ps1   # Windows server launch helper
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

### Stage Names
`tokenization`, `queue_wait`, `prefill`, `decode`, `detokenization`, `network`, `total`

### Request Log CSV Schema
Exact column order:
```text
request_id,send_ts,end_ts,total_latency_s,prompt_tokens,output_tokens,status,target_rps,scenario_tag
```

---

## Quick Start

### 1. Setup Virtual Environment
```bash
python -m venv .venv
# Linux / WSL:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Verify Environment
```bash
python llm_profiler/experiments/check_env.py
```

### 3. Start Reference Server
```bash
# In WSL / Linux:
bash llm_profiler/experiments/start_server.sh
```

### 4. Run Smoke Test
```bash
python llm_profiler/experiments/load_generator.py --num-requests 5 --rps 1 --scenario-tag smoke_test
```

### 5. Run Baseline B1
```bash
python llm_profiler/experiments/run_baseline_b1.py
```

### 6. Run Tests
```bash
pytest
```
