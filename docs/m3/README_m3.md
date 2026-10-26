# M3 — GPU and Deep-Dive Lead: Final Overview

**Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**  
**Role:** M3 — GPU metrics, deep-dive profiling, and memory analysis  
**Repo:** `D:\AI-DS\SEM-5\OS\OS_project\LLM_Inference_Profiler`  
**Branch:** `main`  
**Environment:** Dev on Windows laptop (Python 3.14, no GPU, no torch), runs on Kaggle T4 (Python 3.13, torch + transformers + pynvml available)

---

## Task 1 — Environment Check

**Files:** `llm_profiler/collectors/gpu_hello.py`

Validates NVIDIA driver visibility, NVML initialization, pynvml availability, and basic GPU telemetry before building the collector. Logs CUDA driver version, GPU name, free/used/total VRAM, and power draw in a human-readable table. Also generates a Markdown report.

**How to run:**
```bash
python llm_profiler/collectors/gpu_hello.py           # stdout table + Markdown report
python llm_profiler/collectors/gpu_hello.py --json    # machine-readable JSON
```

**What it proves:** NVML is reachable, GPU name and VRAM are correct, and the environment is ready for metric collection.

---

## Task 2 — GPU Collector

**Files:** `llm_profiler/collectors/gpu_collector.py`, `config.yaml`, `llm_profiler/config.yaml`, `requirements-m3.txt`, `docker/Dockerfile.gpu_collector`, `tests/test_gpu_collector.py`

Serves three real-time GPU telemetry metrics on port 9102 via Prometheus HTTP exposition format, using a dedicated `CollectorRegistry` for clean isolation. Gracefully handles NVML unavailability, unsupported metrics (e.g. no power on virtual GPU), and SIGINT/SIGTERM. Also exports the memory breakdown snapshot (Task 5b, see below).

**Shared contract metrics (always-on):**

| Metric | Type | Label | Unit |
|--------|------|-------|------|
| `llmprof_gpu_utilization_percent` | Gauge | `gpu` | % (0–100) |
| `llmprof_gpu_memory_used_bytes` | Gauge | `gpu` | bytes |
| `llmprof_gpu_power_watts` | Gauge | `gpu` | watts |

**How to run:**
```bash
python llm_profiler/collectors/gpu_collector.py           # continuous daemon, port 9102
python llm_profiler/collectors/gpu_collector.py --once    # single snapshot to stdout
python llm_profiler/collectors/gpu_collector.py --help    # all options
```

**What it proves:** GPU metrics are available at `http://localhost:9102/metrics` for Prometheus to scrape. Exit code 1 if NVML fails (fails fast instead of exporting zeros).

---

## Task 3 — GPU Dashboard Panels

**Files:** `dashboards/llm_profiler_main.json`, `tests/test_dashboard_gpu_panels.py`

Filled Row 3 ("GPU Metrics (M3)") in the main Grafana dashboard with three timeseries panels (ids 31–33) for utilization, memory, and power. Row 3 is expanded (collapsed: false) with its child panels as top-level siblings in the JSON array, as required by Grafana.

**Panel layout:**

| Panel | Metric | gridPos |
|-------|--------|---------|
| 31 | `llmprof_gpu_utilization_percent` | y=17, x=0, w=8, h=8 |
| 32 | `llmprof_gpu_memory_used_bytes` | y=17, x=8, w=8, h=8 |
| 33 | `llmprof_gpu_power_watts` | y=17, x=16, w=8, h=8 |

All three use `{{gpu}}` legend format, table-mode legend, and all-mode tooltip.

**How to run:** Import or provision `dashboards/llm_profiler_main.json` in Grafana after Prometheus is configured to scrape the collector.

**What it proves:** The Grafana dashboard layout is valid and the GPU row is ready to populate when the collector is running.

---

## Task 4 — Deep-Dive Mode (Single Request Profiling)

**Files:** `llm_profiler/collectors/deep_dive.py`, `llm_profiler/collectors/deep_dive_nsys.sh`, `llm_profiler/collectors/deep_dive_nsys.ps1`, `tests/test_deep_dive.py`

Profiles a single request through the local model (Qwen/Qwen2.5-0.5B-Instruct loaded directly via HuggingFace, not via vLLM) at the Python level using `torch.profiler`. Designed to be separate from the always-on collector — profiling overhead is too high for continuous use.

**Stage names (contract):** `tokenization → prefill → decode → detokenization`

**Output files (in `results/`):**
- `{request_id}_stage_trace.jsonl` — per-stage JSON lines with `request_id, stage, start_ts, end_ts, duration_s`
- `{request_id}_summary.json` — total_s, prefill_s, mean_decode_token_s, p95_decode_token_s, output_tokens
- `torch_profiler/` — Chrome trace JSON for `chrome://tracing` or Perfetto

**Nsight Systems hooks:** `deep_dive_nsys.sh` (Linux/Kaggle) and `deep_dive_nsys.ps1` (Windows/RTX 3060) wrap the script with `nsys profile` to produce `.nsys-rep` files.

**How to run (Kaggle or RTX 3060):**
```bash
python llm_profiler/collectors/deep_dive.py --mode deep_dive --prompt-tokens 128 --output-tokens 64
```

**What it proves:** Stage-level timing for a single request. Distinguishes prefill (batched attention) from per-token decode latency.

---

## Task 5a — Memory Breakdown Measurement

**Files:** `llm_profiler/collectors/memory_breakdown.py`, `docs/m3/task5a_memory_breakdown.md`, `tests/test_memory_breakdown.py`

Measures GPU memory allocation at the phase level for a single request: where bytes go during prefill vs. decode, split into `weights`, `kv_cache`, `activations`, and `other`.

**Measurement method:**
- Weights: `torch.cuda.memory_allocated()` before the request (after model load + warmup)
- KV cache: analytical formula based on model config × token count, validated against `past_key_values` tensor size
- Activations: `peak_allocated − allocated_at_phase_start − kv_growth_in_phase` (transient memory only; avoids counting resident weights or leaked logit tensors)
- Other: `nvml_used − weights − kv_cache − activations`

**Output files:**
- `results/deep_dive/memory_breakdown.json` — structured JSON with `phases`, `diagnostics`, `request_id`
- `results/deep_dive/*_memory_breakdown.png` — stacked bar chart (all components + dynamic-only)
- `results/deep_dive/m3_evidence.json` — generated evidence file for M4 analyzer

**How to run (Kaggle T4):**
```bash
python llm_profiler/collectors/memory_breakdown.py --prompt-tokens 128 --output-tokens 64
```

**What it proves:** It measures phase-level allocation for this single request and checks the analytical KV-cache estimate against the measured cache tensor size.

---

## Task 5b — Memory Export via gpu_collector

**Files:** `llm_profiler/collectors/gpu_collector.py` (extended), `config.yaml`, `llm_profiler/config.yaml`, `dashboards/llm_profiler_main.json` (Row 4 filled), `tests/test_gpu_collector.py`, `tests/test_dashboard_gpu_panels.py`, `docs/m3/task5b_memory_export.md`

Exports the memory breakdown snapshot as a Prometheus metric:

| Metric | Type | Labels | Unit |
|--------|------|--------|------|
| `llmprof_memory_breakdown_bytes` | Gauge | `component`, `phase` | bytes |

The collector reads `results/deep_dive/memory_breakdown.json`, watches for file changes via `os.stat()` each cycle, and updates 8 label-sets on the gauge. Values are static (from the last deep-dive run) — not live memory. If the file is deleted or corrupt, the gauge is cleared (no stale values exported).

Row 4 of the dashboard ("Memory Breakdown (M3)") is now expanded with two bar gauge panels:

| Panel | Metric Filter | Title | gridPos |
|-------|---------------|-------|---------|
| 41 | `{phase="prefill"}` | Memory by component, prefill | y=26, x=0, w=12, h=8 |
| 42 | `{phase="decode"}` | Memory by component, decode | y=26, x=12, w=12, h=8 |

See [`docs/m3/task5b_memory_export.md`](task5b_memory_export.md) for the docker-compose service block.

**How to run:**
```bash
python llm_profiler/collectors/gpu_collector.py --memory-breakdown-file results/deep_dive/memory_breakdown.json
```
Then query `llmprof_memory_breakdown_bytes` in Prometheus.

**What it proves:** A valid deep-dive file produces the eight component/phase byte series; the dashboard visualizes the last saved snapshot alongside live GPU telemetry.

---

## Kaggle T4 Measured Results

All values from a single run: model `Qwen/Qwen2.5-0.5B-Instruct`, 128 prompt tokens, 64 output tokens.

### Memory Breakdown (MiB)

| Component | Prefill | Decode |
|-----------|---------|--------|
| weights | 942.29 | 942.29 |
| kv_cache | 1.50 | 2.24 |
| activations | 38.22 | 2.47 |
| other | 659.18 | 694.18 |
| **NVML total** | **1641.19** | **1641.19** |

Additional diagnostics:
- PyTorch reserved: 1,060.00 MiB
- Resident beyond weights at phase start: **18.84 MiB** (7.88 MiB at load + 10.96 MiB during warmup; cause not identified)
- Outside PyTorch's allocator: **581.19 MiB** (NVML minus reserved)
- KV cache formula: 12,288 bytes/token — matches `past_key_values` tensor size (validates the model config)

### Timing

| Mode | Prefill | Decode/token | Total (128→64) |
|------|---------|-------------|----------------|
| Plain HF eager | 34 ms | 29.5 ms | 1.89 s |
| With `torch.profiler` | — | — | 3.22 s |
| **Profiler overhead** | | | **70.2%** |

---

## Evidence for M4

`memory_breakdown.py` writes `m3_evidence.json` beside `memory_breakdown.json` in its output directory (by default, `results/deep_dive/`). The generated JSON is not present in this checkout; its schema is defined by the deep-dive collector. The table below explains every top-level field emitted by that schema. Values shown are from the Kaggle T4 run where available; GPU-capacity/free-space estimates depend on the actual device.

| Field | Kaggle T4 run / unit | What it supports |
|-------|----------------------|------------------|
| `gpu_total_bytes` | Device capacity, bytes (read from NVML; device-specific) | Denominator for free-memory estimates; not the amount used by this request. |
| `weights_bytes` | 942.29 MiB | Model-weight allocation for this checkpoint; about 57% of the measured 1641.19 MiB NVML use. |
| `kv_cache_bytes_per_token` | 12,288 bytes/token | Measured KV growth rate; it matches the analytical model formula for this model. |
| `kv_cache_bytes_end_of_request` | About 2.24 MiB | KV cache at the end of the 128-in/64-out run. |
| `kv_share_of_used_memory_decode` | About 0.00136 (fraction; about 0.14%) | Decode KV cache as a fraction of NVML-used memory in this run only. |
| `activations_peak_prefill_bytes` | 38.22 MiB | Estimated peak prefill activations for this request. |
| `activations_peak_decode_bytes` | 2.47 MiB | Estimated peak decode activations for this request. |
| `other_bytes_decode` | 694.18 MiB | Decode-phase residual component. It includes unclassified resident GPU use; it does not identify a specific cause. |
| `nvml_free_after_decode_bytes` | Bytes; device- and run-specific | Free VRAM after decode, computed from device capacity minus NVML use. |
| `kv_tokens_that_fit_in_free_memory_estimate` | Estimated token count; device-specific | A simple free-VRAM / KV-bytes-per-token estimate, not a safe context or batch-size limit; it ignores weights, activations, and runtime overhead. |
| `dominant_component_prefill` | `weights` | Largest of the four measured/estimated components during prefill. |
| `dominant_component_decode` | `weights` | Largest of the four measured/estimated components during decode. |
| `prefill_s` *(optional)* | Seconds; populated only when a timing summary is supplied | Prefill-stage duration from that summary. |
| `decode_total_s` *(optional)* | Seconds; populated only when a timing summary is supplied | Total decode-stage duration from that summary. |
| `decode_mean_token_s` *(optional)* | Seconds/token; populated only when a timing summary is supplied | Mean decode-token duration from that summary. |
| `notes` | Array of explanatory strings | Records that the run used standalone HF eager (not vLLM), vLLM's allocator differs, component estimates are derived from allocated/NVML peaks, and resident pre-request allocation beyond weights is included in `other`. |

For the tested 128-in/64-out request, the KV cache and activations together account for a small share of total NVML-used memory. This is evidence about this model, sequence length, and single-request eager run—not a claim about longer contexts, concurrent requests, or vLLM. The separate NVML-minus-PyTorch-reserved result (581.19 MiB) is an allocator comparison, not an independent field in `m3_evidence.json`; the 18.84 MiB resident-beyond-weights observation has an unidentified cause and is included in `other`.

**Inference:** KV cache grows with token count according to `n_layers × 2 × n_heads × head_dim × n_tokens × dtype_bytes`; therefore longer contexts or larger batches can increase its share. This run establishes the per-token formula, but does not measure those larger workloads.

---

## Limitations

| Limitation | Detail |
|------------|--------|
| Kaggle T4, not RTX 3060 | Power envelope, VRAM, and driver behavior differ. The RTX 3060 run should be compared. |
| HuggingFace eager mode, not vLLM | Continuous batching, paged KV cache, and vLLM's CUDA graphs will change memory and latency profiles significantly. |
| Nsight Systems not run on Kaggle | Nsight requires a local install and persistent storage. Only `torch.profiler` Chrome traces were produced. |
| `torch.profiler` overhead: 70.2% | Deep-dive timing is not representative of production latency. Use always-on stage timing from M2 for latency metrics. |
| Memory breakdown values are static | The Prometheus metric shows the last deep-dive run — not live inference memory. It does not reflect batched requests or vLLM's memory pool. |
| Resident 18.84 MiB unexplained | 7.88 MiB at model load + 10.96 MiB during warmup not attributed to a specific component; its cause remains unknown. |

---

## M1 Handoff Checklist

> Run these steps after cloning the repo and starting the Docker stack.

1. **Enable the gpu_collector compose block** — add the service block from [`docs/m3/task5b_memory_export.md`](task5b_memory_export.md) to `docker/docker-compose.yml`. In Prometheus UI (`http://localhost:9090/targets`), confirm `gpu_collector` shows **UP** with labels `job="gpu_collector"`.

2. **Load the Grafana dashboard and screenshot** — import `dashboards/llm_profiler_main.json`. Under load (run the load generator at rps=5), screenshot:
   - Row 3 "GPU Metrics (M3)" — should show live utilization/memory/power timeseries
   - Row 4 "Memory Breakdown (M3)" — should show bar gauges for prefill and decode phases

3. **Run the Nsight Systems deep-dive on the RTX 3060** — execute `llm_profiler/collectors/deep_dive_nsys.ps1` (Windows) with the model running locally. Send back the `.nsys-rep` file for kernel-level analysis. **Do not commit large trace files** (including `.nsys-rep` and Chrome trace JSON files).

4. **Mount the results folder and confirm 8 series in Prometheus** — after running `memory_breakdown.py` on the RTX 3060, mount `results/deep_dive/` read-only and pass `--memory-breakdown-file /data/deep_dive/memory_breakdown.json` to the collector. In Prometheus, query `llmprof_memory_breakdown_bytes` and confirm all 8 component/phase time series are present.

---

*Last updated: Task 5b completion. All 141 tests pass (5 skipped — live GPU tests).*
