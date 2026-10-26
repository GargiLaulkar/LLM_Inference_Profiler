# Task 5b — Memory Breakdown Export via gpu_collector

**Role:** M3 (GPU and Deep-Dive Lead)  
**File:** `llm_profiler/collectors/gpu_collector.py`  
**Metric:** `llmprof_memory_breakdown_bytes{component, phase}` — 8 label-set Gauge

---

## How the Export Works

### Metric Contract

| Name | Type | Labels | Unit |
|------|------|--------|------|
| `llmprof_memory_breakdown_bytes` | Gauge | `component`, `phase` | bytes |

**component** values: `weights`, `kv_cache`, `activations`, `other`  
**phase** values: `prefill`, `decode`  
Total: **8 time series**, all exported on the same port 9102 as the GPU metrics.

### Source of Values

The values come from `results/deep_dive/memory_breakdown.json`, which is written by `memory_breakdown.py` (Task 5a). They represent a **static snapshot** from a single deep-dive run, not live per-request measurements. Values do not change until the file is overwritten by another deep-dive run.

### Schema Read (and What Is Ignored)

Only the `phases` key is read:

```json
{
  "phases": {
    "prefill": { "weights": ..., "kv_cache": ..., "activations": ..., "other": ... },
    "decode":  { "weights": ..., "kv_cache": ..., "activations": ..., "other": ... }
  }
}
```

Extra top-level keys — `diagnostics`, `request_id`, `nvml_used_bytes`, `torch_peak_allocated_bytes`, and any others — are **silently ignored**. The file format is forward-compatible.

---

## File-Watch Behaviour

Each sampling cycle the collector calls `os.stat()` on the configured file path and compares `(mtime, size)` to the previous cycle's values.

| Condition | Action |
|-----------|--------|
| File not present at startup | Log one INFO; continue with zero breakdown samples |
| File present, unchanged stat | Skip JSON parsing (no I/O) |
| File present, stat changed | Re-parse JSON; if valid: set 8 series; if invalid: clear series + log WARNING once |
| File was present, now deleted | `gauge.clear()` (remove all 8 series) + log WARNING |

This means:
- **No stale values are ever exported.** If the file disappears or becomes corrupt, the gauge clears.
- **No zeros are exported.** The gauge only has samples after a successful load.
- **Prometheus scrapes will simply not see the metric** (no samples) until a valid file appears.

---

## Why Values Are Static

The deep-dive profiler (`deep_dive.py` + `memory_breakdown.py`) adds overhead (70.2% latency overhead observed on Kaggle T4 with `torch.profiler`). It is designed to run **once**, on demand, for a single request and then exit. The results file is a permanent artifact of that run.

The `gpu_collector` serves these static values alongside real-time GPU metrics. Grafana will display the breakdown as a bar gauge (no time dimension). This is correct: memory layout for a given model checkpoint is stable across requests of similar length.

---

## Configuration

### config.yaml (both copies)

```yaml
gpu_collector:
  port: 9102
  sample_interval_s: 1.0
  memory_breakdown_file: results/deep_dive/memory_breakdown.json
```

The key `memory_breakdown_file` follows the same fallback-constant pattern as `port` and `sample_interval_s`:

1. Value from config file (if found and parseable)  
2. Default constant: `results/deep_dive/memory_breakdown.json`  
3. CLI flag `--memory-breakdown-file PATH` overrides any config value  
4. CLI flag `--memory-breakdown-file ""` (empty string) **disables** breakdown export

The active setting is logged at startup, including its source.

### CLI Examples

```bash
# Default: read from config.yaml
python gpu_collector.py

# Override path
python gpu_collector.py --memory-breakdown-file /data/deep_dive/memory_breakdown.json

# Disable breakdown (GPU-only export)
python gpu_collector.py --memory-breakdown-file ""

# Single snapshot to stdout (includes breakdown if file is present)
python gpu_collector.py --once
```

---

## Docker Compose Service Block for M1

The Dockerfile (`docker/Dockerfile.gpu_collector`) already contains the Python entrypoint (`python -u gpu_collector.py`) and installs `pyyaml`, `prometheus_client`, and `nvidia-ml-py`. Compose's `command` is therefore only the arguments appended to that entrypoint; do not repeat the Python invocation there. No Dockerfile change is required.

**Paste this into `docker/docker-compose.yml` under the `services:` key:**

```yaml
  gpu_collector:
    build:
      context: ..
      dockerfile: docker/Dockerfile.gpu_collector
    restart: unless-stopped
    ports:
      - "9102:9102"
    volumes:
      - "../results/deep_dive:/data/deep_dive:ro"   # mount memory_breakdown.json read-only
    command:
      - "--memory-breakdown-file"
      - "/data/deep_dive/memory_breakdown.json"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

> **Note:** The `../results/deep_dive` host path is relative to `docker/`. Adjust to an absolute path if needed. The volume is mounted `:ro` (read-only) so the collector cannot accidentally modify the results.

### Verification After Starting

```bash
# In Prometheus UI, run:
llmprof_memory_breakdown_bytes

# Should return 8 time series, one per {component, phase} combination.
# In the Grafana dashboard, open "Memory Breakdown (M3)" row — panels 41 and 42
# should show horizontal bar gauges for prefill and decode phases.
```

---

## Pure Functions (for Testing without GPU)

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_memory_breakdown(path)` | `str -> dict or None` | Parse and validate JSON; return `{(comp, phase): float}` or `None` |
| `update_breakdown_gauge(gauge, data)` | `(Gauge, dict) -> None` | Set 8 label-set values on the gauge |

Neither function imports torch, pynvml, or any GPU library — they work on any machine.
