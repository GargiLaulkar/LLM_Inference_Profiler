# M3 Task 2: GPU Telemetry Prometheus Collector

## 1. Overview & Architecture
As part of **Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**, Role M3 (GPU & Deep-Dive Lead) exposes real-time hardware telemetry from all attached NVIDIA GPUs to Prometheus.

Prometheus scrapes this exporter every 5 seconds at `http://gpu_collector:9102/metrics` as configured in [`docker/prometheus/prometheus.yml`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/docker/prometheus/prometheus.yml#L23-L29).

### Shared Metric Contract
| Metric Name | Prometheus Type | Labels | Unit | Source |
|---|---|---|---|---|
| `llmprof_gpu_utilization_percent` | Gauge | `gpu="<index>"` | % (0.0 to 100.0) | NVML `nvmlDeviceGetUtilizationRates` (`rates.gpu`) |
| `llmprof_gpu_memory_used_bytes` | Gauge | `gpu="<index>"` | Bytes | NVML `nvmlDeviceGetMemoryInfo` (`mem.used`) |
| `llmprof_gpu_power_watts` | Gauge | `gpu="<index>"` | Watts | NVML `nvmlDeviceGetPowerUsage` (`power_mw / 1000.0`) |

---

## 2. Technical Decision: `pynvml` vs. NVIDIA DCGM Exporter

For this profiling platform, we chose an in-process Python NVML collector (`pynvml` / `nvidia-ml-py`) over NVIDIA DCGM (`dcgm-exporter`):

1. **Hardware Universality (Data Center & Consumer GPUs)**:
   - `dcgm-exporter` is designed strictly for enterprise data-center GPUs (Tesla T4, V100, A100, H100). When run on consumer GPUs (such as an RTX 3060, RTX 4090, or laptop GPUs), the DCGM host engine daemon frequently refuses to start or marks devices as unsupported.
   - `pynvml` uses the standard NVIDIA Management Library (NVML) C-API (`libnvidia-ml.so` / `nvml.dll`), which is baked directly into the NVIDIA display driver across **all** GeForce RTX, Quadro, and Tesla cards.
2. **Exact Contract Alignment Without Relabeling Overhead**:
   - `dcgm-exporter` emits vendor-specific metrics (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_POWER_USAGE`) requiring heavy PromQL metric relabeling or mapping configs.
   - Our collector emits the exact shared contract metric names (`llmprof_gpu_*`) directly, matching Grafana dashboards out-of-the-box.
3. **Resilience & Graceful Metric Omission**:
   - If a particular GPU or virtualized cloud instance does not support instantaneous power telemetry, `pynvml` catches the exception and omits the gauge for that device. It never publishes a deceptive fake `0 W`.
4. **Lightweight Footprint**:
   - Zero background daemons; negligible CPU and memory overhead; single Python process with zero C-compilation required.

---

## 3. How It Works

1. **Dedicated Registry Isolation**:
   The collector registers its three gauges on a dedicated `CollectorRegistry` rather than the default global registry. This ensures unit tests remain completely isolated without metric collision.
2. **Decoupled Architecture**:
   - [`read_gpu_samples()`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/llm_profiler/collectors/gpu_collector.py): Discovers all GPUs, queries NVML, and returns raw Python dictionary samples.
   - [`update_gauges()`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/llm_profiler/collectors/gpu_collector.py): Binds samples to Prometheus gauges, skipping any metrics that are `None` (unsupported).
3. **Configuration & CLI Overrides**:
   - Default port (`9102`) and sample interval (`1.0s`) are defined as centralized constants at the top of [`llm_profiler/collectors/gpu_collector.py`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/llm_profiler/collectors/gpu_collector.py).
   - If present, settings are read from the `gpu_collector` block in `config.yaml`.
   - CLI flags (`--port`, `--interval`, `--config`) cleanly override file values.
4. **Graceful Shutdown**:
   Listens for `SIGINT` (Ctrl+C) and `SIGTERM` to safely terminate the polling loop and call `nvmlShutdown()` in a `finally` block.

---

## 4. How to Run

### A. Quick Snapshot / Dry Run (`--once`)
```bash
# Print a single Prometheus metrics exposition snapshot to stdout:
python llm_profiler/collectors/gpu_collector.py --once
```

### B. Standalone Continuous Exporter (Kaggle / Colab / Workstation)
```bash
# Run daemon on default port 9102 (scraped by Prometheus at http://localhost:9102/metrics):
python llm_profiler/collectors/gpu_collector.py

# Custom port or polling rate:
python llm_profiler/collectors/gpu_collector.py --port 9102 --interval 0.5
```

---

## 5. Docker Integration: Ready-to-Paste Compose Block for M1

When M1 is ready to enable the GPU collector in [`docker/docker-compose.yml`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/docker/docker-compose.yml), replace lines 128–143 with this service definition:

```yaml
  # ─── GPU Collector (M3: pynvml Prometheus Exporter) ─────────────────────────
  gpu_collector:
    build:
      context: ..
      dockerfile: docker/Dockerfile.gpu_collector
    image: llmprof_gpu_collector:latest
    container_name: llmprof_gpu_collector
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=utility
      - GPU_COLLECTOR_PORT=${GPU_COLLECTOR_PORT:-9102}
    ports:
      - "${GPU_COLLECTOR_PORT:-9102}:9102"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - llmprof
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9102/metrics', timeout=2).read()"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
```

### Why `NVIDIA_DRIVER_CAPABILITIES=utility` is Required:
By default, Docker containers with `runtime: nvidia` or `capabilities: [gpu]` only mount CUDA compute libraries (`libcuda.so`, `libcudart.so`) for neural network inference. 

NVML (`libnvidia-ml.so` and `nvidia-smi`) is classified by the NVIDIA Container Toolkit under the **`utility`** driver capability. Specifying:
```yaml
environment:
  - NVIDIA_DRIVER_CAPABILITIES=utility
```
(or `NVIDIA_DRIVER_CAPABILITIES=compute,utility`) instructs the NVIDIA container runtime to mount the host's NVML library into the container filesystem at `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1`. Without this variable, `pynvml.nvmlInit()` will fail with `NVMLError_LibraryNotFound`.
