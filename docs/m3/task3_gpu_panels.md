# M3 Task 3: Grafana Dashboard GPU Metrics Panels

## 1. Overview
In **Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**, Role M3 (GPU & Deep-Dive Lead) is responsible for visualizing GPU hardware metrics in the Grafana observability dashboard (`llmprof-main`, located in [`dashboards/llm_profiler_main.json`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/dashboards/llm_profiler_main.json)).

Under the **GPU Metrics (M3)** row, the placeholder text panel has been replaced with three side-by-side timeseries panels hooked directly into Prometheus metrics exported by [`gpu_collector.py`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/llm_profiler/collectors/gpu_collector.py) on port `9102`.

---

## 2. Panel Specifications & PromQL Queries

| Panel Title | Panel ID | PromQL Expression | Unit | Bounds | Description |
|---|---|---|---|---|---|
| **GPU Utilization (%)** | `31` | `llmprof_gpu_utilization_percent` | `percent` | `0` to `100` | Percentage of time kernels are active on the GPU cores |
| **GPU Memory Used** | `32` | `llmprof_gpu_memory_used_bytes` | `bytes` (IEC) | `min: 0` | Physical VRAM allocated for model weights, KV cache, and activations |
| **GPU Power (W)** | `33` | `llmprof_gpu_power_watts` | `watt` | `min: 0` | Instantaneous total power draw of the graphics card |

### Why Plain Metric Names (No `rate()`) are Used
In Prometheus, the `rate()` and `irate()` functions calculate the per-second rate of change over time, which is strictly designed for **cumulative counters** (such as `vllm:prompt_tokens_total` or `llmprof_tokens_in_total` that monotonically increase).

In contrast, our GPU metrics are **gauges**:
- Utilization (e.g. $45\%$) is an instantaneous reading.
- Memory used (e.g. $8.5\text{ GiB}$) is the current state of allocated VRAM.
- Power draw (e.g. $65.0\text{ W}$) is the instantaneous power dissipation.

Applying `rate()` to a gauge is mathematically meaningless and causes fluctuating, negative, or blank graphs. Using plain PromQL metric names renders continuous lines reflecting actual instantaneous hardware values.

### Multi-GPU Support & Legend Formatting
Each query uses the legend template:
```text
GPU {{gpu}}
```
When running on multi-GPU setups (such as a machine with Docker and an NVIDIA GPU, for example M1's RTX 3060 stack), Prometheus renders separate lines for `GPU 0` and `GPU 1`. The legend table at the bottom displays real-time `Last`, `Max`, and `Mean` statistics for each accelerator.

---

## 3. Grid Coordinates & Layout
The three panels are positioned side-by-side on an 8-column grid directly beneath the GPU row header (`y=16`):
- **Panel 31** ("GPU Utilization (%)"): `x=0, y=17, w=8, h=8`
- **Panel 32** ("GPU Memory Used"): `x=8, y=17, w=8, h=8`
- **Panel 33** ("GPU Power (W)"): `x=16, y=17, w=8, h=8`

Together they fill the 24-unit width without overlapping or shifting any surrounding rows.

---

## 4. Manual Verification Checklist for M1 / Team

Once the platform stack is running on a machine with Docker and an NVIDIA GPU (for example M1's RTX 3060 stack) via `make up` or `docker compose -f docker/docker-compose.yml up -d`:

1. **Open Grafana**: Navigate to [http://localhost:3000](http://localhost:3000) (Login: `admin` / `.env` password).
2. **Open Dashboard**: Go to **Dashboards** > **LLM Inference Profiler** (`uid: llmprof-main`).
3. **Inspect the GPU Row**:
   - Locate the **GPU Metrics (M3)** row.
   - Confirm that three panels appear side-by-side: **GPU Utilization (%)**, **GPU Memory Used**, and **GPU Power (W)**.
   - Confirm the legend table displays `GPU 0` (and `GPU 1` if multi-GPU) with `Last`, `Max`, and `Mean`.
4. **Trigger Active Load**:
   Run synthetic request traffic using the open-loop load generator:
   ```bash
   python experiments/load_generator.py --rps 2 --num-requests 20 --prompt-tokens 128 --output-tokens 64 --scenario-tag test
   ```
5. **Observe Metric Response**:
   - **GPU Memory Used**: Should show baseline allocation from model weights and static KV cache blocks.
   - **GPU Utilization (%)**: Should spike above idle ($0\%$) during prompt evaluation (prefill) and token generation (decode).
   - **GPU Power (W)**: Should rise above idle baseline (e.g. from $\sim 10\text{ W}$ up to $40\text{--}70\text{ W}$ on a T4).
