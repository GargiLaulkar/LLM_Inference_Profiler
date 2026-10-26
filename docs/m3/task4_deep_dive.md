# Task 4 — Deep-Dive Mode: Single-Request GPU Profiling

**Role**: M3 (GPU and Deep-Dive Lead)
**Project**: 6 — Inference Cost and Latency Profiler for LLM Serving Stacks
**Week**: 4 of 5

---

## Overview

Deep-dive mode profiles **one** synthetic request in detail, measuring per-stage wall-clock and GPU time for the four contract stages: `tokenization → prefill → decode → detokenization`. It produces:

| File | Contents |
|------|----------|
| `<req_id>_stages.jsonl` | Per-stage timing in contract schema (joined by M2) |
| `<req_id>_summary.json` | Aggregate statistics (total_s, mean/p95 decode token time) |
| `<req_id>_trace.json` *(torch mode)* | Chrome-format trace (open in Perfetto UI) |
| `<req_id>_kernels.txt` *(torch mode)* | Kernel table sorted by CUDA time |
| `<req_id>_nsys.nsys-rep` *(nsys mode)* | Nsight Systems report (open in GUI) |

> [!IMPORTANT]
> Deep-dive runs **separately** from the always-on Prometheus metrics collected by `gpu_collector.py`. Never run both at the same time: profiling adds significant overhead and would distort benchmark numbers.

---

## Why Deep-Dive Is Separate from Always-On Metrics

The always-on stack (Tasks 2 & 3) uses **NVML polling** every second — zero overhead to the model itself. Deep-dive inserts timing probes and, optionally, activates the CUPTI profiling interface. This causes:

1. **10–40 % runtime overhead** per forward pass (CUPTI kernel tracing).
2. **VRAM inflation** when `profile_memory=True` is active.
3. **Startup latency** of 1–2 s for CUPTI initialization.

Running both simultaneously would make latency metrics meaningless.

---

## Why `torch.profiler` and Nsight Systems Cannot Run Together

Both use the CUDA Profiling Tools Interface (**CUPTI**). CUPTI allows only **one subscriber** at a time. Attempting to activate both causes:

```
CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
```

This is a hard NVIDIA limitation — not a bug in the code. The two modes are therefore mutually exclusive and selected at launch via `--mode {torch,nsys}`.

---

## Files

| File | Purpose |
|------|---------|
| `llm_profiler/collectors/deep_dive.py` | Main profiling script (standalone) |
| `llm_profiler/collectors/deep_dive_nsys.sh` | Bash wrapper: `nsys profile … python deep_dive.py --mode nsys` |
| `llm_profiler/collectors/deep_dive_nsys.ps1` | PowerShell equivalent (Windows) |
| `tests/test_deep_dive.py` | Unit tests (torch-free where possible) |

---

## How to Run

### Prerequisites

```bash
# Stop vLLM container if running (frees GPU VRAM and CUPTI)
make down
# or:
docker compose -f docker/docker-compose.yml down
```

### Mode 1 — `torch.profiler` (recommended for kernel analysis)

```bash
# From repo root, with venv activated:
python llm_profiler/collectors/deep_dive.py \
    --mode torch \
    --prompt-tokens 128 \
    --output-tokens 64 \
    --warmup 1 \
    --out-dir results/deep_dive
```

Outputs → `results/deep_dive/dd-<epoch>_{trace.json,kernels.txt,stages.jsonl,summary.json}`

> [!TIP]
> On Kaggle (no nsys): use `--mode torch` only. The notebook can call the script directly:
> ```python
> import subprocess, sys
> subprocess.run([sys.executable, "deep_dive.py",
>                 "--mode", "torch", "--prompt-tokens", "128", "--output-tokens", "64"])
> ```

### Mode 2 — Nsight Systems (`nsys`) — RTX 3060 / M1's machine

```bash
# On Linux with nsys installed (M1's RTX 3060 machine):
bash llm_profiler/collectors/deep_dive_nsys.sh \
    --prompt-tokens 128 \
    --output-tokens 64
```

Or on Windows (M1's machine, PowerShell):

```powershell
.\llm_profiler\collectors\deep_dive_nsys.ps1 `
    --prompt-tokens 128 `
    --output-tokens 64
```

> [!WARNING]
> Nsight Systems requires `perf_event_paranoid <= 1` on Linux:
> ```bash
> echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
> ```
> Kaggle notebooks do **not** have `nsys` installed and cannot run mode 2.

---

## How to Read the Chrome Trace (torch mode)

1. Open **https://ui.perfetto.dev** in Chrome/Edge.
2. Click **Open trace file** → select `<req_id>_trace.json`.
3. In the timeline, look for the thread named after the Python process.
4. Expand the **record_function** spans — you will see:
   - `tokenization` (typically < 1 ms on CPU)
   - `prefill` (largest CUDA burst — all prompt tokens in one forward pass)
   - `decode` (many short CUDA bursts — one per output token)
   - `detokenization` (CPU only, typically < 1 ms)
5. Click any CUDA kernel under `prefill` to see its name and duration.

---

## How to Read the Kernel Table (`kernels.txt`)

```
Key averages (grouped by operator name):
Name                  Self CPU %   Self CPU    CPU total   CUDA total  ...
--------------------  ----------  ----------  ----------  ----------  ---
aten::mm              ...         ...         ...         18.4 ms     ...
aten::scaled_dot_...  ...         ...         ...         12.1 ms     ...
```

The **`cuda_time_total`** column shows which CUDA kernels dominate. On T4/RTX 3060:
- `aten::mm` (GEMM) dominates prefill for large prompts.
- `aten::scaled_dot_product_attention` or cuBLAS batched GEMM dominates decode.

Share this file with M4 to identify bottleneck operators.

---

## How M2 Joins the Stage Timing Data

The `_stages.jsonl` file uses the **shared contract schema**:

```jsonl
{"request_id": "dd-1700000000", "stage": "tokenization", "start_ts": 1700000000.1, "end_ts": 1700000000.11, "duration_s": 0.010}
{"request_id": "dd-1700000000", "stage": "prefill",      "start_ts": 1700000000.11, "end_ts": 1700000000.41, "duration_s": 0.300}
{"request_id": "dd-1700000000", "stage": "decode",       "start_ts": 1700000000.41, "end_ts": 1700000000.91, "duration_s": 0.500}
{"request_id": "dd-1700000000", "stage": "detokenization","start_ts": 1700000000.91, "end_ts": 1700000000.92, "duration_s": 0.010}
```

M2 can join by `request_id` field. The `start_ts` and `end_ts` are **epoch seconds** (float), compatible with Prometheus timestamps.

---

## Nsight Systems Permissions (Linux)

| Setting | Required value | How to set |
|---------|---------------|------------|
| `perf_event_paranoid` | `<= 1` | `echo 1 \| sudo tee /proc/sys/kernel/perf_event_paranoid` |
| `nsys` on PATH | yes | Add NVIDIA Nsight Systems `bin/` to `~/.bashrc` |
| CUPTI library | bundled with CUDA toolkit | Ensure CUDA toolkit is installed |

Kaggle notebooks run as `root` in a container — `perf_event_paranoid` is irrelevant there, but **nsys is not pre-installed on Kaggle**. Use `--mode torch` on Kaggle.

---

## Manual Steps for M1 (RTX 3060 Machine, nsys mode)

1. SSH into M1's machine (or sit at it directly).
2. Stop the vLLM container:
   ```bash
   make down
   ```
3. Activate the project venv:
   ```bash
   source .venv/bin/activate
   ```
4. Run the Nsight wrapper:
   ```bash
   bash llm_profiler/collectors/deep_dive_nsys.sh \
       --prompt-tokens 128 \
       --output-tokens 64
   ```
5. Copy the `.nsys-rep` file to your machine and open it in **Nsight Systems GUI**:
   - File → Open → `results/deep_dive/dd-<epoch>_nsys.nsys-rep`
   - In the timeline: look at the **NVTX** row.
   - Coloured ranges show `tokenization`, `prefill`, `decode`, `detokenization`.
   - Under each range: CUDA kernels show real GPU execution time.

---

## Design Decisions

### Lazy imports (`torch`, `transformers`)

`import torch` and `from transformers import …` are inside functions, not at module level. This means:
- `python deep_dive.py --help` works on the dev laptop with **no torch installed**.
- Unit tests for `build_prompt`, `write_stage_jsonl`, and `summarize` run without torch.
- Only tests that call `run_request` are skipped via `pytest.importorskip("torch")`.

### Zero intra-repo dependencies

`deep_dive.py` imports nothing from the rest of `llm_profiler/`. It can be copy-pasted into a Kaggle notebook as a single cell.

### `torch.cuda.synchronize()` placement

GPU operations are asynchronous. Without synchronization, `time.perf_counter()` measures only the time to *enqueue* work, not to *execute* it. The `stage()` context manager calls `synchronize()` **before** reading the end timestamp for `prefill` and `decode` (the two CUDA-heavy stages).

### pynvml vs DCGM (for always-on collection)

| | pynvml | DCGM |
|--|--------|------|
| Dependency | `nvidia-ml-py` (pip) | NVIDIA Data Center GPU Manager (system install) |
| Access | User-space NVML | Privileged daemon |
| Use case | Development, single-GPU | Production, multi-GPU datacenter |
| Our choice | **pynvml** | Future enhancement |

For Task 4 deep-dive, `torch.profiler` provides kernel-level timing that NVML cannot. NVML is used only in the always-on `gpu_collector.py` (Task 2).

---

## Results Table (fill in after Kaggle run)

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Device | Tesla T4 (Kaggle) |
| Prompt tokens | 128 |
| Output tokens | 64 |
| Warmup runs | 1 |
| Mode | torch |

| Stage | Duration (ms) |
|-------|--------------|
| tokenization | ___ |
| prefill | ___ |
| decode (total) | ___ |
| detokenization | ___ |
| **TOTAL** | ___ |

| Metric | Value |
|--------|-------|
| Mean decode token time (ms) | ___ |
| p95 decode token time (ms) | ___ |
| Top-3 CUDA kernels (from kernels.txt) | ___ |
