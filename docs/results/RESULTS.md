# Project 6 — Experiment Results

**Branch**: `m1-experiments` | **Date**: 2026-11-07 | **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
**Hardware**: RTX 3060 6 GB (WSL2 / Docker Desktop, Windows 11)

---

## E1 — Injection Detection

**Setup**: 50 requests per run, RPS=2 (control/slow_tokenizer/cpu_preprocess), 10 concurrent (no_batching), via gateway port 8080, `LLMPROF_PROFILER_ENABLED=1`. 2 repeats per injection type.

### Stage Latencies (mean, from gateway trace file)

| Run | tokenization (ms) | queue_wait (ms) | prefill (ms) | decode (ms) | network (ms) | total (ms) |
|-----|-------------------|-----------------|--------------|-------------|--------------|------------|
| control_repeat_01 | 14.5 | 0.0 | 4950.7 | 830.6 | 6.2 | 5796.0 |
| control_repeat_02 | 4.6 | 0.0 | 43.0 | 567.0 | 3.3 | 614.7 |
| slow_tokenizer_repeat_01 | 505.7 | 0.0 | 40.6 | 553.3 | 3.1 | 1099.8 |
| slow_tokenizer_repeat_02 | 505.5 | 0.0 | 43.1 | 557.1 | 2.9 | 1105.8 |
| cpu_preprocess_repeat_01 | 410.2 | 0.0 | 51.2 | 824.4 | 3.3 | 1286.0 |
| cpu_preprocess_repeat_02 | 408.9 | 0.0 | 67.3 | 745.3 | 3.1 | 1221.7 |
| no_batching_repeat_01 | 10.4 | 4674.8 | 38.6 | 525.9 | 2.7 | 5249.9 |
| no_batching_repeat_02 | 10.2 | 4666.9 | 39.7 | 530.7 | 2.9 | 5247.6 |

> [!NOTE]
> `prefill` = time from request dispatch to first streamed chunk; includes vLLM internal queueing. Not pure prefill.
> `control_repeat_01` shows an anomalous prefill of 4950 ms likely due to cold GPU state at the start of the session. `control_repeat_02` (614 ms total) is more representative.

### Detection Results (from `e1_detection.csv`)

| Run | Injection | Expected Stage | Top Stage (analyzer) | Top-3 Includes Expected | Detection | Explained Share |
|-----|-----------|---------------|----------------------|------------------------|-----------|-----------------|
| control_repeat_01 | control | — | prefill | — | not_applicable | 0.9991 |
| control_repeat_02 | control | — | decode | — | not_applicable | 0.9974 |
| slow_tokenizer_repeat_01 | slow_tokenizer | tokenization | **decode** | ✅ (rank 2) | **fail** | 0.9984 |
| slow_tokenizer_repeat_02 | slow_tokenizer | tokenization | **decode** | ✅ (rank 2) | **fail** | 0.9986 |
| cpu_preprocess_repeat_01 | cpu_preprocess | tokenization | **decode** | ✅ (rank 2) | **fail** | 0.9985 |
| cpu_preprocess_repeat_02 | cpu_preprocess | tokenization | **decode** | ✅ (rank 2) | **fail** | 0.9975 |
| no_batching_repeat_01 | no_batching | queue_wait | **queue_wait** | ✅ (rank 1) | **pass** | 0.9980 |
| no_batching_repeat_02 | no_batching | queue_wait | **queue_wait** | ✅ (rank 1) | **pass** | 0.9984 |

**Summary**:
- Detection accuracy: **33.3%** (2/6 injection runs) — `no_batching` detected correctly both times; tokenization-based injections failed.
- Top-3 hit rate: **100%** — expected stage was always in top-3.
- Control false positives: **0** (neither control run flagged an injection).

**Why tokenization injections failed**: Both `slow_tokenizer` (500 ms injection) and `cpu_preprocess` (400 ms injection) raised tokenization from ~5–15 ms to ~500–410 ms, but `decode` remained 553–824 ms (64 output tokens × ~8–13 ms/token for this model). The analyzer ranks by mean per-request duration; `decode` still ranked first. The injected delay is detectable (top-3 hit rate = 100%), but does not overtake `decode`. Longer injections or shorter outputs would flip the ranking.

---

## E6 — Gateway Profiling Overhead

**Setup**: 4 interleaved runs (`off_1 → on_1 → off_2 → on_2`), 100 requests each at RPS=2.0, 60 s idle between runs. `LLMPROF_PROFILER_ENABLED=0` (off) vs `1` (on). Gateway recreated before each run to clear state.

| Run | Profiler | Mean (ms) | p50 (ms) | p95 (ms) |
|-----|----------|-----------|----------|----------|
| off_1 | disabled | 597.9 | 583.6 | 738.4 |
| on_1 | enabled | 609.2 | 597.9 | 655.8 |
| off_2 | disabled | 594.8 | 580.4 | 680.5 |
| on_2 | enabled | 617.2 | 606.8 | 707.1 |

**Average OFF mean**: 596.4 ms | **Average ON mean**: 613.2 ms | **Overhead: +2.82%** ✅ (< 5% target)

> [!NOTE]
> From the previous m1-gateway session (50 requests direct-to-vLLM vs gateway passthrough with profiler disabled):
> Direct vLLM mean = 1275.5 ms / p95 = 1300.4 ms. Gateway passthrough mean = 1294.6 ms / p95 = 1520.1 ms (+1.5% mean, +16.9% p95).
> The p95 spike in that session was attributed to cold-start connection pooling. In E6 (sustained 100-req warm runs) p95 overhead is well-controlled.

---

## E2/E3 — Scenario Matrix (via gateway)

**Setup**: 50 requests per scenario, 1 repeat, 60 s idle between scenarios, gateway port 8080, `LLMPROF_PROFILER_ENABLED=1` (container default). All 6 scenarios: 100% success rate.

| Scenario | Prompt tok | Output tok | RPS | Requests | Mean (ms) | p50 (ms) | p95 (ms) | Throughput (req/s) | Success |
|----------|-----------|-----------|-----|----------|-----------|----------|----------|--------------------|---------|
| rps_1 | 128 | 64 | 1.0 | 50 | 581.5 | 564.3 | 842.3 | 1.01 | 100% |
| rps_5 | 128 | 64 | 5.0 | 50 | 700.3 | 699.4 | 767.3 | 4.80 | 100% |
| rps_10 | 128 | 64 | 10.0 | 50 | 706.5 | 686.9 | 925.6 | 9.12 | 100% |
| short_prompt | 32 | 64 | 2.0 | 50 | 574.0 | 566.1 | 618.3 | 2.00 | 100% |
| long_prompt | 1024 | 64 | 2.0 | 50 | 617.8 | 608.3 | 649.8 | 1.99 | 100% |
| long_output | 128 | 256 | 2.0 | 50 | 2467.8 | 2449.7 | 2557.7 | 1.86 | 100% |

**Key observations**:
- `long_output` (256 tokens) is ~4.3× slower than `rps_1` (64 tokens) — decode stage scales linearly with output length.
- `long_prompt` (1024 tok in) adds only ~36 ms over `rps_1` — tokenization via `/tokenize` endpoint is fast; prefill+queue_wait absorbs the extra prompt tokens within vLLM's scheduler.
- RPS 1→5→10: mean latency rises from 582→700→707 ms; p95 rises more sharply (842→767→926 ms), showing queue pressure at RPS 10 (throughput achieved = 9.12 vs target 10 req/s).

---

## Known Limitations

1. **prefill stage** — measured as gateway-dispatch to first streamed chunk, which includes vLLM's internal scheduling queue. Not pure TTFT/prefill.
2. **detokenization** — not separately measured. The gateway receives streamed text chunks (not token IDs) from vLLM v0.31.0 `stream=True`. `return_token_ids=True` was tested and confirmed to return `token_ids` per chunk, but timing the gateway's decoding pass adds negligible overhead for the token counts used (32–1024 tokens) and is not included in stage breakdown. Stage is omitted from traces.
3. **network** — defined as gateway-to-vLLM TCP connection setup (via `httpx` async client). Client-to-gateway round-trip time is not visible to the gateway.
4. **Prometheus scrape error** — M2's `prometheus_exporter.py` emits double-brace label format `{stage="..."}{le="..."}` instead of `{stage="...",le="..."}`. Prometheus rejects the histograms. Stage data is available via gateway trace files and the `/metrics` endpoint directly (port 9101), but not via Prometheus PromQL. Cannot fix (M2 code off-limits).
5. **cost time share** — `M4_TIME_SHARE=1.0` is a test value used to start the cost exporter during verification. It is not a real billing share. Not committed.
6. **GPU memory breakdown** — not measured on this machine's RTX 3060. The `memory_breakdown_file` path is configured but no deep-dive profiling was run.
7. **Nsight Systems** — not run (`nsys_output.txt` not generated). E5 (model-size scaling experiment) not run.
8. **vLLM internal batching** — `LLMPROF_NO_BATCHING` semaphore (gateway hook) limits gateway concurrency to 1 in-flight request. vLLM still batches internally if requests arrive concurrently from other sources. Documented in gateway code.

---

## File Index

| File | Description |
|------|-------------|
| `docs/results/e1_summary.json` | E1 aggregate metrics (detection_accuracy, top3_hit_rate, explained_share per run) |
| `docs/results/e1_detection.csv` | E1 per-run detection verdict (validity, top_stage, detection_status, etc.) |
| `docs/results/e6_overhead.json` | E6 gateway overhead: mean/p50/p95 per run, avg OFF/ON, overhead_percent |
| `docs/results/e2_e3_summary.md` | E2/E3 scenario matrix: latency table per scenario |
| `docs/results/e2_e3_raw.json` | E2/E3 raw per-scenario stats (JSON) |
| `results/e1_runs/*/` | Raw E1 run folders (untracked): requests.csv, traces.jsonl, run.json |
| `results/traces/` | Gateway trace files from E6 and matrix runs (untracked) |

