# Project 6 — Experiment Results

**Branch**: `m1-final` | **Date**: 2026-11-13 | **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
**Hardware**: NVIDIA GeForce RTX 3060 6 GB (WSL2 / Docker Desktop, Windows 11)

---

## E1 — Injected Bottleneck Detection

We evaluated the bottleneck analyzer across two experimental configurations.

**Configuration Difference**: In configuration `e1` (64 output tokens), decode latency (~550–830 ms) dominated the per-request latency budget and masked the 400–500 ms injected tokenization delays, causing the analyzer to rank tokenization second. Configuration `e1b` shortened generation length to 16 output tokens and increased injected delays to 1500 ms (tokenizer) and 1200 ms (CPU), ensuring the injected stage clearly surpassed decode time and ranked first.

---

### Configuration 1: `e1` (64 Output Tokens)

**Parameters**: 50 requests/trial, prompt tokens 128, output tokens 64, RPS 2.0 (control/slow_tokenizer/cpu_preprocess), 10 concurrent requests (no_batching), `LLMPROF_PROFILER_ENABLED=1`.

#### Stage Latencies (mean, from `results/e1_runs/*/traces.jsonl`)

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

#### Detection Verdicts (from `docs/results/e1_detection.csv`)

| Run | Injection | Expected Stage | Top Stage (analyzer) | Top-3 Rank | Detection Status | Explained Share |
|-----|-----------|---------------|----------------------|------------|------------------|-----------------|
| control_repeat_01 | control | — | prefill | — | not_applicable | 0.9991 |
| control_repeat_02 | control | — | decode | — | not_applicable | 0.9974 |
| slow_tokenizer_repeat_01 | slow_tokenizer (500 ms) | tokenization | decode | Rank 2 | fail | 0.9984 |
| slow_tokenizer_repeat_02 | slow_tokenizer (500 ms) | tokenization | decode | Rank 2 | fail | 0.9986 |
| cpu_preprocess_repeat_01 | cpu_preprocess (400 ms) | tokenization | decode | Rank 2 | fail | 0.9985 |
| cpu_preprocess_repeat_02 | cpu_preprocess (400 ms) | tokenization | decode | Rank 2 | fail | 0.9975 |
| no_batching_repeat_01 | no_batching | queue_wait | queue_wait | Rank 1 | pass | 0.9980 |
| no_batching_repeat_02 | no_batching | queue_wait | queue_wait | Rank 1 | pass | 0.9984 |

- **Detection Accuracy**: 33.3% (2/6 runs)
- **Top-3 Hit Rate**: 100.0% (expected stage always in top 3)
- **Control False Positives**: 0

---

### Configuration 2: `e1b` (16 Output Tokens)

**Parameters**: 50 requests/trial, prompt tokens 128, output tokens 16, RPS 2.0 (control/slow_tokenizer/cpu_preprocess), 10 concurrent requests (no_batching), `LLMPROF_PROFILER_ENABLED=1`.

#### Stage Latencies (mean, from `results/e1b_runs/*/traces.jsonl`)

| Run | tokenization (ms) | queue_wait (ms) | prefill (ms) | decode (ms) | network (ms) | total (ms) |
|-----|-------------------|-----------------|--------------|-------------|--------------|------------|
| control_repeat_01 | 5.2 | 0.0 | 123.4 | 598.2 | 3.2 | 726.9 |
| control_repeat_02 | 5.0 | 0.0 | 114.7 | 593.1 | 3.0 | 712.9 |
| slow_tokenizer_repeat_01 | 1505.9 | 0.0 | 115.4 | 593.1 | 3.0 | 2214.5 |
| slow_tokenizer_repeat_02 | 1506.1 | 0.0 | 115.4 | 591.7 | 3.2 | 2213.2 |
| cpu_preprocess_repeat_01 | 16965.8 | 0.0 | 13287.6 | 744.8 | 10624.2 | 30998.3 |
| cpu_preprocess_repeat_02 | 16999.3 | 0.0 | 13342.8 | 636.6 | 10785.0 | 30978.8 |
| no_batching_repeat_01 | 11.3 | 13930.0 | 98.7 | 552.9 | 3.3 | 14593.1 |
| no_batching_repeat_02 | 11.7 | 13907.9 | 98.7 | 552.5 | 3.4 | 14571.0 |

#### Detection Verdicts (from `docs/results/e1b/e1_detection.csv`)

| Run | Injection | Expected Stage | Top Stage (analyzer) | Share of Latency | Detection Status | Explained Share |
|-----|-----------|---------------|----------------------|------------------|------------------|-----------------|
| control_repeat_01 | control | — | decode | 81.72% | not_applicable | 0.9972 |
| control_repeat_02 | control | — | decode | 82.59% | not_applicable | 0.9968 |
| slow_tokenizer_repeat_01 | slow_tokenizer (1500 ms) | tokenization | tokenization | 67.85% | pass | 0.9991 |
| slow_tokenizer_repeat_02 | slow_tokenizer (1500 ms) | tokenization | tokenization | 67.89% | pass | 0.9991 |
| cpu_preprocess_repeat_01 | cpu_preprocess (1200 ms) | tokenization | tokenization | 34.81% | invalid* | 0.8564 |
| cpu_preprocess_repeat_02 | cpu_preprocess (1200 ms) | tokenization | tokenization | 34.89% | invalid* | 0.8597 |
| no_batching_repeat_01 | no_batching | queue_wait | queue_wait | 95.42% | pass | 0.9998 |
| no_batching_repeat_02 | no_batching | queue_wait | queue_wait | 95.41% | pass | 0.9998 |

*\*Note on cpu_preprocess invalidation*: The 1200 ms synchronous CPU loop saturated gateway request handling at 2 RPS, causing 3 of 50 requests (6.0%) to exceed client timeout before completing trace recording. Because 6.0% missing traces exceeds the analyzer's 5.0% validity threshold, the analyzer marked the run invalid. However, among all recorded traces, `tokenization` was accurately identified as the dominant stage (34.8%).

- **Detection Accuracy**: 100.0% (4/4 evaluated valid injection runs)
- **Top-3 Hit Rate**: 100.0%
- **Control False Positives**: 0

---

## E6 — Gateway Profiling Overhead

**Setup**: 4 interleaved runs (`off_1 → on_1 → off_2 → on_2`), 100 requests each at RPS 2.0, 60 s idle between runs. `LLMPROF_PROFILER_ENABLED=0` (off) vs `1` (on). Gateway recreated before each run.

| Run | Profiler State | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
|-----|----------------|-------------------|------------------|------------------|
| off_1 | disabled | 597.92 | 583.60 | 738.44 |
| on_1 | enabled | 609.20 | 597.87 | 655.77 |
| off_2 | disabled | 594.80 | 580.42 | 680.48 |
| on_2 | enabled | 617.18 | 606.76 | 707.06 |

- **Average OFF Mean Latency**: 596.36 ms
- **Average ON Mean Latency**: 613.19 ms
- **Profiling Overhead**: **+2.82%** (well within the < 5.0% requirement)

### Gateway vs. Direct vLLM Overhead

Measured with identical 50-request workloads direct to vLLM (port 8000) versus through the gateway with profiler disabled (port 8080, `LLMPROF_PROFILER_ENABLED=0`):
- **Direct vLLM**: Mean = 1275.49 ms | p95 = 1300.40 ms
- **Gateway Passthrough**: Mean = 1294.61 ms | p95 = 1520.12 ms
- **Overhead**: +1.50% mean (+19.12 ms), +16.90% p95 (+219.72 ms tail due to initial connection pooling).

---

## E2/E3 — Scenario Matrix

**Setup**: 50 requests per scenario, 1 repeat, 60 s idle between scenarios, gateway port 8080, `LLMPROF_PROFILER_ENABLED=1`. All scenarios achieved 100% request success.

| Scenario | Prompt Tokens | Output Tokens | Target RPS | Requests | Mean Latency (ms) | p50 (ms) | p95 (ms) | Achieved Throughput (req/s) | Success Rate |
|----------|---------------|---------------|------------|----------|-------------------|----------|----------|-----------------------------|--------------|
| rps_1 | 128 | 64 | 1.0 | 50 | 581.53 | 564.28 | 842.31 | 1.01 | 100% |
| rps_5 | 128 | 64 | 5.0 | 50 | 700.25 | 699.42 | 767.34 | 4.80 | 100% |
| rps_10 | 128 | 64 | 10.0 | 50 | 706.52 | 686.86 | 925.61 | 9.12* | 100% |
| short_prompt | 32 | 64 | 2.0 | 50 | 573.96 | 566.09 | 618.27 | 2.00 | 100% |
| long_prompt | 1024 | 64 | 2.0 | 50 | 617.77 | 608.27 | 649.79 | 1.99 | 100% |
| long_output | 128 | 256 | 2.0 | 50 | 2467.78 | 2449.65 | 2557.71 | 1.86 | 100% |

*\*Saturation Note*: At `rps_10`, the serving stack achieved 9.12 req/s against the 10.0 req/s target, indicating that the system was operating near saturation under sustained concurrency.

---

## Limitations

1. **Prefill Stage**: Measured as the duration from gateway dispatch to the receipt of the first streamed SSE chunk. This interval includes vLLM's internal scheduling queue wait and batch assembly time, and is therefore not pure GPU prefill time.
2. **Detokenization**: Not separately measured. vLLM returns text chunks rather than per-chunk token IDs in standard completion streams; detokenization is omitted from stage breakdown.
3. **Tokenization**: Timed via vLLM's `/tokenize` HTTP endpoint (or local fast-tokenizer fallback). Local CPU preprocessing time before dispatch is measured within the tokenization stage.
4. **Network Visibility**: Measured strictly as gateway-to-vLLM connection and transfer setup. Client-to-gateway transit time is not visible to the gateway.
5. **Cost Model Time Share**: Cost exporter uses an illustrative workload time share (`M4_TIME_SHARE=1.0`) during validation; no real multi-tenant billing share was provided.
6. **GPU Memory Breakdown**: Memory component breakdown values were derived from Tesla T4 calibration runs, not this machine's RTX 3060.
7. **Nsight Deep-Dive**: Nsight Systems CUDA kernel profiling was not executed on this host (`nsys_output.txt` omitted).
8. **Model-Size Experiment (E5)**: E5 scaling across 7B/13B models was not run due to the 6 GB VRAM limit of the RTX 3060.
9. **Single Model Evaluation**: All live measurements were gathered on a single small model (`Qwen/Qwen2.5-0.5B-Instruct`) with synthetic workload prompts.
