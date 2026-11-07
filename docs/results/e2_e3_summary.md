# E2/E3 Scenario Matrix Results

Traffic routed through gateway (port 8080, `LLMPROF_PROFILER_ENABLED=1`).  
50 requests per scenario, 1 repeat, 60 s idle between scenarios.

| Scenario | Prompt tok | Output tok | RPS | Requests | Mean (ms) | p50 (ms) | p95 (ms) | Throughput (req/s) | Success |
|----------|-----------|-----------|-----|----------|-----------|----------|----------|--------------------|---------|
| rps_1 | 128 | 64 | 1.0 | 50 | 581.53 | 564.28 | 842.31 | 1.008 | 100.0% |
| rps_5 | 128 | 64 | 5.0 | 50 | 700.25 | 699.42 | 767.34 | 4.799 | 100.0% |
| rps_10 | 128 | 64 | 10.0 | 50 | 706.52 | 686.86 | 925.61 | 9.131 | 100.0% |
| short_prompt | 32 | 64 | 2.0 | 50 | 573.96 | 566.09 | 618.27 | 1.999 | 100.0% |
| long_prompt | 1024 | 64 | 2.0 | 50 | 617.77 | 608.27 | 649.79 | 1.994 | 100.0% |
| long_output | 128 | 256 | 2.0 | 50 | 2467.78 | 2449.65 | 2557.71 | 1.863 | 100.0% |

## Notes

- `rps_10` may show increased tail latency as vLLM's queue begins to fill (10 req/s > single-request throughput).
- `long_prompt` (1024 tok in) exercises tokenization and prefill+queue_wait; expect higher total latency.
- `long_output` (256 tok out) exercises the decode stage; expect higher decode share.
- Latency = end-to-end measured by load generator (send_ts to end_ts).
- Prometheus stage breakdown not queryable (M2 exporter label format bug); per-request stage data in gateway trace files.

