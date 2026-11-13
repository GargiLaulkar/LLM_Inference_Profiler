# Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks

## Project Overview
This project provides an automated, non-invasive profiling and observability framework for production LLM serving stacks. It decomposes end-to-end inference latency into fine-grained canonical stages (`tokenization`, `queue_wait`, `prefill`, `decode`, `network`), measures hardware resource utilization and power draw, computes token-level serving costs, and automatically attributes system bottlenecks under synthetic and real-world workloads.

## System Architecture
1. **Load Generator**: Sends open-loop HTTP inference requests with deterministic prompt/generation shapes and `X-Request-ID` headers.
2. **Gateway (`llmprof_gateway:8080`)**: Reverse-proxy stage timer measuring per-request stage latencies, generating JSONL traces, and posting durations.
3. **vLLM Engine (`llmprof_vllm:8000`)**: OpenAI-compatible model server executing GPU-accelerated inference (`Qwen/Qwen2.5-0.5B-Instruct`).
4. **Profiler Exporter (`llmprof_profiler_exporter:9101`)**: Prometheus metrics server aggregating histogram latencies and cumulative token counters.
5. **GPU Collector (`llmprof_gpu_collector:9102`)**: NVML background exporter gathering GPU core utilization, VRAM allocation, and wattage.
6. **Cost Exporter (`llmprof_m4_cost_exporter:9103`)**: Prometheus exporter translating compute consumption into dollar cost per million tokens.
7. **Prometheus (`llmprof_prometheus:9090`)**: Scrapes and stores metrics across all exporters at 5-second intervals.
8. **Grafana (`llmprof_grafana:3000`) & Offline Analyzer**: Visualizes real-time performance and attributes bottleneck stages from run artifacts.

## Stack Deployment & Verification
To bring up the entire serving and telemetry stack:
```bash
# Set cost model time share and launch all services with gateway and m4 profiles
$env:M4_TIME_SHARE="1.0"
docker compose -f docker/docker-compose.yml --profile gateway --profile m4 up -d
```

To run a single benchmark load test through the gateway:
```bash
python experiments/load_generator.py --config config_gateway.yaml --rps 2 --num-requests 50 --scenario-tag quick_test
```

## Headline Experimental Results
- **Gateway Profiling Overhead**: **+2.82%** (Mean OFF: 596.36 ms vs. Mean ON: 613.19 ms; target < 5.0%)  
  *Source: `docs/results/e6_overhead.json`*
- **Gateway vs. Direct vLLM Overhead**: Direct mean = 1275.49 ms (p95 = 1300.40 ms) vs. Gateway mean = 1294.61 ms (p95 = 1520.12 ms) (+1.50% mean)  
  *Source: `docs/results/RESULTS.md`*
- **Bottleneck Detection Accuracy**:
  - `e1` (64 output tokens): **33.3%** detection accuracy, **100.0%** top-3 hit rate (decode stage dominated 500 ms tokenization delays).  
    *Source: `docs/results/e1_summary.json`*
  - `e1b` (16 output tokens): **100.0%** detection accuracy on valid runs, **100.0%** top-3 hit rate, **0** control false positives.  
    *Source: `docs/results/e1b/e1_summary.json`*
- **Scenario Matrix Throughput & Tail Latency** (*Source: `docs/results/e2_e3_raw.json`*):
  - `rps_1`: Mean 581.53 ms, p95 842.31 ms, throughput 1.01 req/s, 100% success
  - `rps_5`: Mean 700.25 ms, p95 767.34 ms, throughput 4.80 req/s, 100% success
  - `rps_10`: Mean 706.52 ms, p95 925.61 ms, throughput 9.12 req/s (indicates saturation under 10.0 target RPS)
  - `short_prompt` (32 in, 64 out): Mean 573.96 ms, p95 618.27 ms
  - `long_prompt` (1024 in, 64 out): Mean 617.77 ms, p95 649.79 ms
  - `long_output` (128 in, 256 out): Mean 2467.78 ms, p95 2557.71 ms (decode-bound scaling)

## Known Limitations
1. **Prefill Timing**: Time to first streamed chunk includes vLLM internal scheduling queue delay and batch construction; not isolated GPU prefill.
2. **Detokenization**: Completion chunk tokens are not separately timed from streamed SSE generation; detokenization is omitted from traces.
3. **Tokenization Endpoint**: Timed via vLLM's `/tokenize` HTTP endpoint and local tokenizer fallback.
4. **Client Network Scope**: Network stage measures gateway-to-vLLM round trip; client-to-gateway network time is unmeasured by the proxy.
5. **Cost Allocation**: Cost model utilizes an illustrative single-tenant test share (`M4_TIME_SHARE=1.0`); no production multi-tenant accounting was applied.
6. **GPU Memory Calibration**: Memory breakdown values originate from Tesla T4 profiling data rather than the local RTX 3060.
7. **Nsight Profiling**: CUDA kernel profiling via NVIDIA Nsight Systems was omitted on this environment (`nsys_output.txt` not collected).
8. **Model Scaling**: Model size scaling tests (E5) were constrained by the 6 GB physical VRAM boundary.
9. **Evaluation Dataset**: Experiments evaluated a single small model (`Qwen/Qwen2.5-0.5B-Instruct`) driven by synthetic prompt distributions.

