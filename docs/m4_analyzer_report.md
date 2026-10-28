# M4 analyzer report

## 1. Problem and research question
This work evaluates whether per-stage timing and cost information can identify latency bottlenecks in LLM serving, first with synthetic contract validation and later with measured live requests.

## 2. Design and architecture
The analyzer consumes request CSVs, per-stage trace JSONL, and optional GPU/memory telemetry. It correlates requests and traces, ranks measured stages, checks explained share, and calculates GPU-time cost. Synthetic checks validate code paths only; they are not live performance evidence.

## 3. Implementation per component
- `llm_profiler/analyzer/bottleneck_analyzer.py`: ranks stages and reports explained share, evidence, and suggested actions.
- `llm_profiler/analyzer/cost_model.py`: calculates amortized cost from GPU busy time and produced tokens.
- `llm_profiler/analyzer/readers.py`: reads the request, trace, and optional telemetry contracts.
- `llm_profiler/experiments/run_injection_validation.py`: runs the local E1 harness when M2 hooks are available.
- `llm_profiler/experiments/analyze_run_dir.py`: analyzes externally supplied E1 run folders without starting the stack.
- `llm_profiler/experiments/run_experiments.py`: prepares the E2-E5 scenario matrix.
- `llm_profiler/experiments/run_overhead.py`: defines the E6 measurement modes and checks live readiness.

## 4. Experiments E1-E6
- **E1, injected-bottleneck detection:** Live measurement is pending M2 hooks and trace output. Synthetic checks are not live results.
- **E2, load level:** Live RPS comparisons are `[PENDING LIVE MEASUREMENT]`.
- **E3, prompt/output mix:** Live length comparisons are `[PENDING LIVE MEASUREMENT]`.
- **E4, batching on/off:** Live comparison is `[PENDING LIVE MEASUREMENT]` pending a reliable server-side batching control.
- **E5, model size:** Live comparison is `[PENDING LIVE MEASUREMENT]` pending a second model confirmed to fit the GPU.
- **E6, profiler overhead:** Profiler OFF means the gateway/server handles requests in pass-through mode with stage timing, trace writing, and metric ingest disabled. Profiler ON means all three are enabled. Both trials use the same gateway and load-generator workload; their difference isolates profiler cost. An optional direct-to-vLLM versus through-the-gateway comparison measures gateway overhead separately. Live comparison is `[PENDING LIVE MEASUREMENT]` pending `LLMPROF_PROFILER_ENABLED=0|1`.

## 5. Results
No synthetic performance values are reported here as results. All live result fields remain `[PENDING LIVE MEASUREMENT]` until sourced from measured result files.

### Cost per 1M output tokens
The cost formula is:

`GPU price per hour × GPU busy hours ÷ output tokens produced × 1,000,000`

`GPU busy hours` is the wall-clock request window multiplied by `time_share`. The default `time_share` is based on the union of active request intervals, so overlapping requests are not double counted. A dedicated-instance setting may use a full window share.

Measured cost per 1M output tokens: **[PENDING LIVE MEASUREMENT]**.

### Explained share and detection
Per-run explained share, top-1 accuracy, top-3 hit rate, and the control false-positive check: **[PENDING LIVE MEASUREMENT]**.

### E6 overhead
Profiler overhead and the 5% target result: **[PENDING LIVE MEASUREMENT]**. The dry-run validates output shape only and reports no performance measurements.

### Memory breakdown
Prefill versus decode memory breakdown: **[PENDING LIVE MEASUREMENT]**. Charts are skipped when the real `memory_breakdown.json` file is absent.

### B1 vs B2 vs B3
- B1, manual timing: **[PENDING LIVE MEASUREMENT]** from M1's baseline files.
- B2, stack-native metrics: **[PENDING LIVE MEASUREMENT]** from M1's B2 artifacts or Prometheus data.
- B3, profiler: **[PENDING LIVE MEASUREMENT]** from the profiler's live run files.

## 6. Limitations
- Single-GPU testing limits generalization to multi-GPU deployments.
- A small model and GPU memory constraints limit model-size comparisons.
- Localhost/gateway topology may not represent remote network latency.
- Injection conclusions depend on which stage M2 attributes each hook to.
- Cost accounting uses GPU busy time and does not represent all infrastructure costs.
- Synthetic data validates software behavior only and is not measured system evidence.
- Memory evidence came from a single Kaggle Tesla T4 run (Qwen2.5-0.5B, 128 prompt / 64 output tokens), reconstructed from exported llmprof_memory_breakdown_bytes values, not from the target RTX 3060.

## 7. Conclusion and future work
The analysis path is designed to consume measured files and preserve their evidence without fabricating missing values. Run E1-E6 after the gateway hooks, traces, profiler toggle, and required model variants are available; then replace each pending marker with values from the corresponding real result files.
