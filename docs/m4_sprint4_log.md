# M4 Sprint 4 log

## What ran
- Added a synthetic-safe E2-E5 runner at `llm_profiler/experiments/run_experiments.py` and thin wrappers in `experiments/`.
- Added E6 overhead benchmark at `llm_profiler/experiments/run_overhead.py`.
- Added baseline comparison and chart generation utilities.
- Added M4 scenario definitions in `experiments/m4_scenarios.yaml`.
- Updated the demo script to clearly mark the default data source as synthetic unless live hooks are available.

## What remains pending live
- M2 injection hooks are not merged yet: `LLMPROF_SLOW_TOKENIZER_MS`, `LLMPROF_NO_BATCHING`, and `LLMPROF_CPU_PREPROCESS_MS`.
- M2 trace output is not yet in place: `LLMPROF_TRACE_PATH` is still required for per-stage JSONL traces.
- The live E6 on/off profiler switch is not implemented in the stack. E6 requires the gateway/server to support `LLMPROF_PROFILER_ENABLED=0|1`; off means pass-through with stage timing, trace writing, and metric ingest disabled, while on enables all three. Both runs use the same gateway and workload. Optional direct-to-vLLM versus gateway results are reported separately as gateway overhead.
- A second model sized for RTX 3060 is not confirmed, so E5 runs remain pending_live in live mode.

## Commands to run once M2 hooks and Docker are ready
```bash
# 1) Start the serving stack
python llm_profiler/experiments/start_server.py --dry-run
# or, in the project stack, use the existing Docker compose flow

# 2) Run the scenario matrix
python experiments/run_experiments.py --live --out-dir experiments/results

# 3) Run the E6 overhead benchmark
python experiments/run_overhead.py --live --out experiments/results/e6_overhead.json

# 4) Compare baselines
python experiments/compare_baselines.py --out experiments/results/baseline_comparison.json

# 5) Create charts
python experiments/make_report_charts.py --summary experiments/results/suite_summary.json --out-dir experiments/results/charts
```

The default path is synthetic-safe; live results are only valid after the hooks, trace exports, and server toggle are merged and the stack remains healthy.
