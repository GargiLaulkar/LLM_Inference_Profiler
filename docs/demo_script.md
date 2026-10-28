# Demo script (approx. 5 minutes)

This demo is intentionally run in a synthetic-safe mode by default until M2's injection hooks and trace output are merged.

1. Start the stack in a normal local environment:
   ```bash
   docker compose -f docker/docker-compose.yml up -d vllm prometheus grafana
   ```
2. Run a short synthetic or live load sample:
   ```bash
   python llm_profiler/experiments/load_generator.py --dry-run
   ```
   or, when live hooks are available, use the real CLI with a chosen scenario.
3. Open Grafana at http://localhost:3000 and show the per-stage latency panels, GPU telemetry, and cost panels.
4. Trigger an injected bottleneck by setting one of the pending M2 environment variables, such as `LLMPROF_CPU_PREPROCESS_MS=400`.
5. Run the analyzer:
   ```bash
   python experiments/run_experiments.py --dry-run
   ```
6. Show the top 3 bottlenecks and the suggested action from the analysis summary.
7. Explain that the live stack is still pending until the hooks and trace path are merged; the synthetic run is used for validation only.

Expected result: a short walkthrough of the stack, workload, metrics, injection, and analyzer output in under five minutes.
