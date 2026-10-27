# M4 Sprint 3 Validation Log

## Injection hook availability

Searched the repository source for each proposed M2 environment variable. No implementation matches were found; these statuses describe code availability, not synthetic injector behavior.

| Injection | Environment variable | Status |
|---|---|---|
| Slow tokenizer | `LLMPROF_SLOW_TOKENIZER_MS` | pending |
| No batching | `LLMPROF_NO_BATCHING` | pending |
| CPU preprocessing | `LLMPROF_CPU_PREPROCESS_MS` | pending |

## Sprint 3 validation status

- Added evidence-tied suggested actions and conditional queue/GPU, decode-power, and KV-cache evidence rules to the analyzer.
- Added a resumable E1 runner that calls the existing load-generator CLI for live trials, starts each trial's server with a clean hook environment, and expects its JSONL trace at `LLMPROF_TRACE_PATH`.
- Added a mocked `--dry-run` that creates only labeled synthetic CSV/trace data. The local dry-run completed 9/9 injected trials with top-1 accuracy 1.0, top-3 hit rate 1.0, and no control false positive; these values are synthetic checks only.
- Synthetic scenario tests now write under pytest temporary directories. The overhead test uses the best of five samples and a generous 1% ceiling.
- E1 live injection validation is **pending_live** until M2 implements the hooks and per-request trace output.
- Synthetic data is used only by the explicit `--dry-run` path and is labeled `synthetic`; it is not evidence of live E1 behavior.
- No E2–E6 experiments or report were run as part of this sprint.
