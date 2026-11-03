# M4 Bottleneck Injection Plan

**Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**  
**Role:** M4 (Analyzer and Validation Lead)
**Status:** Proposed validation contract for M2 hooks; no injection runs have been performed.

The environment variables below are proposals, not yet implemented by M2. With no injection variable set, server behavior must remain unchanged. The analyzer must not report an injected stage as dominant during the control run.

## Scenarios and validation

### Control: no injection

- **Setting:** None of the `LLMPROF_*` variables below are set.
- **Expected top stage:** The stage actually measured as largest; `decode` is the synthetic baseline expectation.
- **Expected evidence:** Normal measured stage breakdown, with no injected delay or injection-specific dominant-stage evidence.
- **Pass:** The analyzer reports measured stage data and the `explained_share` invariant is within `[0.90, 1.10]`; no injected stage is flagged as dominant.
- **Fail:** Any injection hook changes behavior when unset, or the analyzer attributes a synthetic injected delay to a stage.

### Slow tokenizer

- **Variable:** `LLMPROF_SLOW_TOKENIZER_MS` — adds a delay in milliseconds in the tokenization stage.
- **Expected top stage:** `tokenization`.
- **Expected evidence:** Tokenization is the largest latency share; at the proposed 500 ms setting it should exceed 50% of total latency and be marked dominant.
- **Pass:** `tokenization` ranks first with `share_of_latency > 0.50`, and the suggested action recommends a faster tokenizer, caching, or off-path tokenization.
- **Fail:** Tokenization does not rank first or its share is at or below 0.50.

### No batching / queue saturation

- **Variable:** `LLMPROF_NO_BATCHING` — when set to `1`, disables batching or forces one request at a time.
- **Expected top stage:** `queue_wait`.
- **Expected evidence:** Queue wait is the largest latency share; at the proposed 800 ms equivalent delay it should exceed 60% of total latency and be marked dominant.
- **Pass:** `queue_wait` ranks first with `share_of_latency > 0.60`, and the suggested action addresses concurrency, scaling, KV cache allocation, or rate limiting.
- **Fail:** Queue wait does not rank first.
- **Open question for M1/M2:** Disabling batching may require a server start argument (for example, vLLM's maximum concurrent sequences) rather than only an environment variable. M1 and M2 must decide how `LLMPROF_NO_BATCHING=1` is implemented.

### CPU-bound preprocessing

- **Variable:** `LLMPROF_CPU_PREPROCESS_MS` — adds CPU-bound busy work in milliseconds before tokenization.
- **Expected top stage:** `tokenization`, assuming M2 attributes preprocessing time to tokenization. If M2 instruments preprocessing separately, the attributed preprocessing stage is expected to rank first instead.
- **Expected evidence:** The stage that M2 attributes the preprocessing work to is the largest latency share; at the proposed 400 ms setting it should exceed 50% of total latency and be marked dominant.
- **Pass:** Tokenization ranks first under the stated attribution assumption (or the separately instrumented preprocessing stage does), and `explained_share` remains within `[0.90, 1.10]`.
- **Fail:** The measured preprocessing delay is not reflected in the attributed stage or that stage does not rank first.

## Summary

| Scenario | Proposed setting | Expected top stage | Dominant-stage validation |
|---|---|---|---|
| Control | None of the variables set | Measured baseline (synthetic expectation: `decode`) | No injection-induced dominant stage |
| Slow tokenizer | `LLMPROF_SLOW_TOKENIZER_MS=500` | `tokenization` | Share > 0.50 |
| No batching | `LLMPROF_NO_BATCHING=1` | `queue_wait` | Share > 0.60 |
| CPU preprocessing | `LLMPROF_CPU_PREPROCESS_MS=400` | `tokenization` by assumption; otherwise M2's preprocessing stage | Share > 0.50 |
