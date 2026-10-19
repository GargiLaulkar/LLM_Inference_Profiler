# Baseline B1 Evaluation Report

## Overview
Baseline B1 establishes black-box end-to-end performance profiles across varying load and prompt scenarios.

## Summary Metrics

| Scenario | Target RPS | Mean (s) | P50 (s) | P95 (s) | P99 (s) | Throughput (req/s) | Error Rate (%) |
|---|---|---|---|---|---|---|---|
| `low_load` | 1.0 | 0.1388 | 0.1387 | 0.1400 | 0.1403 | 1.01 | 0.00% |
| `medium_load` | 5.0 | 3.2205 | 3.2196 | 3.2957 | 4.0033 | 4.41 | 0.00% |
| `rps_1` | 1.0 | 2.6717 | 2.6540 | 2.7024 | 3.2021 | 0.97 | 0.00% |

## Architectural Findings & Analysis

B1 shows only total latency; it cannot say which stage is slow. Without fine-grained stage timers decomposing tokenization, queue wait, KV cache lookup, prefill compute, iterative decode steps, detokenization, and network transmission, black-box benchmarks cannot isolate server-side bottlenecks or identify whether high tail latency stems from memory bandwidth saturation, scheduling contention, or prompt prefix processing overhead.
