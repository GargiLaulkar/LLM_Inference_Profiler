# Baseline B2 Evaluation Report: Stack Built-in Metrics vs Manual B1

## Overview
Baseline B2 evaluates LLM serving performance using the stack's built-in telemetry (vLLM Prometheus histograms) compared against client-side end-to-end measurements from Baseline B1.

## Comparison Table: B1 (Manual Client-Side) vs B2 (Stack Built-In)

| Scenario | Metric Source | Mean E2E (s) | Mean TTFT (s) | Mean Queue (s) | Achieved RPS | Requests Count |
|---|---|---|---|---|---|---|
| `low_load` | **B1 (Client Manual)** | 0.1388 | N/A (Black-box) | N/A (Black-box) | 0.98 | 100 |
| `low_load` | **B2 (vLLM Built-in)** | 2.6199 | 0.1185 | 0.0000 | 0.98 | 100 |

## Architectural Analysis: What B2 Can and Cannot Tell Us

Baseline B2 provides essential visibility into server-side execution dynamics by capturing coarse-grained operational stages: request queue wait time, prefill duration through Time to First Token (TTFT), and aggregate end-to-end inference latency directly from vLLM engine histograms. However, built-in telemetry alone remains insufficient for comprehensive inference stack profiling. B2 cannot isolate tokenization or detokenization latency on the CPU, cannot quantify client-server network transmission overhead, cannot attribute GPU High Bandwidth Memory (HBM) consumption across individual lifecycle phases (weights vs. KV-cache allocation vs. dynamic activation spikes), and cannot compute monetary inference cost per 1M tokens. Isolating tail latency anomalies and identifying memory saturation bottlenecks requires dedicated stage timers (M2), GPU kernel and power collectors (M3), and offline cost analyzers (M4).
