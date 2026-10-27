# M4 Analyzer & Cost Model Design Specification

**Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**  
**Role:** Member M4 (Analyzer and Validation Lead)  
**Document:** System Design & Methodological Specification  

---

## 1. Inference Cost Model Mathematics

### 1.1 Exact Cost Formula
The dollar cost per 1 million output tokens produced is defined as:

$$\text{cost\_per\_million\_tokens\_usd} = \frac{\text{gpu\_price\_per\_hour\_usd} \times \text{gpu\_hours\_used}}{\text{tokens\_produced}} \times 1{,}000{,}000$$

Where:
- `gpu_price_per_hour_usd`: Hourly rate of the GPU device loaded dynamically from `config.yaml` (default: 1.0 USD/hour for RTX 3060).
- `tokens_produced`: Sum of `output_tokens` across all successful inference requests (`status == "ok"`). Failed requests are excluded.
- `cost_per_million_total_tokens_usd`: An optional companion metric computed using $\text{total\_tokens} = \sum (\text{prompt\_tokens} + \text{output\_tokens})$ in the denominator.

### 1.2 GPU Hours Used & `time_share` Definition
$$\text{gpu\_hours\_used} = \left(\frac{\text{wall\_clock\_window\_seconds}}{3600.0}\right) \times \text{time\_share}$$

$$\text{wall\_clock\_window\_seconds} = \max(end\_ts) - \min(send\_ts) \quad \text{over successful requests}$$

The parameter **`time_share`** addresses multi-tenant and batch scheduling concurrency:
1. **Dynamic Active Time Share (Default)**:  
   $$\text{time\_share} = \frac{\text{active\_seconds}}{\text{wall\_clock\_window\_seconds}}$$
   Where `active_seconds` is the **union of all request intervals** $[send\_ts_i, end\_ts_i]$.
   - **Overlapping requests are not double counted**: If $N$ requests run in parallel over the same 10-second window, the union is 10 seconds, not $10 \times N$ seconds.
   - **Idle periods are discounted**: If the server is idle between request bursts, `active_seconds` only accounts for periods where at least one request was in flight.
2. **Dedicated Instance Flag (`time_share = 1.0`)**:  
   When `--full-time-share` is enabled, `time_share = 1.0`. This represents dedicated host billing (e.g., reserving a cloud GPU instance for an entire benchmark window regardless of request gaps).

---

## 2. Bottleneck Ranking Rules & Dominant-Stage Threshold

### 2.1 Ranking Priority
1. **Primary Key**: Share of total request latency ($\text{share\_of\_latency}$ descending):
   $$\text{share\_of\_latency}(s) = \frac{\sum_{i=1}^N \text{duration}(s)_i}{\sum_{i=1}^N \text{total\_latency\_s}_i}$$
2. **Secondary Evidence**: GPU utilization / power draw (from Prometheus `llmprof_gpu_*` metrics) and memory component attribution (from `memory_breakdown.json`).

### 2.2 Dominant-Stage Threshold
- **Threshold**: **`50.0%`** (`DOMINANT_STAGE_THRESHOLD = 0.50`).
- **Rule**: If a single stage accounts for $\ge 50\%$ of total end-to-end latency, it is flagged as the **`DOMINANT BOTTLENECK`**.
- **System Implication (Amdahl's Law)**: Optimizing non-dominant stages (e.g. optimizing tokenization when decode is 80%) yields negligible user-perceived improvement. Remediation must focus aggressively on the dominant stage.

---

## 3. Consistency Validation: `explained_share` Invariant

$$\text{explained\_share} = \frac{\sum_{s \in \text{stages}} \sum_{i=1}^N \text{duration}(s)_i}{\sum_{i=1}^N \text{total\_latency\_s}_i}$$

### Invariant Bounds & Warning Rule:
- **Nominal Range**: **`0.90 <= explained_share <= 1.10`**
- **Warning Trigger**: If `explained_share < 0.90` or `explained_share > 1.10`, the Analyzer emits an explicit warning in the output payload:
  ```json
  "warning": "Warning: explained_share 0.4500 is outside nominal bounds [0.90, 1.10]. Stage timings do not sum to total request latency."
  ```
- **Engineering Justification**:
  - `explained_share < 0.90`: Indicates substantial uninstrumented latency (e.g., framework Python overhead, network socket delays, or missing per-stage traces).
  - `explained_share > 1.10`: Indicates overlapping sub-stage measurements or clock synchronization skew.
  - In all cases, numbers are reported as measured; data is never silently normalized or invented.

---

## 4. Stage-to-Action Remediation Reference Table

| Canonical Stage | Primary Root Cause | Domain-Specific Suggested Serving Action |
|---|---|---|
| **`prefill`** | High prompt token count, quadratic self-attention matrix multiplication | Enable chunked prefill (`--enable-chunked-prefill`), utilize prompt prefix caching (`--enable-prefix-caching`), optimize prompt length, or increase tensor parallelism. |
| **`decode`** | Autoregressive memory bandwidth saturation (loading full model weights per token) | Enable speculative decoding with a draft model, optimize KV cache block size, raise tensor parallelism, or restrict generation `max_tokens`. |
| **`queue_wait`** | High request arrival rate (RPS) exceeding server execution concurrency | Increase server concurrency limit (`--max-num-seqs`), scale out server replicas behind a load balancer, optimize KV cache memory allocation, or apply client rate limiting. |
| **`tokenization`** | CPU-bound string tokenization and vocabulary lookup serialization | Switch to a fast Rust-based HuggingFace tokenizer, cache pre-tokenized inputs on client/worker, or offload tokenization off the serving critical path. |
| **`detokenization`** | Serial byte-to-UTF8 decoding and string concatenation | Stream incremental tokens directly to client, batch UTF-8 detokenization passes, or offload detokenization to client. |
| **`network`** | Client-to-server HTTP round-trip latency, TCP handshake, or transmission jitter | Colocate client and server in same VPC/subnet, enable persistent HTTP keep-alive connection pooling, or migrate to gRPC / HTTP/2 transport. |
| **`total`** | Cumulative end-to-end latency | Investigate cross-stage pipeline concurrency and system bottlenecks. |

---

## 5. Input Sources, Fallbacks & Missing Data Handling

The Analyzer handles telemetry inputs with graceful degradation:

| Input Source | Primary Artifact | Purpose | Fallback Behavior When Missing |
|---|---|---|---|
| **Request Log CSV** | `results/*.csv` | End-to-end request latencies, token counts, status, scenario tag | **Required**. If missing, Analyzer raises `FileNotFoundError`. |
| **Per-Stage Traces** | `results/*_traces.jsonl` | Microsecond timestamps and durations for individual stages | **Optional**. If missing, Analyzer reports `missing_stage_traces` bottleneck, sets `explained_share = 0.0`, calculates cost from CSV, and suggests enabling M2 exporter. |
| **Memory Breakdown** | `results/deep_dive/memory_breakdown.json` | VRAM breakdown (`weights`, `kv_cache`, `activations`, `other`) | **Optional**. When absent, evidence text explicitly states `"memory evidence unavailable"`. If present, validates `phases` (`prefill`, `decode`) and reports schema mismatches clearly. |
| **GPU Metrics** | Prometheus `/metrics` | Instantaneous GPU utilization, power draw, and memory usage | **Optional**. When absent, GPU evidence is omitted; when present, GPU utilization % and power (W) are incorporated into bottleneck evidence. |
