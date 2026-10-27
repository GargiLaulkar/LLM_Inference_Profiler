# M4 Bottleneck Injection Plan (Answer Key for Member 2)

**Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**  
**Role:** Member M4 (Analyzer and Validation Lead) &rarr; Member M2 (Instrumentation Lead)  
**Document:** Bottleneck Injection Answer Key and Verification Matrix  

---

## 1. Overview
This document serves as the formal **Answer Key** provided by Member 4 to Member 2. It specifies synthetic and server-side latency injection scenarios used to validate that stage instrumentation and the offline Bottleneck Analyzer accurately pinpoint serving anomalies.

For each scenario, this document defines:
1. The proposed configuration flag name for `config.yaml` or CLI arguments.
2. The expected Rank #1 bottleneck stage.
3. The expected empirical evidence emitted by the Analyzer.
4. The strict Pass/Fail validation rule.

---

## 2. Injection Scenarios & Answer Key

### Scenario 0: No-Injection Control Baseline
- **Description**: Standard reference inference serving with zero injected faults. Serves as the ground-truth control.
- **Proposed Config Key**:
  ```yaml
  injection:
    enabled: false
    scenario: "none"
  ```
- **Expected Top Stage**: **`decode`** (for standard conversational workloads with 128 prompt / 64 output tokens on RTX 3060).
- **Expected Evidence**:
  `'decode' accounts for 60%–70% of total latency (mean: ~0.080s across requests). [memory evidence unavailable] [DOMINANT BOTTLENECK >= 50%]`
- **Pass/Fail Rule**:
  - **PASS**: `explained_share` is between `0.90` and `1.10`; Rank #1 stage is `decode` (or `prefill` if prompt > 512 tokens); no artificial queue or tokenization delays detected.
  - **FAIL**: Uninstrumented stage dominates, or `explained_share` falls outside `[0.90, 1.10]`.

---

### Scenario 1: Slow Tokenizer Anomaly
- **Description**: Simulates a heavy, unoptimized, or uncompiled pure-Python tokenizer performing quadratic vocabulary lookups or un-cached regex tokenization.
- **Proposed Config Key**:
  ```yaml
  injection:
    enabled: true
    scenario: "slow_tokenizer"
    delay_s: 0.50
    target_stage: "tokenization"
  ```
- **Expected Top Stage**: **`tokenization`**
- **Expected Evidence**:
  `'tokenization' accounts for >65% of total latency (mean: ~0.501s across requests). [memory evidence unavailable] [DOMINANT BOTTLENECK >= 50%]`
- **Pass/Fail Rule**:
  - **PASS**: `tokenization` is ranked **Rank #1** with $\text{share\_of\_latency} > 0.50$; Analyzer suggested action prescribes: *"Switch to a fast Rust-based HuggingFace tokenizer, cache tokenized prompts on client/worker, or offload tokenization off the serving critical path."*
  - **FAIL**: `tokenization` is ranked lower than Rank #1 or share < 0.50.

---

### Scenario 2: No Batching / Concurrency Saturation (`no_batching_queue_heavy`)
- **Description**: Simulates serving without continuous batching or with `--max-num-seqs 1`, forcing incoming requests into an accumulating FIFO wait queue.
- **Proposed Config Key**:
  ```yaml
  injection:
    enabled: true
    scenario: "no_batching"
    delay_s: 0.80
    target_stage: "queue_wait"
  ```
- **Expected Top Stage**: **`queue_wait`**
- **Expected Evidence**:
  `'queue_wait' accounts for >70% of total latency (mean: ~0.812s across requests). [memory evidence unavailable] [DOMINANT BOTTLENECK >= 50%]`
- **Pass/Fail Rule**:
  - **PASS**: `queue_wait` is ranked **Rank #1** with $\text{share\_of\_latency} > 0.60$; Analyzer suggested action prescribes: *"Increase serving concurrency (--max-num-seqs), scale out server replicas behind a load balancer, optimize KV cache memory allocation, or apply client rate limiting."*
  - **FAIL**: `queue_wait` is ranked lower than Rank #1.

---

### Scenario 3: CPU-Bound Preprocessing
- **Description**: Simulates computationally heavy CPU preprocessing tasks (e.g., synchronous document normalization, JSON formatting, or content moderation) preceding tokenization and inference.
- **Proposed Config Key**:
  ```yaml
  injection:
    enabled: true
    scenario: "cpu_preprocessing"
    delay_s: 0.40
    target_stage: "tokenization"
  ```
- **Expected Top Stage**: **`tokenization`**
- **Expected Evidence**:
  `'tokenization' accounts for >55% of total latency (mean: ~0.401s across requests). [memory evidence unavailable] [DOMINANT BOTTLENECK >= 50%]`
- **Pass/Fail Rule**:
  - **PASS**: `tokenization` is ranked **Rank #1**; Analyzer flags the stage as dominant; `explained_share` remains within nominal bounds `[0.90, 1.10]`.
  - **FAIL**: Stage time fails to correlate with total latency increase or fails to elevate to Rank #1.

---

## 3. Verification Protocol Summary Table

| Scenario | Injected Delay | Target Stage | Expected Rank 1 | Dominant Check | Expected Explained Share |
|---|---|---|---|---|---|
| **Control Baseline** | 0.0s | N/A | `decode` | Yes ($\ge 50\%$) | $0.90 \le \text{share} \le 1.10$ |
| **Slow Tokenizer** | +0.50s | `tokenization` | `tokenization` | Yes ($> 65\%$) | $0.90 \le \text{share} \le 1.10$ |
| **No Batching** | +0.80s | `queue_wait` | `queue_wait` | Yes ($> 70\%$) | $0.90 \le \text{share} \le 1.10$ |
| **CPU Preprocessing** | +0.40s | `tokenization` | `tokenization` | Yes ($> 55\%$) | $0.90 \le \text{share} \le 1.10$ |
