# Member 4: Inference Cost and Bottleneck Analyzer

Part of **Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**.  
**Lead:** Member M4 (Analyzer and Validation Lead)  
**Branch:** `m4-analyzer-validation`

---

## 1. Overview
The **Analyzer** decomposes end-to-end LLM serving latency across canonical request stages, validates consistency between stage sums and total latency, ranks the top 3 bottlenecks with empirical evidence, suggests domain-specific remediation actions, and computes the amortized dollar cost per 1M tokens.

---

## 2. Shared Contracts Adherence

### Canonical Stage Names
`tokenization`, `queue_wait`, `prefill`, `decode`, `detokenization`, `network`, plus `total`.

### Prometheus Metrics
- `llmprof_cost_per_million_tokens_usd` (Gauge, no labels)
- Scraped along with M2 stage latency histograms and M3 GPU metrics.

### Request Log CSV Schema
Exact column sequence:
```text
request_id,send_ts,end_ts,total_latency_s,prompt_tokens,output_tokens,status,target_rps,scenario_tag
```

### Per-Stage Trace JSON Lines Schema
```json
{"request_id": "...", "stage": "...", "start_ts": 100.0, "end_ts": 100.05, "duration_s": 0.05}
```

### Analyzer Contract Output JSON
```json
{
  "top_bottlenecks": [
    {
      "rank": 1,
      "stage": "prefill",
      "share_of_latency": 0.6241,
      "evidence": "'prefill' accounts for 62.4% of total latency (mean: 0.1420s, p95: 0.2280s across 100 requests).",
      "suggested_action": "Enable chunked prefill (--enable-chunked-prefill), utilize prompt prefix caching (--enable-prefix-caching), optimize prompt token length, or scale tensor parallelism."
    }
  ],
  "cost_per_million_tokens_usd": 1.446759,
  "explained_share": 0.9982
}
```

---

## 3. Cost Model Mathematics

$$\text{Cost per 1M Tokens (USD)} = \left(\frac{\text{gpu\_price\_per\_hour\_usd} \times (\sum \text{total\_latency\_s} / 3600)}{\sum (\text{prompt\_tokens} + \text{output\_tokens})}\right) \times 1{,}000{,}000$$

- `gpu_price_per_hour_usd`: Loaded dynamically from `config.yaml` (`1.0` USD default for RTX 3060). Zero hardcoded pricing.
- Respects failed requests (`status != "ok"` excluded from operational inference cost).

---

## 4. Usage & CLI Commands

### 1. Calculate Cost on a Request CSV
```bash
python analyzer/cost_model.py --csv results/baseline_b1_low_load.csv --export-metrics
```

### 2. Run Bottleneck Analysis
```bash
python analyzer/bottleneck_analyzer.py --csv results/baseline_b1_low_load.csv --traces results/traces.jsonl --out results/analysis.json
```

### 3. Inject Bottleneck Anomalies (Synthetic Validation)
```bash
python analyzer/bottleneck_injector.py --stage prefill --delay 0.5 --num-requests 50 --out-dir results/injected/
```

### 4. Run Experiment Suite E1 to E6
```bash
python experiments/run_experiments_e1_to_e6.py --out-dir results/experiments/
```

### 5. Measure Profiler Observer Overhead (E6)
```bash
python experiments/measure_overhead.py --iterations 10000
```

### 6. Run the End-to-End Interactive Demo
```bash
python experiments/demo_m4.py
```

---

## 5. Consistency & Quality Assurance (Rule 6)
- **Never trust numbers blindly**: `explained_share` calculates $\frac{\sum \text{stage durations}}{\text{total latency}}$. If traces are missing or uninstrumented framework/OS overhead exists, it is reported explicitly without fabricating numbers.
- **Unit and Integration Tests**: 100% passing across `test_cost_model.py`, `test_bottleneck_analyzer.py`, `test_bottleneck_injector.py`, `test_measure_overhead.py`, and `test_experiments_e1_to_e6.py`.
