# Task 5a — GPU Memory Breakdown: Prefill vs Decode

**Role**: M3 (GPU and Deep-Dive Lead)  
**Project**: 6 — Inference Cost and Latency Profiler for LLM Serving Stacks  
**Week**: 4 of 5  

---

## 1. Overview and Purpose

When serving Large Language Models (LLMs), GPU High-Bandwidth Memory (HBM/VRAM) is shared across multiple competing consumers:
1. Static model parameters (**weights**).
2. Dynamic conversation history (**kv_cache**).
3. Transient scratchpad computations (**activations**).
4. System, runtime, and allocator overheads (**other**).

Task 5a measures how GPU memory splits across these four components during **prefill** (processing the prompt) versus **decode** (generating tokens one-by-one). It outputs:
- `<request_id>_memory_breakdown.json` & `memory_breakdown.json` (detailed per-phase byte breakdown with 7 diagnostics fields).
- `<request_id>_memory_breakdown.png` (stacked-bar visualization in MiB, with small-segment side labels).
- `m3_evidence.json` (compact summary with KV capacity estimates for M4's bottleneck analyzer).

---

## 2. Component Definitions and Measurement Methodology

| Component | Definition | How It Is Measured |
|---|---|---|
| `weights` | Model weights and persistent buffers loaded into VRAM. | `sum(p.numel() * p.element_size() for p in model.parameters()) + sum(b.numel() * b.element_size() for b in model.buffers())` |
| `kv_cache` | Dynamic Key and Value tensors stored across all attention layers. | Measured directly from the active HuggingFace cache object using `cache_nbytes()`, and validated against the analytical formula. |
| `activations` | **Transient memory during the phase**: dynamic scratchpad buffers allocated during the forward computation. | Measured relative to phase baseline: `max(0, peak_alloc - allocated_at_phase_start - kv_growth_in_phase)`. |
| `other` | CUDA context, cuBLAS workspace, PyTorch caching allocator slack, loaded libraries, and resident overhead. | Remainder up to NVML reported usage: `max(0, nvml_used - weights - kv_cache - activations)`. |

### Conservation Invariant
By construction:
$$\text{weights} + \text{kv\_cache} + \text{activations} + \text{other} = \text{nvml\_used}$$
whenever $\text{nvml\_used} \ge \text{peak\_alloc}$. Each component is guaranteed non-negative ($\ge 0$). Any resident memory present at phase start beyond weights is absorbed into `other`.

---

## 3. Why the Old Formula Overstated Activations & The Phase-Start Method

### The Flaw in `peak - weights - kv`
The earlier naive formula:
$$\text{activations} = \text{peak\_allocated} - \text{weights} - \text{kv\_cache}$$
implicitly assumed that active memory at the beginning of any phase equals strictly `weights` (for prefill) or `weights + prefill_kv` (for decode).

However, empirical diagnostics revealed that `torch.cuda.memory_allocated()` at the start of prefill was **961.13 MiB**, whereas weights were only **942.29 MiB**. A resident baseline of **~18.84 MiB** (`resident_beyond_weights_bytes`) was already permanently allocated on the GPU before the request began and persisted through both phases.
- Under the old formula, that 18.84 MiB resident baseline was mistakenly attributed to activations.
- For decode, where the true transient allocation is only ~2.47 MiB, the old formula reported **21.31 MiB** ($18.84 + 2.47$), inflating decode activations by nearly an order of magnitude!

### The Transient Phase-Start Formula
To isolate true transient activations generated during the phase:
$$\text{activations} = \max\left(0, \text{peak\_allocated} - \text{allocated\_at\_phase\_start} - \Delta\text{kv}\right)$$
where $\Delta\text{kv} = \text{kv\_end} - \text{kv\_start}$ is the KV cache growth during that phase:
- **Prefill**: $\text{kv\_start} = 0$, so $\Delta\text{kv} = \text{kv\_prefill}$.
- **Decode**: $\text{kv\_start} = \text{kv\_prefill}$, so $\Delta\text{kv} = \text{kv\_decode} - \text{kv\_prefill}$.

### Resident Memory Beyond Weights
The root cause of `resident_beyond_weights_bytes` (~18.84 MiB) is currently being diagnosed. The new diagnostics track:
- `allocated_after_load_bytes`: Memory allocated immediately after model instantiation.
- `allocated_after_warmup_bytes`: Memory allocated after warmup forward passes and `gc.collect()`.
Comparing these two values reveals whether the 18.84 MiB originates during initial model/tokenizer loading or is retained after the warmup pass.

### Reserved vs Allocated Memory (Why "Other" Shifts while NVML Stays Flat)
PyTorch's caching allocator requests large contiguous virtual memory blocks from CUDA via `cudaMalloc` (tracked by `torch.cuda.memory_reserved()` and visible in `NVML Total`).
- Inside PyTorch, memory is split into **allocated** (actively held by tensors: `weights + kv + activations`) and **reserved slack** (cached blocks held by the allocator to satisfy future tensor allocations without OS calls).
- Because `other = nvml_used - weights - kv - activations`, any change in transient activations or KV size causes a complementary shift in `other`. NVML total stays identical because the caching allocator does not return reserved memory blocks to the OS between prefill and decode.

---

## 4. Analytical KV Cache Formula

For a decoder-only model with Grouped-Query Attention (GQA) or Multi-Query Attention (MQA):
$$\text{KV Cache Bytes} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times T \times S_{\text{dtype}}$$

Where:
- $2$: Accounts for both Key and Value matrices.
- $N_{\text{layers}}$: Transformer layers (`num_hidden_layers`).
- $N_{\text{kv\_heads}}$: Key/Value attention heads (`num_key_value_heads`).
- $d_{\text{head}}$: Head dimension (`hidden_size // num_attention_heads` or `head_dim`).
- $T$: Number of tokens cached in memory.
- $S_{\text{dtype}}$: Bytes per element ($2$ bytes for `float16` / `bfloat16`).

### Example: Qwen/Qwen2.5-0.5B-Instruct
- $N_{\text{layers}} = 24$, $N_{\text{kv\_heads}} = 2$, $d_{\text{head}} = 64$, $S_{\text{dtype}} = 2$ (fp16)

**Per-token analytical footprint**:
$$2 \times 24 \times 2 \times 64 \times 1 \times 2 = 12,288\text{ bytes/token } (\approx 0.0117\text{ MiB/token})$$

For $192$ tokens ($128$ prompt $+ 64$ generated):
$$12,288 \times 192 = 2,359,296\text{ bytes } (\approx 2.25\text{ MiB})$$

The script verifies that measured bytes per token match analytical bytes per token within 5% (`kv_matches_analytical: true`).

---

## 5. Activations: Prefill vs Decode

During prefill, the model processes the entire prompt ($S = 128$ tokens) in a single forward pass:
1. **Self-Attention**: Materializes attention scores across all prompt positions.
2. **Vocabulary Projection**: Qwen2.5-0.5B has a vocabulary size of $151,936$.
   In standard Hugging Face eager mode, the LM head projects all prompt positions to logits:
   $$\text{Logit tensor size} = 1 \times S \times V \times 2\text{ bytes} = 128 \times 151,936 \times 2 = 38,895,616\text{ bytes } \approx 37.09\text{ MiB}$$
3. During decode, $S = 1$, so the single-token logit tensor is only $1 \times 1 \times 151,936 \times 2 \approx 0.29\text{ MiB}$.

Therefore, prefill transient activations are expected to be significantly larger than decode activations (~38.2 MiB vs ~2.5 MiB).

---

## 6. How to Run on Kaggle (Tesla T4)

```bash
python llm_profiler/collectors/memory_breakdown.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompt-tokens 128 \
    --output-tokens 64 \
    --warmup 1 \
    --out-dir results/deep_dive
```

---

## 7. Results (Kaggle Tesla T4 Benchmark)

### Hardware & Configuration
- **GPU**: NVIDIA Tesla T4 (15,360 MiB)
- **Model**: Qwen/Qwen2.5-0.5B-Instruct (`float16`)
- **Tokens**: 128 prompt $\to$ 64 output

### Memory Breakdown Table (MiB)

| Component | Prefill (MiB) | Decode (MiB) | Notes |
|---|---|---|---|
| `weights` | 942.29 | 942.29 | Static model weights + persistent buffers |
| `kv_cache` | 1.50 | 2.24 | 128 tokens (prefill) $\to$ 191 tokens (decode end) |
| `activations` | 38.22 | 2.47 *(re-measured with the phase-start method)* | True transient computation overhead |
| `other` | 365.99 | 401.00 | CUDA context, cuBLAS workspace, allocator slack, resident overhead |
| **NVML Total** | **1348.00** | **1348.00** | Identical across phases (allocator block reuse) |

### Diagnostics (Allocation & Reservation Tracker)
- **Allocated after load**: `942.29 MiB`
- **Allocated after warmup**: `961.13 MiB`
- **Prefill start allocated**: `961.13 MiB`
- **Decode start allocated**: `962.63 MiB` ($961.13 + 1.50\text{ MiB KV}$)
- **Resident beyond weights**: `18.84 MiB` (appears after warmup; absorbed into `other`)
- **Reserved at prefill end**: `1258.00 MiB`
- **Reserved at decode end**: `1258.00 MiB`
- **KV Cache per token**: Measured: `12,288 B` | Analytical: `12,288 B` | Matches: **Yes**
