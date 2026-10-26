#!/usr/bin/env python3
"""M3 GPU Memory Breakdown: Prefill vs Decode Allocation Analysis.

Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks
Role: M3 (GPU and Deep-Dive Lead)

Purpose:
    Measure where GPU memory goes (weights, kv_cache, activations, other)
    during prefill versus decode for ONE request. Produces a detailed JSON
    breakdown, an M4-consumable evidence file, and a stacked-bar chart.

Contract:
    - Components: weights, kv_cache, activations, other
    - Phases: prefill, decode
    - Invariant: weights + kv_cache + activations + other == nvml_used (when nvml_used >= peak_alloc)

Design:
    - Standalone file with ZERO intra-repo dependencies.
    - torch, transformers, and pynvml are imported LAZILY inside functions
      so CLI --help and unit tests execute cleanly on development laptops without CUDA/GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── CONSTANTS (single source of truth) ──────────────────────────────────────
DEFAULT_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT_TOKENS: int = 128
DEFAULT_OUTPUT_TOKENS: int = 64
DEFAULT_WARMUP: int = 1
DEFAULT_OUT_DIR: str = "results/deep_dive"
DEFAULT_DEVICE_INDEX: int = 0
CONTRACT_COMPONENTS: List[str] = ["weights", "kv_cache", "activations", "other"]
CONTRACT_PHASES: List[str] = ["prefill", "decode"]
CONTRACT_STAGES: List[str] = ["tokenization", "prefill", "decode", "detokenization"]

CONFIG_SEARCH_PATHS: List[str] = [
    os.path.join("llm_profiler", "config.yaml"),
    "config.yaml",
    os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
]

logger = logging.getLogger("memory_breakdown")


# ─── CONFIG LOADER ──────────────────────────────────────────────────────────

def load_model_from_config() -> Optional[str]:
    """Search for config.yaml and return model_name if defined."""
    try:
        import yaml  # lazy
    except ImportError:
        return None

    for path in CONFIG_SEARCH_PATHS:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                model = data.get("model_name")
                if model:
                    logger.info("Loaded model_name '%s' from config at '%s'", model, path)
                    return str(model)
            except Exception as exc:
                logger.debug("Failed to read config from '%s': %s", path, exc)
    return None


# ─── PURE MATH & COMPONENT FUNCTIONS ────────────────────────────────────────

def compute_components(
    weights: int,
    kv: int,
    peak_alloc: int,
    nvml_used: int,
    allocated_at_start: Optional[int] = None,
    kv_start: int = 0,
) -> Dict[str, int]:
    """Compute GPU memory breakdown components according to contract maths.

    Transient Activations Formula:
        kv_growth_in_phase = max(0, kv - kv_start)
        alloc_start = allocated_at_start if allocated_at_start is not None else (weights + kv_start)
        activations = max(0, peak_alloc - alloc_start - kv_growth_in_phase)
        other       = max(0, nvml_used - weights - kv - activations)

    By construction:
        weights + kv_cache + activations + other == nvml_used
        whenever nvml_used >= peak_alloc. All component values are >= 0.
        Any resident memory present at phase start beyond weights is absorbed
        into 'other'.

    Args:
        weights: Model weights and persistent buffers in bytes.
        kv: Key-Value cache allocated bytes at phase end.
        peak_alloc: torch.cuda.max_memory_allocated() for the phase in bytes.
        nvml_used: Total GPU memory used reported by NVML in bytes.
        allocated_at_start: torch.cuda.memory_allocated() at the start of the phase.
        kv_start: Key-Value cache bytes at phase start (0 for prefill,
                  prefill end KV for decode).

    Returns:
        Dict with exact keys: 'weights', 'kv_cache', 'activations', 'other'.
    """
    kv_growth = max(0, kv - kv_start)
    alloc_start = allocated_at_start if allocated_at_start is not None else (weights + kv_start)
    activations = max(0, peak_alloc - alloc_start - kv_growth)
    other = max(0, nvml_used - weights - kv - activations)
    return {
        "weights": int(weights),
        "kv_cache": int(kv),
        "activations": int(activations),
        "other": int(other),
    }


def analytical_kv_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    tokens: int,
    dtype_bytes: int = 2,
) -> int:
    """Calculate analytical KV cache size in bytes.

    Formula:
        2 * num_layers * num_kv_heads * head_dim * tokens * dtype_bytes
        (Factor of 2 accounts for both keys and values).

    Args:
        num_layers: Number of transformer decoder layers.
        num_kv_heads: Number of key/value attention heads (GQA/MQA).
        head_dim: Dimensionality of each attention head.
        tokens: Number of tokens cached.
        dtype_bytes: Bytes per element (2 for float16 / bfloat16).

    Returns:
        Total KV cache size in bytes.
    """
    if tokens < 0:
        raise ValueError(f"tokens must be >= 0, got {tokens}")
    return 2 * num_layers * num_kv_heads * head_dim * tokens * dtype_bytes


def cache_nbytes(cache: Any) -> int:
    """Measure the memory footprint of a HuggingFace KV cache in bytes.

    Supports three common cache structures:
        (i) Modern Cache object with .layers attribute (each layer having .keys/.values).
        (ii) Cache object with .key_cache and .value_cache lists of tensors.
        (iii) Legacy tuple of (key_tensor, value_tensor) pairs.

    Args:
        cache: HuggingFace past_key_values or custom cache object.

    Returns:
        Total size of all cached tensors in bytes.
    """
    if cache is None:
        return 0

    total_bytes = 0

    def _tensor_bytes(t: Any) -> int:
        if t is not None and hasattr(t, "numel") and hasattr(t, "element_size"):
            return int(t.numel() * t.element_size())
        return 0

    # (i) Modern Cache object with .layers
    if hasattr(cache, "layers") and isinstance(cache.layers, (list, tuple)):
        for layer in cache.layers:
            # Layer may have .keys / .values or .key_cache / .value_cache
            for attr in ("keys", "values", "key_cache", "value_cache"):
                val = getattr(layer, attr, None)
                if val is not None:
                    total_bytes += _tensor_bytes(val)
        if total_bytes > 0:
            return total_bytes

    # (ii) Cache object with .key_cache and .value_cache lists
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        kc = getattr(cache, "key_cache")
        vc = getattr(cache, "value_cache")
        if isinstance(kc, (list, tuple)):
            for t in kc:
                total_bytes += _tensor_bytes(t)
        if isinstance(vc, (list, tuple)):
            for t in vc:
                total_bytes += _tensor_bytes(t)
        if total_bytes > 0:
            return total_bytes

    # (iii) Legacy tuple of (key_tensor, value_tensor) tuples
    if isinstance(cache, (tuple, list)):
        for item in cache:
            if isinstance(item, (tuple, list)):
                for t in item:
                    total_bytes += _tensor_bytes(t)
            else:
                total_bytes += _tensor_bytes(item)

    return total_bytes


def extract_prefill_and_cleanup(
    prefill_out: Any,
) -> Tuple[Any, Any]:
    """Extract past_key_values and next_token_id, then release references to prefill_out.

    This ensures large prefill tensors (such as the full-sequence logits tensor,
    ~37.1 MiB for 128 tokens x 151,936 vocab in fp16) are not kept alive during
    the decode phase, which would distort peak decode memory measurements.

    Args:
        prefill_out: Model output object from prefill forward pass.

    Returns:
        Tuple of (past_key_values, next_token_id).
    """
    past_key_values = getattr(prefill_out, "past_key_values", None)
    next_token_id = None

    if hasattr(prefill_out, "logits") and prefill_out.logits is not None:
        logits = prefill_out.logits
        if hasattr(logits, "argmax"):
            next_token_id = logits[:, -1, :].argmax(dim=-1)
            if hasattr(next_token_id, "clone"):
                next_token_id = next_token_id.clone()
        else:
            next_token_id = getattr(prefill_out, "next_token_id", None)
    elif isinstance(prefill_out, (tuple, list)) and len(prefill_out) > 0:
        logits = prefill_out[0]
        past_key_values = prefill_out[1] if len(prefill_out) > 1 else None
        if hasattr(logits, "argmax"):
            next_token_id = logits[:, -1, :].argmax(dim=-1)
            if hasattr(next_token_id, "clone"):
                next_token_id = next_token_id.clone()
    else:
        next_token_id = getattr(prefill_out, "next_token_id", None)

    return past_key_values, next_token_id


def build_prompt(
    tokenizer: Any,
    target_tokens: int,
    seed_word: str = "inference",
) -> Tuple[str, List[int]]:
    """Build a synthetic prompt of EXACTLY target_tokens input tokens."""
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")

    candidate = (seed_word + " ") * (target_tokens * 2)
    ids = tokenizer.encode(candidate, add_special_tokens=False)

    if len(ids) >= target_tokens:
        ids = ids[:target_tokens]
    else:
        pad_id = getattr(tokenizer, "pad_token_id", None) or 0
        ids = ids + [pad_id] * (target_tokens - len(ids))

    text = tokenizer.decode(ids, skip_special_tokens=False)
    return text, ids


# ─── ARTIFACT BUILDERS ──────────────────────────────────────────────────────

def build_breakdown_json(
    request_id: str,
    generated_at_epoch: float,
    model: str,
    device: str,
    gpu_total_bytes: int,
    prompt_tokens: int,
    output_tokens: int,
    method: str,
    nvml_idle_bytes: int,
    prefill_components: Dict[str, int],
    prefill_peak_alloc: int,
    prefill_nvml_used: int,
    decode_components: Dict[str, int],
    decode_peak_alloc: int,
    decode_nvml_used: int,
    kv_per_token_measured: float,
    kv_per_token_analytical: int,
    kv_matches_analytical: bool,
    prefill_alloc_start: int = 0,
    decode_alloc_start: int = 0,
    diagnostics: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Construct the standardized memory breakdown dictionary.

    Note on nvml_idle_bytes:
        Represents the NVML baseline after model loading and warmup, before
        request execution. It includes persistent model weights and buffers.
    """
    diag_data = {
        "allocated_after_load_bytes": 0,
        "allocated_after_warmup_bytes": 0,
        "prefill_allocated_at_start_bytes": int(prefill_alloc_start),
        "decode_allocated_at_start_bytes": int(decode_alloc_start),
        "reserved_at_prefill_end_bytes": 0,
        "reserved_at_decode_end_bytes": 0,
        "resident_beyond_weights_bytes": 0,
    }
    if diagnostics:
        for k in diag_data:
            if k in diagnostics:
                diag_data[k] = int(diagnostics[k])

    return {
        "request_id": request_id,
        "generated_at_epoch": generated_at_epoch,
        "model": model,
        "device": device,
        "gpu_total_bytes": int(gpu_total_bytes),
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
        "method": method,
        "nvml_idle_bytes": int(nvml_idle_bytes),  # NVML baseline after model load, before request
        "phases": {
            "prefill": {
                "weights": int(prefill_components["weights"]),
                "kv_cache": int(prefill_components["kv_cache"]),
                "activations": int(prefill_components["activations"]),
                "other": int(prefill_components["other"]),
                "nvml_used_bytes": int(prefill_nvml_used),
                "torch_peak_allocated_bytes": int(prefill_peak_alloc),
            },
            "decode": {
                "weights": int(decode_components["weights"]),
                "kv_cache": int(decode_components["kv_cache"]),
                "activations": int(decode_components["activations"]),
                "other": int(decode_components["other"]),
                "nvml_used_bytes": int(decode_nvml_used),
                "torch_peak_allocated_bytes": int(decode_peak_alloc),
            },
        },
        "diagnostics": diag_data,
        "kv_cache_bytes_per_token_measured": float(kv_per_token_measured),
        "kv_cache_bytes_per_token_analytical": int(kv_per_token_analytical),
        "kv_matches_analytical": bool(kv_matches_analytical),
    }


def build_evidence(
    gpu_total_bytes: int,
    weights_bytes: int,
    kv_cache_bytes_per_token: float,
    kv_cache_bytes_end_of_request: int,
    nvml_used_decode: int,
    activations_peak_prefill_bytes: int,
    activations_peak_decode_bytes: int,
    other_bytes_decode: int,
    prefill_components: Dict[str, int],
    decode_components: Dict[str, int],
    summary_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the evidence artifact consumable by the M4 bottleneck analyzer."""
    kv_share = (
        float(kv_cache_bytes_end_of_request / nvml_used_decode)
        if nvml_used_decode > 0
        else 0.0
    )
    free_bytes = max(0, gpu_total_bytes - nvml_used_decode)
    tokens_fit = (
        int(free_bytes // kv_cache_bytes_per_token)
        if kv_cache_bytes_per_token > 0
        else 0
    )

    dominant_prefill = max(
        CONTRACT_COMPONENTS, key=lambda c: prefill_components.get(c, 0)
    )
    dominant_decode = max(
        CONTRACT_COMPONENTS, key=lambda c: decode_components.get(c, 0)
    )

    evidence: Dict[str, Any] = {
        "gpu_total_bytes": int(gpu_total_bytes),
        "weights_bytes": int(weights_bytes),
        "kv_cache_bytes_per_token": float(kv_cache_bytes_per_token),
        "kv_cache_bytes_end_of_request": int(kv_cache_bytes_end_of_request),
        "kv_share_of_used_memory_decode": round(kv_share, 6),
        "activations_peak_prefill_bytes": int(activations_peak_prefill_bytes),
        "activations_peak_decode_bytes": int(activations_peak_decode_bytes),
        "other_bytes_decode": int(other_bytes_decode),
        "nvml_free_after_decode_bytes": int(free_bytes),
        "kv_tokens_that_fit_in_free_memory_estimate": int(tokens_fit),
        "dominant_component_prefill": dominant_prefill,
        "dominant_component_decode": dominant_decode,
    }

    if summary_data:
        stage_durations = summary_data.get("stage_duration_s", {})
        evidence["prefill_s"] = stage_durations.get("prefill")
        evidence["decode_total_s"] = stage_durations.get("decode")
        evidence["decode_mean_token_s"] = summary_data.get("mean_decode_token_s")

    evidence["notes"] = [
        "Standalone Hugging Face eager request, not the vLLM allocator.",
        "vLLM preallocates KV cache blocks, so actual vLLM memory profile will differ.",
        "Component values (activations, other) are derived estimates using peak allocated and NVML totals.",
        "Resident memory allocated before the request beyond model weights (resident_beyond_weights_bytes) is counted under 'other' so that the four components sum to total NVML used.",
    ]

    return evidence


# ─── VISUALIZATION (MATPLOTLIB) ─────────────────────────────────────────────

def build_figure(breakdown_data: Dict[str, Any]) -> Any:
    """Build and return the matplotlib Figure for GPU memory breakdown.

    Args:
        breakdown_data: Standard memory breakdown dictionary with 'phases'.

    Returns:
        matplotlib.figure.Figure with two subplots: all components and dynamic components.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt

    phases = CONTRACT_PHASES  # ["prefill", "decode"]
    phases_data = breakdown_data["phases"]

    colors = {
        "weights": "#4C78A8",     # Blue
        "kv_cache": "#F58518",    # Orange
        "activations": "#E45756", # Red
        "other": "#72B7B2",       # Teal
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Helper: Convert bytes to MiB
    def to_mib(b: float) -> float:
        return b / (1024.0 * 1024.0)

    # ── Subplot 1: All Components ──
    x_pos = [0, 1]
    width = 0.5

    bottoms1 = [0.0, 0.0]
    for comp in CONTRACT_COMPONENTS:
        heights = [to_mib(phases_data[p][comp]) for p in phases]
        bars = ax1.bar(
            x_pos, heights, width, bottom=bottoms1,
            label=comp, color=colors[comp], edgecolor="white"
        )
        for i, h in enumerate(heights):
            if h >= 25.0:  # Large enough to annotate inside without overlapping borders
                ax1.text(
                    x_pos[i], bottoms1[i] + h / 2.0,
                    f"{h:.1f}", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold"
                )
            elif h >= 1.0:  # Small segment (e.g. decode activations ~2.5 MiB) -> place outside bar
                y_offsets = {"kv_cache": -12, "activations": 12}
                ax1.annotate(
                    f"← {h:.1f} ({comp})",
                    xy=(x_pos[i] + width / 2.0, bottoms1[i] + h / 2.0),
                    xytext=(10, y_offsets.get(comp, 0)),
                    textcoords="offset points",
                    ha="left", va="center",
                    color=colors[comp], fontsize=8.5, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": colors[comp], "lw": 0.7},
                    annotation_clip=False,
                )
        bottoms1 = [b + h for b, h in zip(bottoms1, heights)]

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(["Prefill", "Decode"], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Memory (MiB)", fontsize=11)
    ax1.set_title("All Components (Total Used)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_xlim(-0.4, 2.8)
    tallest1 = max(bottoms1)
    ax1.set_ylim(bottom=0.0, top=max(ax1.get_ylim()[1], tallest1 * 1.05))

    # ── Subplot 2: Dynamic Memory (Without Weights) ──
    dynamic_components = ["kv_cache", "activations", "other"]
    bottoms2 = [0.0, 0.0]
    for comp in dynamic_components:
        heights = [to_mib(phases_data[p][comp]) for p in phases]
        bars = ax2.bar(
            x_pos, heights, width, bottom=bottoms2,
            label=comp, color=colors[comp], edgecolor="white"
        )
        for i, h in enumerate(heights):
            if h >= 10.0:  # Large enough to annotate inside the bar
                ax2.text(
                    x_pos[i], bottoms2[i] + h / 2.0,
                    f"{h:.1f}", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold"
                )
            elif h > 0.05:  # Small segment (under 10 MiB, e.g. kv_cache, decode activations)
                raw_y = bottoms2[i] + h / 2.0
                y_offsets = {
                    ("prefill", "kv_cache"): 24,
                    ("decode", "kv_cache"): -12,
                    ("decode", "activations"): 12,
                }
                ax2.annotate(
                    f"← {h:.2f} MiB ({comp})",
                    xy=(x_pos[i] + width / 2.0, raw_y),
                    xytext=(10, y_offsets.get((phases[i], comp), 0)),
                    textcoords="offset points",
                    ha="left", va="center",
                    color=colors[comp], fontsize=8.5, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": colors[comp], "lw": 0.7},
                    annotation_clip=False,
                )
        bottoms2 = [b + h for b, h in zip(bottoms2, heights)]

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["Prefill", "Decode"], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Memory (MiB)", fontsize=11)
    ax2.set_title("Dynamic Memory (Weights Excluded)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_xlim(-0.4, 2.8)
    tallest2 = max(bottoms2)
    ax2.set_ylim(bottom=-2.0, top=max(ax2.get_ylim()[1], tallest2 * 1.05))

    # Global title
    model_name = breakdown_data.get("model", "Model")
    device_name = breakdown_data.get("device", "GPU")
    prompt_tokens = breakdown_data.get("prompt_tokens", 0)
    output_tokens = breakdown_data.get("output_tokens", 0)
    fig.suptitle(
        f"GPU Memory Breakdown — {model_name} on {device_name}\n"
        f"Prompt: {prompt_tokens} tokens | Output: {output_tokens} tokens",
        fontsize=13, fontweight="bold", y=0.98
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig


def plot_breakdown(
    breakdown_data: Dict[str, Any],
    out_path: str,
) -> None:
    """Generate and save dual-subplot stacked bar chart of memory components."""
    import matplotlib.pyplot as plt

    fig = build_figure(breakdown_data)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── MEASUREMENT RUNNER (GPU / CUDA) ─────────────────────────────────────────

def run_phases(
    model: Any,
    tokenizer: Any,
    device: Any,
    device_index: int,
    prompt_tokens: int,
    output_tokens: int,
    warmup: int,
) -> Dict[str, Any]:
    """Execute prefill and decode measurement passes on active CUDA device."""
    import pynvml
    import torch

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def _get_nvml_used() -> int:
        info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        return int(info.used)

    gpu_total_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).total)

    # Record allocated bytes right after model loading (before warmup)
    allocated_after_load = int(torch.cuda.memory_allocated(device))

    # 1. Weights and persistent buffers
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    logger.info("Measured model weights + buffers: %.2f MiB", weights_bytes / (1024 * 1024))

    # 2. Warmup passes
    _, warmup_ids = build_prompt(tokenizer, prompt_tokens)
    warmup_tensor = torch.tensor([warmup_ids], dtype=torch.long).to(device)

    if warmup > 0:
        logger.info("Running %d warmup pass(es)...", warmup)
        for _ in range(warmup):
            with torch.inference_mode():
                out = model(input_ids=warmup_tensor, use_cache=True)
                pkv = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1)
                for _ in range(min(4, max(1, output_tokens - 1))):
                    s_out = model(input_ids=nxt.unsqueeze(0), past_key_values=pkv, use_cache=True)
                    pkv = s_out.past_key_values
                    nxt = s_out.logits[:, -1, :].argmax(dim=-1)
        del out, pkv, nxt
        gc.collect()
        torch.cuda.empty_cache()

    allocated_after_warmup = int(torch.cuda.memory_allocated(device))
    nvml_idle_bytes = _get_nvml_used()
    logger.info(
        "NVML baseline after model load, before request: %.2f MiB",
        nvml_idle_bytes / (1024 * 1024),
    )

    # 3. Prepare real prompt
    _, prompt_ids = build_prompt(tokenizer, prompt_tokens)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    # 4. PREFILL PHASE
    logger.info("Executing prefill phase (%d tokens)...", prompt_tokens)
    torch.cuda.reset_peak_memory_stats(device)
    prefill_alloc_start = int(torch.cuda.memory_allocated(device))
    torch.cuda.synchronize(device)

    with torch.inference_mode():
        prefill_out = model(input_ids=input_ids, use_cache=True)

    torch.cuda.synchronize(device)
    prefill_peak_alloc = int(torch.cuda.max_memory_allocated(device))
    reserved_prefill_end = int(torch.cuda.memory_reserved(device))
    kv_prefill = cache_nbytes(prefill_out.past_key_values)
    nvml_prefill_used = _get_nvml_used()

    # Explicitly extract decode prerequisites, clean up prefill output and logits,
    # and run garbage collection so ~37.1 MiB prefill logits do not leak into decode peak.
    # Note: torch.cuda.empty_cache() is intentionally NOT called between phases so NVML stays comparable.
    past_key_values, next_token_id = extract_prefill_and_cleanup(prefill_out)
    del prefill_out
    gc.collect()

    # 5. DECODE PHASE
    logger.info("Executing decode phase (%d tokens)...", output_tokens)
    torch.cuda.reset_peak_memory_stats(device)
    decode_alloc_start = int(torch.cuda.memory_allocated(device))
    torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(output_tokens - 1):
            step_out = model(
                input_ids=next_token_id.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = step_out.past_key_values
            next_token_id = step_out.logits[:, -1, :].argmax(dim=-1)

    torch.cuda.synchronize(device)
    decode_peak_alloc = int(torch.cuda.max_memory_allocated(device))
    reserved_decode_end = int(torch.cuda.memory_reserved(device))
    kv_decode = cache_nbytes(past_key_values)
    nvml_decode_used = _get_nvml_used()

    resident_beyond_weights = max(0, prefill_alloc_start - weights_bytes)

    pynvml.nvmlShutdown()

    return {
        "gpu_total_bytes": gpu_total_bytes,
        "weights_bytes": weights_bytes,
        "nvml_idle_bytes": nvml_idle_bytes,
        "diagnostics": {
            "allocated_after_load_bytes": allocated_after_load,
            "allocated_after_warmup_bytes": allocated_after_warmup,
            "prefill_allocated_at_start_bytes": prefill_alloc_start,
            "decode_allocated_at_start_bytes": decode_alloc_start,
            "reserved_at_prefill_end_bytes": reserved_prefill_end,
            "reserved_at_decode_end_bytes": reserved_decode_end,
            "resident_beyond_weights_bytes": resident_beyond_weights,
        },
        "prefill": {
            "peak_alloc": prefill_peak_alloc,
            "kv_bytes": kv_prefill,
            "nvml_used": nvml_prefill_used,
        },
        "decode": {
            "peak_alloc": decode_peak_alloc,
            "kv_bytes": kv_decode,
            "nvml_used": nvml_decode_used,
        },
    }


# ─── CLI & MAIN ─────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M3 GPU Memory Breakdown: Prefill vs Decode Allocation Analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID (reads config.yaml, falls back to constant)",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=DEFAULT_PROMPT_TOKENS,
        help="Exact number of input (prompt) tokens",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help="Exact number of tokens to generate",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup iterations before measurement",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where output files are written",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Request identifier (default: mb-<epoch>)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=DEFAULT_DEVICE_INDEX,
        help="CUDA device index",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path to deep_dive *_summary.json",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        default=False,
        help="Skip generating matplotlib breakdown chart",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)

    # 1. Environment & CUDA Guard
    try:
        import pynvml
        import torch
    except ImportError as exc:
        logger.error("Required package missing: %s. Install torch and nvidia-ml-py.", exc)
        return 1

    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. Memory breakdown measurement requires an NVIDIA GPU."
        )
        return 1

    # 2. Resolve Model and Paths
    model_name = args.model or load_model_from_config() or DEFAULT_MODEL
    request_id = args.request_id or f"mb-{int(time.time())}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device_index}")
    logger.info("Initializing %s on %s...", model_name, device)

    # 3. Load Model and Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device).eval()

    # 4. Execute Measurement
    raw_metrics = run_phases(
        model=model,
        tokenizer=tokenizer,
        device=device,
        device_index=args.device_index,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        warmup=args.warmup,
    )

    weights_bytes = raw_metrics["weights_bytes"]
    gpu_total_bytes = raw_metrics["gpu_total_bytes"]
    nvml_idle_bytes = raw_metrics["nvml_idle_bytes"]

    prefill_raw = raw_metrics["prefill"]
    decode_raw = raw_metrics["decode"]
    diagnostics = raw_metrics.get("diagnostics", {})

    prefill_comps = compute_components(
        weights=weights_bytes,
        kv=prefill_raw["kv_bytes"],
        peak_alloc=prefill_raw["peak_alloc"],
        nvml_used=prefill_raw["nvml_used"],
        allocated_at_start=diagnostics.get("prefill_allocated_at_start_bytes", weights_bytes),
        kv_start=0,
    )

    decode_comps = compute_components(
        weights=weights_bytes,
        kv=decode_raw["kv_bytes"],
        peak_alloc=decode_raw["peak_alloc"],
        nvml_used=decode_raw["nvml_used"],
        allocated_at_start=diagnostics.get("decode_allocated_at_start_bytes", weights_bytes + prefill_raw["kv_bytes"]),
        kv_start=prefill_raw["kv_bytes"],
    )

    # 5. Analytical KV Calculation
    num_layers = getattr(
        model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0)
    )
    num_kv_heads = getattr(
        model.config, "num_key_value_heads", getattr(model.config, "num_attention_heads", 0)
    )
    head_dim = getattr(model.config, "head_dim", None)
    if not head_dim:
        hidden_size = getattr(model.config, "hidden_size", 0)
        num_heads = getattr(model.config, "num_attention_heads", 1)
        head_dim = hidden_size // num_heads if num_heads else 64

    analytical_per_token = analytical_kv_bytes(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens=1,
        dtype_bytes=2,  # fp16
    )

    total_cache_tokens = args.prompt_tokens + max(0, args.output_tokens - 1)
    measured_per_token = (
        float(decode_raw["kv_bytes"] / total_cache_tokens)
        if total_cache_tokens > 0
        else 0.0
    )
    kv_matches = (
        bool(abs(measured_per_token - analytical_per_token) / analytical_per_token <= 0.05)
        if analytical_per_token > 0
        else False
    )

    if not kv_matches:
        logger.warning(
            "Measured KV bytes/token (%.1f) differs >5%% from analytical (%d)",
            measured_per_token,
            analytical_per_token,
        )

    # 6. Build and Write Breakdown JSON
    breakdown_data = build_breakdown_json(
        request_id=request_id,
        generated_at_epoch=time.time(),
        model=model_name,
        device=f"cuda:{args.device_index}",
        gpu_total_bytes=gpu_total_bytes,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        method="torch_inference_mode_eager",
        nvml_idle_bytes=nvml_idle_bytes,
        prefill_components=prefill_comps,
        prefill_peak_alloc=prefill_raw["peak_alloc"],
        prefill_nvml_used=prefill_raw["nvml_used"],
        decode_components=decode_comps,
        decode_peak_alloc=decode_raw["peak_alloc"],
        decode_nvml_used=decode_raw["nvml_used"],
        kv_per_token_measured=measured_per_token,
        kv_per_token_analytical=analytical_per_token,
        kv_matches_analytical=kv_matches,
        prefill_alloc_start=diagnostics.get("prefill_allocated_at_start_bytes", 0),
        decode_alloc_start=diagnostics.get("decode_allocated_at_start_bytes", 0),
        diagnostics=diagnostics,
    )

    req_json_path = out_dir / f"{request_id}_memory_breakdown.json"
    stable_json_path = out_dir / "memory_breakdown.json"
    for p in (req_json_path, stable_json_path):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(breakdown_data, f, indent=2)
    logger.info("Breakdown JSON: %s (and stable memory_breakdown.json)", req_json_path)

    # 7. Optional Deep Dive Summary Ingestion
    summary_data = None
    if args.summary and os.path.isfile(args.summary):
        try:
            with open(args.summary, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            logger.info("Loaded timing summary from '%s'", args.summary)
        except Exception as exc:
            logger.warning("Could not read summary from '%s': %s", args.summary, exc)

    # 8. Build and Write Evidence JSON
    evidence_data = build_evidence(
        gpu_total_bytes=gpu_total_bytes,
        weights_bytes=weights_bytes,
        kv_cache_bytes_per_token=measured_per_token,
        kv_cache_bytes_end_of_request=decode_raw["kv_bytes"],
        nvml_used_decode=decode_raw["nvml_used"],
        activations_peak_prefill_bytes=prefill_comps["activations"],
        activations_peak_decode_bytes=decode_comps["activations"],
        other_bytes_decode=decode_comps["other"],
        prefill_components=prefill_comps,
        decode_components=decode_comps,
        summary_data=summary_data,
    )
    evidence_path = out_dir / "m3_evidence.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, indent=2)
    logger.info("Evidence file: %s", evidence_path)

    # 9. Chart Generation
    if not args.no_chart:
        chart_path = out_dir / f"{request_id}_memory_breakdown.png"
        stable_chart_path = out_dir / "memory_breakdown.png"
        plot_breakdown(breakdown_data, str(chart_path))
        plot_breakdown(breakdown_data, str(stable_chart_path))
        size_kb = chart_path.stat().st_size / 1024.0
        if size_kb < 20.0:
            logger.warning(
                "Chart generated at %s is suspiciously small (%.1f KB < 20 KB); image may be blank or corrupted",
                chart_path,
                size_kb,
            )
        logger.info("Charts generated: %s (%.1f KB) (and stable memory_breakdown.png)", chart_path, size_kb)

    # 10. Human-Readable Report
    print("\n" + "=" * 68)
    print(f"  M3 GPU Memory Breakdown Report  [{request_id}]")
    print("=" * 68)
    print(f"  Model       : {model_name}")
    print(f"  Device      : cuda:{args.device_index}")
    print(f"  Tokens      : {args.prompt_tokens} prompt -> {args.output_tokens} output")
    print(f"  Total VRAM  : {gpu_total_bytes / (1024**2):.1f} MiB")
    print(f"  NVML baseline after model load, before request: {nvml_idle_bytes / (1024**2):.1f} MiB")
    print("-" * 68)
    print(f"  {'Component':<15} {'Prefill (MiB)':<18} {'Decode (MiB)':<18}")
    print("-" * 68)
    for comp in CONTRACT_COMPONENTS:
        p_val = prefill_comps[comp] / (1024**2)
        d_val = decode_comps[comp] / (1024**2)
        print(f"  {comp:<15} {p_val:>12.2f}       {d_val:>12.2f}")
    print("-" * 68)
    print(f"  {'NVML Total':<15} {prefill_raw['nvml_used'] / (1024**2):>12.2f}       {decode_raw['nvml_used'] / (1024**2):>12.2f}")
    print(f"  {'Peak Alloc':<15} {prefill_raw['peak_alloc'] / (1024**2):>12.2f}       {decode_raw['peak_alloc'] / (1024**2):>12.2f}")
    print("=" * 68)
    print("  Diagnostics (Allocation & Reservation Tracker):")
    for label, val in [
        ("Allocated after load", diagnostics.get("allocated_after_load_bytes", 0)),
        ("Allocated after warmup", diagnostics.get("allocated_after_warmup_bytes", 0)),
        ("Prefill start allocated", diagnostics.get("prefill_allocated_at_start_bytes", 0)),
        ("Decode start allocated", diagnostics.get("decode_allocated_at_start_bytes", 0)),
        ("Resident beyond weights", diagnostics.get("resident_beyond_weights_bytes", 0)),
        ("Reserved at prefill end", diagnostics.get("reserved_at_prefill_end_bytes", 0)),
        ("Reserved at decode end", diagnostics.get("reserved_at_decode_end_bytes", 0)),
    ]:
        print(f"    {label:<28}: {val / (1024**2):8.2f} MiB")
    print("-" * 68)
    print(f"  KV Cache/token : measured={measured_per_token:.1f} B | analytical={analytical_per_token} B (Match: {kv_matches})")
    print(f"  Free VRAM      : {evidence_data['nvml_free_after_decode_bytes'] / (1024**2):.1f} MiB")
    print(f"  KV capacity est: ~{evidence_data['kv_tokens_that_fit_in_free_memory_estimate']:,} tokens fit in remaining VRAM")
    print("=" * 68 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
