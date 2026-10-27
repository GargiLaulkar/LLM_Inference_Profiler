"""Evidence-based remediation actions for measured latency bottlenecks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DOMINANT_STAGE_SHARE = 0.50
LOW_GPU_UTILIZATION_PERCENT = 50.0
HIGH_GPU_POWER_LIMIT_SHARE = 0.80
KV_CACHE_PRESSURE_SHARE = 0.50

SUGGESTED_ACTIONS: Dict[str, str] = {
    "prefill": (
        "Enable chunked prefill (--enable-chunked-prefill), use prompt prefix caching "
        "(--enable-prefix-caching), optimize prompt length, or scale tensor parallelism."
    ),
    "decode": (
        "Enable speculative decoding, optimize KV cache block size, raise tensor parallelism, "
        "or restrict generation max_tokens to reduce autoregressive memory-bandwidth pressure."
    ),
    "queue_wait": (
        "Increase serving concurrency (--max-num-seqs), scale out server replicas, optimize "
        "KV cache allocation, or apply client rate limiting."
    ),
    "tokenization": (
        "Use a fast Rust-based tokenizer, cache pre-tokenized prompts, or move tokenization "
        "off the serving critical path."
    ),
    "detokenization": (
        "Stream incremental tokens, batch UTF-8 detokenization, or offload detokenization to the client."
    ),
    "network": (
        "Colocate client and server, enable persistent HTTP keep-alive connections, or use gRPC / HTTP/2."
    ),
    "total": "Investigate cumulative end-to-end latency across compute and communication stages.",
}


def build_evidence_notes(
    stage: str,
    share: float,
    gpu_metrics: Optional[Dict[str, Any]] = None,
    memory_data: Optional[Dict[str, Any]] = None,
    batching_enabled: Optional[bool] = None,
) -> List[str]:
    """Return only evidence rules supported by supplied measurements and context."""
    notes: List[str] = []
    gpu_metrics = gpu_metrics or {}

    if (
        stage == "queue_wait"
        and share >= DOMINANT_STAGE_SHARE
        and gpu_metrics.get("gpu_utilization_percent") is not None
        and float(gpu_metrics["gpu_utilization_percent"]) < LOW_GPU_UTILIZATION_PERCENT
    ):
        utilization = float(gpu_metrics["gpu_utilization_percent"])
        notes.append(
            f"high queue_wait share ({share:.1%}) with low GPU utilization ({utilization:.1f}%)"
        )

    if (
        stage == "decode"
        and share >= DOMINANT_STAGE_SHARE
        and gpu_metrics.get("gpu_power_watts") is not None
        and gpu_metrics.get("gpu_power_limit_watts") is not None
    ):
        power = float(gpu_metrics["gpu_power_watts"])
        power_limit = float(gpu_metrics["gpu_power_limit_watts"])
        if power_limit > 0 and power / power_limit >= HIGH_GPU_POWER_LIMIT_SHARE:
            notes.append(
                f"high decode share ({share:.1%}) with GPU power at {power / power_limit:.1%} "
                f"of its measured power limit ({power:.1f}W/{power_limit:.1f}W)"
            )

    if stage == "decode" and batching_enabled is False and memory_data:
        decode_memory = memory_data.get("phases", {}).get("decode", {})
        values = {
            key: float(decode_memory.get(key, 0.0))
            for key in ("weights", "kv_cache", "activations", "other")
        }
        measured_total = sum(values.values())
        kv_cache_share = values["kv_cache"] / measured_total if measured_total > 0 else 0.0
        if values["kv_cache"] > 0:
            note = (
                f"batching is off; decode KV cache is {values['kv_cache'] / (1024 * 1024):.1f} MiB "
                f"({kv_cache_share:.1%} of measured decode memory)"
            )
            if kv_cache_share >= KV_CACHE_PRESSURE_SHARE:
                note += " and is the largest measured decode-memory component"
            notes.append(note)

    return notes


def build_suggested_action(
    stage: str,
    share: float,
    gpu_metrics: Optional[Dict[str, Any]] = None,
    memory_data: Optional[Dict[str, Any]] = None,
    batching_enabled: Optional[bool] = None,
) -> str:
    """Build a remediation tied to the measured stage share and supporting evidence."""
    action = SUGGESTED_ACTIONS.get(
        stage,
        "Profile server traces and adjust the measured bottleneck's resource allocation.",
    )
    evidence = build_evidence_notes(
        stage,
        share,
        gpu_metrics=gpu_metrics,
        memory_data=memory_data,
        batching_enabled=batching_enabled,
    )
    result = f"{stage} share {share:.1%}: {action}"
    if evidence:
        result += " Evidence: " + "; ".join(evidence) + "."
    return result
