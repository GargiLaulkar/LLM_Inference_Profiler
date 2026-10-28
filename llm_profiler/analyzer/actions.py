"""Evidence-based remediation actions for measured latency bottlenecks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DOMINANT_STAGE_SHARE = 0.50
LOW_GPU_UTILIZATION_PERCENT = 50.0
HIGH_GPU_POWER_LIMIT_SHARE = 0.80

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

    if stage == "decode" and batching_enabled is False:
        notes.append("batching is off")

    if stage in ("prefill", "decode") and memory_data:
        phases = memory_data.get("phases", {})
        phase = phases.get(stage, {})
        components = {
            key: float(phase.get(key, 0.0))
            for key in ("weights", "kv_cache", "activations", "other")
        }
        measured_total = sum(components.values())
        if measured_total > 0:
            shares = {
                key: value / measured_total
                for key, value in components.items()
            }
            if stage == "prefill":
                top_components = sorted(
                    components.items(), key=lambda item: item[1], reverse=True
                )[:2]
                top_share_text = ", ".join(
                    f"{name} {shares[name]:.1%}" for name, _ in top_components
                )
                notes.append(
                    f"prefill memory: {top_share_text}; "
                    f"activations {shares['activations']:.1%}; "
                    f"KV cache {shares['kv_cache']:.1%}"
                )
            else:
                top_components = sorted(
                    components.items(), key=lambda item: item[1], reverse=True
                )[:2]
                top_share_text = ", ".join(
                    f"{name} {shares[name]:.1%}" for name, _ in top_components
                )
                notes.append(
                    f"decode memory: {top_share_text}; KV cache {shares['kv_cache']:.1%}"
                )

    return notes


def build_suggested_action(
    stage: str,
    share: float,
    gpu_metrics: Optional[Dict[str, Any]] = None,
    memory_data: Optional[Dict[str, Any]] = None,
    batching_enabled: Optional[bool] = None,
) -> str:
    """Return remediation text; supporting evidence is reported separately."""
    return SUGGESTED_ACTIONS.get(
        stage,
        "Profile server traces and adjust the measured bottleneck's resource allocation.",
    )
