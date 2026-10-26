#!/usr/bin/env python3
"""M3 Deep-Dive: Single-Request GPU Profiling for LLM Inference.

Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks
Role: M3 (GPU and Deep-Dive Lead)

Purpose:
    Profile ONE synthetic request through a local HuggingFace model in detail,
    measuring per-stage wall-clock and GPU time (tokenization, prefill, decode,
    detokenization). Produces Chrome-format traces (torch mode) or NVTX ranges
    (nsys mode) for kernel-level analysis.

⚠️  OVERHEAD WARNING: Profiling (torch.profiler or NVTX) adds significant runtime
    overhead and MUST NOT run concurrently with always-on benchmark collection.
    Stop any vLLM container and active load generator BEFORE running this script.
    On a 6 GB GPU (e.g. RTX 3060) the vLLM container should be stopped first:
        make down       # or: docker compose -f docker/docker-compose.yml down

    torch.profiler and Nsight Systems (nsys) BOTH use the CUPTI profiling
    interface and CANNOT run simultaneously—doing so causes CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED.

Design:
    - Zero intra-repo dependencies: stdlib + lazy torch/transformers + optional pyyaml.
    - torch and transformers are imported LAZILY inside functions so --help works
      without them installed (laptop development workflow).
    - Stage names follow the shared contract: tokenization, prefill, decode, detokenization.

Usage:
    # torch.profiler mode (default) — Chrome trace + kernel table:
    python llm_profiler/collectors/deep_dive.py --mode torch

    # nsys-friendly mode — NVTX ranges, no torch.profiler:
    python llm_profiler/collectors/deep_dive.py --mode nsys

    # Custom parameters:
    python llm_profiler/collectors/deep_dive.py \\
        --mode torch --prompt-tokens 256 --output-tokens 128 \\
        --warmup 2 --out-dir results/deep_dive

    # For Kaggle / Colab (no config.yaml available):
    python deep_dive.py --model Qwen/Qwen2.5-0.5B-Instruct --mode torch
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ─── CONSTANTS (single source of truth, no magic numbers elsewhere) ─────────
DEFAULT_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT_TOKENS: int = 128
DEFAULT_OUTPUT_TOKENS: int = 64
DEFAULT_WARMUP: int = 1
DEFAULT_OUT_DIR: str = "results/deep_dive"
DEFAULT_DEVICE: str = "auto"
DEFAULT_DEVICE_INDEX: int = 0
CONTRACT_STAGES: List[str] = ["tokenization", "prefill", "decode", "detokenization"]

# config.yaml search paths (relative to cwd)
CONFIG_SEARCH_PATHS: List[str] = [
    os.path.join("llm_profiler", "config.yaml"),
    "config.yaml",
    os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
]

logger = logging.getLogger("deep_dive")


# ─── CONFIG LOADING ──────────────────────────────────────────────────────────

def load_model_from_config() -> Optional[str]:
    """Try to read model_name from config.yaml, return None if not found."""
    # Optional yaml dependency
    try:
        import yaml  # type: ignore
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
                    return model
            except Exception as exc:
                logger.debug("Failed to parse config at '%s': %s", path, exc)
    return None


# ─── PROMPT BUILDER ──────────────────────────────────────────────────────────

def build_prompt(
    tokenizer: Any,
    target_tokens: int,
    seed_word: str = "inference",
) -> Tuple[str, List[int]]:
    """Build a synthetic prompt of EXACTLY target_tokens input tokens.

    Repeats seed_word until we have enough raw tokens, then truncates or
    extends to hit the exact token count.

    Args:
        tokenizer: Any tokenizer with encode/decode/convert_tokens_to_ids.
        target_tokens: Exact number of input tokens required.
        seed_word: Simple word to repeat.

    Returns:
        Tuple of (prompt_text, token_ids).
    """
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")

    # Over-generate then trim
    candidate = (seed_word + " ") * (target_tokens * 2)
    ids = tokenizer.encode(candidate, add_special_tokens=False)

    if len(ids) >= target_tokens:
        ids = ids[:target_tokens]
    else:
        # Pad with a pad token if shorter (edge case on tiny vocab tokenizers)
        pad_id = getattr(tokenizer, "pad_token_id", None) or 0
        ids = ids + [pad_id] * (target_tokens - len(ids))

    text = tokenizer.decode(ids, skip_special_tokens=False)
    return text, ids


# ─── STAGE CONTEXT MANAGER ───────────────────────────────────────────────────

@contextlib.contextmanager
def stage(
    name: str,
    mode: str,
    cuda_available: bool,
    records: List[Dict[str, Any]],
) -> Iterator[None]:
    """Context manager that times a profiling stage and records the result.

    In 'torch' mode: wraps the block in torch.profiler.record_function(name).
    In 'nsys' mode:  pushes/pops an NVTX range with torch.cuda.nvtx.
    In both modes:   calls torch.cuda.synchronize() BEFORE reading end time on
                     GPU stages (prefill, decode) so GPU kernel time is captured.

    Args:
        name: Stage name (must be one of CONTRACT_STAGES).
        mode: 'torch' or 'nsys'.
        cuda_available: Whether CUDA is accessible.
        records: List to append the stage record dict into.
    """
    assert name in CONTRACT_STAGES, f"Unknown stage '{name}', must be one of {CONTRACT_STAGES}"

    is_cuda_stage = name in ("prefill", "decode")

    if mode == "torch":
        import torch  # lazy
        rf = torch.profiler.record_function(name)
        rf.__enter__()
    elif mode == "nsys" and cuda_available:
        import torch  # lazy
        torch.cuda.nvtx.range_push(name)

    start_ts = time.time()
    t0 = time.perf_counter()

    try:
        yield
    finally:
        # Synchronize GPU before stopping the clock for CUDA stages
        if is_cuda_stage and cuda_available:
            import torch  # lazy
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        end_ts = time.time()

        if mode == "torch":
            rf.__exit__(None, None, None)  # type: ignore[union-attr]
        elif mode == "nsys" and cuda_available:
            torch.cuda.nvtx.range_pop()  # type: ignore

        duration_s = t1 - t0
        records.append({
            "stage": name,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_s": duration_s,
        })


# ─── CORE REQUEST RUNNER ─────────────────────────────────────────────────────

def run_request(
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt_tokens: int,
    output_tokens: int,
    mode: str,
    cuda_available: bool,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Run a single synthetic request through the model and time all four stages.

    Args:
        model: HuggingFace CausalLM model (already loaded and on device).
        tokenizer: HuggingFace tokenizer.
        device: torch.device.
        prompt_tokens: Exact number of input tokens.
        output_tokens: Exact number of output tokens to generate.
        mode: 'torch' or 'nsys'.
        cuda_available: Whether CUDA is available.

    Returns:
        Tuple of:
            - List of stage record dicts (4 items, in contract order).
            - List of per-token decode durations in seconds.
    """
    import torch  # lazy

    records: List[Dict[str, Any]] = []
    decode_token_s: List[float] = []

    # ── 1. Tokenization ───────────────────────────────────────────────────────
    with stage("tokenization", mode, cuda_available, records):
        _, token_ids = build_prompt(tokenizer, prompt_tokens)
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)

    # ── 2. Prefill: forward pass over entire prompt, use_cache=True ──────────
    past_key_values = None
    next_token_id = None
    with stage("prefill", mode, cuda_available, records):
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]          # last token logits
        next_token_id = logits.argmax(dim=-1)       # greedy
        past_key_values = outputs.past_key_values

    # ── 3. Decode: one token at a time, reusing KV cache ─────────────────────
    with stage("decode", mode, cuda_available, records):
        generated_ids: List[int] = [next_token_id.item()]

        # We already have the first token from prefill; generate the rest
        for _ in range(output_tokens - 1):
            t_tok_start = time.perf_counter()
            if cuda_available:
                torch.cuda.synchronize()

            with torch.inference_mode():
                step_out = model(
                    input_ids=next_token_id.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            next_token_id = step_out.logits[:, -1, :].argmax(dim=-1)
            past_key_values = step_out.past_key_values
            generated_ids.append(next_token_id.item())

            if cuda_available:
                torch.cuda.synchronize()
            t_tok_end = time.perf_counter()
            decode_token_s.append(t_tok_end - t_tok_start)

    # ── 4. Detokenization ─────────────────────────────────────────────────────
    with stage("detokenization", mode, cuda_available, records):
        _ = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return records, decode_token_s


# ─── OUTPUT WRITERS ──────────────────────────────────────────────────────────

def write_stage_jsonl(
    out_path: str,
    request_id: str,
    stage_records: List[Dict[str, Any]],
) -> None:
    """Write per-stage timing records as JSON lines (contract schema).

    Each line has exactly: request_id, stage, start_ts, end_ts, duration_s.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in stage_records:
            line = {
                "request_id": request_id,
                "stage": rec["stage"],
                "start_ts": rec["start_ts"],
                "end_ts": rec["end_ts"],
                "duration_s": rec["duration_s"],
            }
            f.write(json.dumps(line) + "\n")


def summarize(
    request_id: str,
    mode: str,
    model_name: str,
    device_name: str,
    prompt_tokens: int,
    output_tokens: int,
    stage_records: List[Dict[str, Any]],
    decode_token_s: List[float],
) -> Dict[str, Any]:
    """Build a summary dictionary with per-stage and aggregate statistics.

    Args:
        request_id: Unique request identifier.
        mode: 'torch' or 'nsys'.
        model_name: Model identifier string.
        device_name: str(torch.device) or equivalent.
        prompt_tokens: Number of input tokens.
        output_tokens: Number of generated tokens.
        stage_records: List of {stage, start_ts, end_ts, duration_s} dicts.
        decode_token_s: Per-token decode durations.

    Returns:
        Summary dict suitable for JSON serialization.
    """
    # Lazy torch version retrieval
    try:
        import torch  # lazy
        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed"

    stage_duration: Dict[str, float] = {rec["stage"]: rec["duration_s"] for rec in stage_records}
    total_s = sum(stage_duration.values())

    # Decode token statistics
    mean_decode = 0.0
    p95_decode = 0.0
    if decode_token_s:
        mean_decode = sum(decode_token_s) / len(decode_token_s)
        sorted_times = sorted(decode_token_s)
        p95_idx = max(0, math.ceil(0.95 * len(sorted_times)) - 1)
        p95_decode = sorted_times[p95_idx]

    return {
        "request_id": request_id,
        "mode": mode,
        "model": model_name,
        "device": device_name,
        "torch_version": torch_version,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "stage_duration_s": stage_duration,
        "total_s": total_s,
        "decode_token_s": decode_token_s,
        "mean_decode_token_s": mean_decode,
        "p95_decode_token_s": p95_decode,
    }


# ─── MAIN ENTRYPOINT ─────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "M3 Deep-Dive: Profile a single LLM request with torch.profiler or "
            "Nsight Systems NVTX ranges."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["torch", "nsys"],
        default="torch",
        help="Profiling mode: 'torch' uses torch.profiler; 'nsys' uses NVTX ranges",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID (default: read from config.yaml, then fall back to constant)",
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
        help="Number of warmup passes before the profiled run",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where output files are written",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Request identifier (default: auto-generated as dd-<epoch>)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Device: 'auto', 'cuda', 'cpu', or 'cuda:<N>'",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=DEFAULT_DEVICE_INDEX,
        help="CUDA device index when --device is auto or cuda",
    )
    return parser.parse_args(argv)


def resolve_device(device_arg: str, device_index: int) -> Tuple[Any, bool]:
    """Resolve device string to torch.device and check CUDA availability.

    Returns (torch.device, cuda_is_available).
    """
    import torch  # lazy

    cuda_avail = torch.cuda.is_available()
    if device_arg == "auto":
        if cuda_avail:
            dev = torch.device(f"cuda:{device_index}")
            logger.info("Device resolved to %s (CUDA available)", dev)
        else:
            dev = torch.device("cpu")
            logger.warning(
                "⚠  CUDA not available — running on CPU. "
                "Timings will not reflect GPU performance."
            )
        return dev, cuda_avail
    else:
        dev = torch.device(device_arg)
        return dev, (dev.type == "cuda" and cuda_avail)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)

    # ── Overhead warning ──────────────────────────────────────────────────────
    logger.warning(
        "⚠  DEEP-DIVE MODE: profiling overhead is intentional and significant. "
        "Stop the vLLM container (make down) and any active load generator before "
        "running this script. Do NOT collect always-on benchmark numbers while "
        "deep-dive is active."
    )

    # ── Resolve model name ────────────────────────────────────────────────────
    model_name = args.model
    if model_name is None:
        model_name = load_model_from_config()
    if model_name is None:
        model_name = DEFAULT_MODEL
        logger.info("Using default model: %s", model_name)

    # ── Resolve request-id ────────────────────────────────────────────────────
    request_id = args.request_id or f"dd-{int(time.time())}"

    # ── Prepare output directory ──────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir.resolve())

    # ── Now import torch (will fail loudly if not installed) ─────────────────
    try:
        import torch
    except ImportError:
        logger.error(
            "torch is required to run deep_dive.py. "
            "Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        return 1

    device, cuda_available = resolve_device(args.device, args.device_index)

    # ── Load model and tokenizer ──────────────────────────────────────────────
    logger.info("Loading tokenizer and model '%s'…", model_name)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    dtype = torch.float16 if cuda_available else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,  # explicit placement
    ).to(device).eval()

    device_name = str(device)

    # ── Warmup runs (not profiled) ────────────────────────────────────────────
    if args.warmup > 0:
        logger.info("Running %d warmup pass(es)…", args.warmup)
        for i in range(args.warmup):
            logger.debug("Warmup %d/%d", i + 1, args.warmup)
            run_request(
                model, tokenizer, device,
                args.prompt_tokens, args.output_tokens,
                mode="torch",  # no profiler active during warmup
                cuda_available=cuda_available,
            )
        logger.info("Warmup complete.")

    # ── Profiled run ──────────────────────────────────────────────────────────
    logger.info("Starting profiled run (mode=%s, request_id=%s)…", args.mode, request_id)

    if args.mode == "torch":
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda_available:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            profile_memory=True,
            record_shapes=False,
            with_stack=False,
        ) as prof:
            stage_records, decode_token_s = run_request(
                model, tokenizer, device,
                args.prompt_tokens, args.output_tokens,
                mode="torch",
                cuda_available=cuda_available,
            )

        # ── Write torch outputs ───────────────────────────────────────────────
        trace_path = out_dir / f"{request_id}_trace.json"
        prof.export_chrome_trace(str(trace_path))
        logger.info("Chrome trace: %s", trace_path)

        # Kernel table: prefer cuda_time_total, fall back to device_time_total
        sort_key = "cuda_time_total" if cuda_available else "cpu_time_total"
        kernels_path = out_dir / f"{request_id}_kernels.txt"
        try:
            table = prof.key_averages().table(sort_by=sort_key, row_limit=25)
        except Exception:
            table = prof.key_averages().table(row_limit=25)
        with open(kernels_path, "w", encoding="utf-8") as f:
            f.write(table)
        logger.info("Kernel table: %s", kernels_path)

    else:
        # nsys mode: just run — the shell wrapper invokes `nsys profile`
        stage_records, decode_token_s = run_request(
            model, tokenizer, device,
            args.prompt_tokens, args.output_tokens,
            mode="nsys",
            cuda_available=cuda_available,
        )

    # ── Write stage JSONL (both modes) ────────────────────────────────────────
    jsonl_path = out_dir / f"{request_id}_stages.jsonl"
    write_stage_jsonl(str(jsonl_path), request_id, stage_records)
    logger.info("Stage trace: %s", jsonl_path)

    # ── Write summary JSON (both modes) ──────────────────────────────────────
    summary = summarize(
        request_id=request_id,
        mode=args.mode,
        model_name=model_name,
        device_name=device_name,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        stage_records=stage_records,
        decode_token_s=decode_token_s,
    )
    summary_path = out_dir / f"{request_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", summary_path)

    # ── Human-readable per-stage report ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Deep-Dive Report  [{request_id}]")
    print("=" * 60)
    print(f"  Model    : {model_name}")
    print(f"  Device   : {device_name}")
    print(f"  Mode     : {args.mode}")
    print(f"  In/Out   : {args.prompt_tokens} → {args.output_tokens} tokens")
    print()
    for rec in stage_records:
        print(f"  {rec['stage']:<16s}: {rec['duration_s'] * 1000:8.2f} ms")
    print(f"  {'TOTAL':<16s}: {summary['total_s'] * 1000:8.2f} ms")
    if decode_token_s:
        print(f"\n  Decode mean   : {summary['mean_decode_token_s'] * 1000:.2f} ms/token")
        print(f"  Decode p95    : {summary['p95_decode_token_s'] * 1000:.2f} ms/token")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
