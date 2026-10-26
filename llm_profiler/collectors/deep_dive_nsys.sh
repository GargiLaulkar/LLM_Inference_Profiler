#!/usr/bin/env bash
# deep_dive_nsys.sh — Nsight Systems wrapper for M3 deep-dive profiling.
#
# Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks
# Role: M3 (GPU and Deep-Dive Lead)
#
# Usage:
#   bash llm_profiler/collectors/deep_dive_nsys.sh [deep_dive.py options]
#
# Examples:
#   bash llm_profiler/collectors/deep_dive_nsys.sh
#   bash llm_profiler/collectors/deep_dive_nsys.sh --prompt-tokens 256 --output-tokens 128
#   bash llm_profiler/collectors/deep_dive_nsys.sh --model Qwen/Qwen2.5-0.5B-Instruct
#   OUT_DIR=results/my_run bash llm_profiler/collectors/deep_dive_nsys.sh
#
# Prerequisites:
#   - nsys (Nsight Systems CLI) must be on PATH.
#     Download: https://developer.nvidia.com/nsight-systems
#   - torch and transformers must be installed in the active Python env.
#   - Stop any vLLM container first:  make down
#     (torch.profiler and nsys both use CUPTI — cannot run together)
#
# Outputs written to OUT_DIR (default: results/deep_dive):
#   <request_id>_nsys.nsys-rep   — Nsight Systems report (open in the GUI)
#   <request_id>_stages.jsonl    — per-stage timing (contract schema)
#   <request_id>_summary.json    — aggregate statistics
#
# Reading the report:
#   nsys stats <request_id>_nsys.nsys-rep          # text summary in terminal
#   Open the GUI → File → Open → <request_id>_nsys.nsys-rep
#   Look at the "NVTX" row — coloured ranges show tokenization/prefill/decode/detokenization.

set -euo pipefail

# ── Check nsys is available ────────────────────────────────────────────────────
if ! command -v nsys &>/dev/null; then
    echo "ERROR: 'nsys' not found on PATH." >&2
    echo "Install Nsight Systems from: https://developer.nvidia.com/nsight-systems" >&2
    echo "Typical path on Linux: /opt/nvidia/nsight-systems/*/bin/nsys" >&2
    echo "Add it to PATH, then re-run this script." >&2
    exit 1
fi

# ── Check Python / torch are available ────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON="python"
fi

if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not importable by '$PYTHON'." >&2
    echo "Activate the project venv or install torch." >&2
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_DIVE_PY="${SCRIPT_DIR}/deep_dive.py"
OUT_DIR="${OUT_DIR:-results/deep_dive}"
REQUEST_ID="dd-$(date +%s)"

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "  M3 Deep-Dive — Nsight Systems"
echo "========================================================================"
echo "  nsys     : $(nsys --version 2>&1 | head -1)"
echo "  Python   : $PYTHON ($("$PYTHON" --version 2>&1))"
echo "  Script   : $DEEP_DIVE_PY"
echo "  Out dir  : $OUT_DIR"
echo "  Req ID   : $REQUEST_ID"
echo "========================================================================"
echo ""
echo "  ⚠  IMPORTANT: This script uses CUPTI via nsys."
echo "     torch.profiler and nsys CANNOT run simultaneously."
echo "     Make sure no other profiling tool is active."
echo ""

REPORT_BASE="${OUT_DIR}/${REQUEST_ID}_nsys"

echo "Running: nsys profile ... python deep_dive.py --mode nsys $*"
echo ""

nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output="${REPORT_BASE}" \
    "$PYTHON" "$DEEP_DIVE_PY" \
        --mode nsys \
        --request-id "$REQUEST_ID" \
        --out-dir "$OUT_DIR" \
        "$@"

echo ""
echo "========================================================================"
echo "  Outputs"
echo "========================================================================"
echo "  Nsight report : ${REPORT_BASE}.nsys-rep"
echo "  Stage trace   : ${OUT_DIR}/${REQUEST_ID}_stages.jsonl"
echo "  Summary       : ${OUT_DIR}/${REQUEST_ID}_summary.json"
echo ""
echo "  Text summary (run in terminal):"
echo "    nsys stats ${REPORT_BASE}.nsys-rep"
echo ""
echo "  GUI:"
echo "    Open Nsight Systems → File → Open → ${REPORT_BASE}.nsys-rep"
echo "    Look at the NVTX row: coloured ranges = tokenization/prefill/decode/detokenization"
echo "========================================================================"
