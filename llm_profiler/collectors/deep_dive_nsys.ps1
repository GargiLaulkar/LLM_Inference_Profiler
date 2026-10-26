# deep_dive_nsys.ps1 — Nsight Systems wrapper for M3 deep-dive profiling (PowerShell / Windows).
#
# Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks
# Role: M3 (GPU and Deep-Dive Lead)
#
# Usage:
#   .\llm_profiler\collectors\deep_dive_nsys.ps1 [deep_dive.py options]
#
# Examples:
#   .\llm_profiler\collectors\deep_dive_nsys.ps1
#   .\llm_profiler\collectors\deep_dive_nsys.ps1 --prompt-tokens 256 --output-tokens 128
#   .\llm_profiler\collectors\deep_dive_nsys.ps1 --model Qwen/Qwen2.5-0.5B-Instruct
#   $env:OUT_DIR = "results/my_run"; .\llm_profiler\collectors\deep_dive_nsys.ps1
#
# Prerequisites:
#   - nsys (Nsight Systems CLI) must be on PATH.
#     Download: https://developer.nvidia.com/nsight-systems
#     Typical path: C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe
#   - torch and transformers must be installed in the active Python env.
#   - Stop any vLLM container first:  make down  (Linux) or equivalent.
#     torch.profiler and nsys both use CUPTI — they CANNOT run simultaneously.
#
# Outputs written to $OutDir (default: results/deep_dive):
#   <request_id>_nsys.nsys-rep   — Nsight Systems report (open in the GUI)
#   <request_id>_stages.jsonl    — per-stage timing (contract schema)
#   <request_id>_summary.json    — aggregate statistics

[CmdletBinding()]
param(
    # Pass-through arguments to deep_dive.py (e.g. --prompt-tokens 256 --output-tokens 128)
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helper: coloured messages ──────────────────────────────────────────────────
function Write-Header([string]$msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Err([string]$msg)    { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ── Check nsys ────────────────────────────────────────────────────────────────
$nsysCmd = Get-Command "nsys" -ErrorAction SilentlyContinue
if (-not $nsysCmd) {
    Write-Err ("'nsys' not found on PATH.`n" +
               "Install Nsight Systems from: https://developer.nvidia.com/nsight-systems`n" +
               "Typical path: C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe`n" +
               "Add the folder to your PATH, then re-run this script.")
}

# ── Check Python ───────────────────────────────────────────────────────────────
$pyExe = if (Get-Command "python" -ErrorAction SilentlyContinue) { "python" } else { "python3" }
try {
    & $pyExe -c "import torch" 2>&1 | Out-Null
} catch {
    Write-Err "torch not importable by '$pyExe'. Activate the project venv or install torch."
}

# ── Paths ─────────────────────────────────────────────────────────────────────
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$deepDivePy  = Join-Path $scriptDir "deep_dive.py"
$outDir      = if ($env:OUT_DIR) { $env:OUT_DIR } else { "results/deep_dive" }
$requestId   = "dd-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

Write-Host ""
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Header "M3 Deep-Dive — Nsight Systems (PowerShell)"
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Header "nsys     : $(& nsys --version 2>&1 | Select-Object -First 1)"
Write-Header "Python   : $pyExe ($( & $pyExe --version 2>&1 ))"
Write-Header "Script   : $deepDivePy"
Write-Header "Out dir  : $outDir"
Write-Header "Req ID   : $requestId"
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host ""
Write-Host "  ⚠  IMPORTANT: This uses CUPTI via nsys." -ForegroundColor Yellow
Write-Host "     torch.profiler and nsys CANNOT run simultaneously." -ForegroundColor Yellow
Write-Host "     Ensure no other profiling tool is active." -ForegroundColor Yellow
Write-Host ""

$reportBase = Join-Path $outDir "${requestId}_nsys"

$nsysArgs = @(
    "profile",
    "--trace=cuda,nvtx,osrt",
    "--sample=none",
    "--force-overwrite=true",
    "--output=$reportBase",
    $pyExe,
    $deepDivePy,
    "--mode", "nsys",
    "--request-id", $requestId,
    "--out-dir", $outDir
) + $RemainingArgs

Write-Host "Running: nsys $($nsysArgs -join ' ')"
Write-Host ""

& nsys @nsysArgs

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Header "Outputs"
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Header "Nsight report : ${reportBase}.nsys-rep"
Write-Header "Stage trace   : ${outDir}/${requestId}_stages.jsonl"
Write-Header "Summary       : ${outDir}/${requestId}_summary.json"
Write-Host ""
Write-Header "Text summary (run in terminal):"
Write-Host "    nsys stats ${reportBase}.nsys-rep" -ForegroundColor White
Write-Host ""
Write-Header "GUI:"
Write-Host "    Open Nsight Systems → File → Open → ${reportBase}.nsys-rep" -ForegroundColor White
Write-Host "    Look at the NVTX row: coloured ranges = tokenization/prefill/decode/detokenization" -ForegroundColor White
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host ""

exit $exitCode
