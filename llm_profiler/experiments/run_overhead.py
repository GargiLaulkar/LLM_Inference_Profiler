"""Experiment E6: isolate profiler overhead using the same gateway and workload.

Profiler OFF is a pass-through gateway/server with stage timing, trace writing,
and metric ingest disabled. Profiler ON enables all three. Both measurements
must traverse the same gateway; the optional direct-vLLM comparison is reported
separately as gateway overhead, not profiler overhead.

The current stack has no LLMPROF_PROFILER_ENABLED switch, so live E6 remains
pending until M2 implements the requested 0|1 server-side toggle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments" / "results" / "e6_overhead.json"

PROFILER_MODE_DEFINITION = {
    "off": "Same gateway/server in pass-through mode; stage timing, trace writing, and metric ingest disabled.",
    "on": "Same gateway/server with stage timing, trace writing, and metric ingest enabled.",
}


def evaluate_overhead(*, dry_run: bool = True, compare_direct: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {
            "experiment_id": "E6",
            "name": "Profiler overhead on vs off",
            "data_source": "synthetic",
            "status": "not_evaluated",
            "benchmark_type": "dry_run_validation_only",
            "profiler_modes": PROFILER_MODE_DEFINITION,
            "measurement": None,
            "pass_fail": "not_evaluated",
            "target_overhead_percent": 5.0,
            "gateway_overhead_comparison": (
                {
                    "status": "not_evaluated",
                    "comparison": "direct-to-vLLM versus through-the-same-gateway",
                }
                if compare_direct
                else None
            ),
            "message": "SYNTHETIC - not an E6 result",
            "notes": [
                "Dry-run validates output shape only; it does not generate or report performance measurements."
            ],
        }

    pending = {
        "experiment_id": "E6",
        "name": "Profiler overhead on vs off",
        "data_source": "live",
        "status": "pending_live",
        "profiler_modes": PROFILER_MODE_DEFINITION,
        "pass_fail": "not_evaluated",
        "target_overhead_percent": 5.0,
        "pending_reason": (
            "The gateway/server does not expose LLMPROF_PROFILER_ENABLED=0|1 yet; "
            "no live on/off measurements were taken."
        ),
        "gateway_overhead_comparison": None,
    }
    if compare_direct:
        pending["gateway_overhead_comparison"] = {
            "status": "pending_live",
            "comparison": "direct-to-vLLM versus through-the-same-gateway",
            "purpose": "Separately measure gateway overhead; do not include it in profiler overhead.",
        }
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate E6 output shape or report live E6 as pending until profiler toggle exists."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate output shape without measurements.")
    mode.add_argument("--live", action="store_true", help="Report live E6 readiness; does not launch a server.")
    parser.add_argument(
        "--compare-direct",
        action="store_true",
        help="Separately report the optional direct-vLLM versus gateway overhead comparison.",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="JSON output path.")
    args = parser.parse_args()

    result = evaluate_overhead(dry_run=not args.live, compare_direct=args.compare_direct)
    if args.live:
        print(f"[PENDING LIVE] {result['pending_reason']}")
        if args.compare_direct:
            print("[PENDING LIVE] Direct-to-vLLM versus gateway comparison is separate gateway-overhead work.")
    else:
        print("SYNTHETIC - not an E6 result")
        print("PASS/FAIL: NOT_EVALUATED (dry-run validates output shape only)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f"[INFO] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
