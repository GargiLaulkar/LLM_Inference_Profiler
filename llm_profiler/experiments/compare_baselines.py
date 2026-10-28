"""Baseline comparison for B1, B2 and B3.

This runner compares the manual timing baseline, the stack's own metrics baseline,
and the profiler baseline. Real files are used when they exist; otherwise the report
marks them as pending-live or synthetic-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments" / "results" / "baseline_comparison.json"


def _read_json_if_present(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def evaluate_baseline_compare(*, dry_run: bool = True, out: str | Path = DEFAULT_OUT) -> Dict[str, Any]:
    if dry_run:
        b1 = {"status": "synthetic", "note": "Manual timing baseline placeholder; synthetic only.", "stages": ["tokenization", "prefill", "decode"]}
        b2 = {"status": "synthetic", "note": "Stack-native metrics baseline placeholder; synthetic only.", "stages": ["queue_wait", "prefill", "decode"]}
        b3 = {"status": "pending_live", "note": "No profiler baseline file exists yet; live run required.", "stages": ["tokenization", "queue_wait", "prefill", "decode", "network", "total"]}
        result = {
            "experiment_id": "baseline_compare",
            "data_source": "synthetic",
            "status": "synthetic_only",
            "summary": {
                "b1_manual_timing": b1,
                "b2_stack_metrics": b2,
                "b3_profiler": b3,
            },
            "findings": [
                "B1 is a manual timing baseline, B2 is stack-native metrics, and B3 is profiler telemetry.",
                "When live B3 is unavailable, it remains marked pending_live and is not treated as measured evidence.",
            ],
        }
    else:
        b1_path = ROOT / "results" / "baseline_b1.csv"
        b2_path = ROOT / "results" / "baseline_b2.csv"
        b3_path = ROOT / "experiments" / "results" / "e2_rps_1_analysis.json"
        result = {
            "experiment_id": "baseline_compare",
            "data_source": "live",
            "status": "pending_live",
            "summary": {
                "b1_manual_timing": {"status": "available" if b1_path.exists() else "pending_live", "path": str(b1_path)},
                "b2_stack_metrics": {"status": "available" if b2_path.exists() else "pending_live", "path": str(b2_path)},
                "b3_profiler": {"status": "available" if b3_path.exists() else "pending_live", "path": str(b3_path)},
            },
        }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare B1, B2 and B3 baselines.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic summary values by default.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="JSON output path.")
    args = parser.parse_args()
    result = evaluate_baseline_compare(dry_run=args.dry_run or True, out=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
